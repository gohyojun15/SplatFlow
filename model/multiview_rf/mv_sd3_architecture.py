from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention import JointTransformerBlock, _chunked_feed_forward
from diffusers.models.attention_processor import Attention, AttentionProcessor
from diffusers.models.embeddings import CombinedTimestepTextProjEmbeddings, PatchEmbed, get_3d_sincos_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from einops import rearrange

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class JointTransformerBlockMultiView(JointTransformerBlock):
    def __init__(self, num_views, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_views = num_views

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_view(self):
        for param in self.view_layernorm.parameters():
            param.requires_grad = True
        for param in self.view_adaln.parameters():
            param.requires_grad = True
        for param in self.view_attn.parameters():
            param.requires_grad = True
        for param in self.view_fc.parameters():
            param.requires_grad = True

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
    ):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        # Normal block
        if self.context_pre_only:
            norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)
        else:
            norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
                encoder_hidden_states, emb=temb
            )

        K, N, M = hidden_states.shape
        B = K // self.num_views
        norm_hidden_states = rearrange(norm_hidden_states, "(b v) n m -> b (v n) m", b=B, v=self.num_views, n=N, m=M)
        norm_encoder_hidden_states = rearrange(
            norm_encoder_hidden_states, "(b v) n m -> b (v n) m", b=B, v=self.num_views
        )
        # Attention.
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states, encoder_hidden_states=norm_encoder_hidden_states
        )
        attn_output = rearrange(attn_output, "b (v n) m -> (b v) n m", b=B, v=self.num_views)
        context_attn_output = rearrange(context_attn_output, "b (v n) m -> (b v) n m", b=B, v=self.num_views)

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output
        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.
        if self.context_pre_only:
            encoder_hidden_states = None
        else:
            context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
            encoder_hidden_states = encoder_hidden_states + context_attn_output

            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                context_ff_output = _chunked_feed_forward(
                    self.ff_context, norm_encoder_hidden_states, self._chunk_dim, self._chunk_size
                )
            else:
                context_ff_output = self.ff_context(norm_encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        return encoder_hidden_states, hidden_states


class PatchEmbedMultiView(PatchEmbed):
    def __init__(
        self,
        height: int = 224,
        width: int = 224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        layer_norm=False,
        flatten=True,
        bias=True,
        interpolation_scale=1,
        pos_embed_type="sincos",
        pos_embed_max_size=None,  # For SD3 cropping
        num_views=8,
    ):
        super().__init__(
            height=height,
            width=width,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            layer_norm=layer_norm,
            flatten=flatten,
            bias=bias,
            interpolation_scale=interpolation_scale,
            pos_embed_type=pos_embed_type,
            pos_embed_max_size=pos_embed_max_size,
        )
        self.num_views = num_views
        pos_embed = get_3d_sincos_pos_embed(
            embed_dim=embed_dim,
            spatial_size=(16, 16),
            temporal_size=num_views,
            spatial_interpolation_scale=1.0,
            temporal_interpolation_scale=1.0,
        )
        self.register_buffer("pos_embed", torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward(self, latent):
        latent = self.proj(latent)
        if self.flatten:
            latent = latent.flatten(2).transpose(1, 2)  # BCHW -> BNC
        if self.layer_norm:
            latent = self.norm(latent)
        if self.pos_embed is None:
            return latent.to(latent.dtype)
        pos_embed = self.pos_embed
        b = latent.shape[0] // self.num_views
        view = self.num_views
        latent = rearrange(latent, "(b v) n m -> b v n m", b=b, v=view)
        latent = latent + pos_embed.to(latent.dtype)
        latent = rearrange(latent, "b v n m -> (b v) n m", b=b, v=view)
        return latent


class MultiViewSD3Transformer(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    """
    The `MultiViewSD3Transformer` model is a multi-view extension of the `StableDiffusion3` model. The model is
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: int = 128,
        patch_size: int = 2,
        in_channels: int = 16,
        num_layers: int = 18,
        attention_head_dim: int = 64,
        num_attention_heads: int = 18,
        joint_attention_dim: int = 4096,
        caption_projection_dim: int = 1152,
        pooled_projection_dim: int = 2048,
        out_channels: int = 16,
        pos_embed_max_size: int = 96,
        num_views: int = 8,
    ):
        super().__init__()
        default_out_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else default_out_channels
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim

        self.pos_embed = PatchEmbedMultiView(
            height=self.config.sample_size,
            width=self.config.sample_size,
            patch_size=self.config.patch_size,
            in_channels=16,
            embed_dim=self.inner_dim,
            pos_embed_max_size=pos_embed_max_size,  # hard-code for now.
            num_views=num_views,
        )
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=self.inner_dim, pooled_projection_dim=self.config.pooled_projection_dim
        )

        self.context_embedder = nn.Linear(self.config.joint_attention_dim, self.config.caption_projection_dim)

        # `attention_head_dim` is doubled to account for the mixing.
        # It needs to crafted when we get the actual checkpoints.
        self.transformer_blocks = nn.ModuleList(
            [
                JointTransformerBlockMultiView(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.inner_dim,
                    context_pre_only=i == num_layers - 1,
                    num_views=num_views,
                )
                for i in range(self.config.num_layers)
            ]
        )
        self.patch_size = patch_size
        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)
        self.gradient_checkpointing = False

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_view(self):
        for blk in self.transformer_blocks:
            for param in blk.parameters():
                param.requires_grad = True

    def adjust_output_input_channel_size(self, new_in_channels: int):
        self.config.in_channels = new_in_channels
        self.out_channels = new_in_channels
        old_conv_layer = self.pos_embed.proj

        # Calculate scaling factor
        scaling_factor_conv = (old_conv_layer.in_channels / new_in_channels) ** 0.5

        # Create a new convolutional layer with the desired number of input channels
        new_conv_layer = nn.Conv2d(
            in_channels=new_in_channels,
            out_channels=old_conv_layer.out_channels,
            kernel_size=old_conv_layer.kernel_size,
            stride=old_conv_layer.stride,
            padding=old_conv_layer.padding,
            dilation=old_conv_layer.dilation,
            groups=old_conv_layer.groups,
            bias=old_conv_layer.bias is not None,
        )

        with torch.no_grad():
            channels_to_copy = min(old_conv_layer.in_channels, new_in_channels)

            # Copy existing weights
            new_conv_layer.weight.data[:, :channels_to_copy, :, :] = old_conv_layer.weight.data[
                :, :channels_to_copy, :, :
            ]

            # Copy existing weights to new input channels via modulo indexing
            for i in range(channels_to_copy, new_in_channels):
                idx = i % old_conv_layer.in_channels
                new_conv_layer.weight.data[:, i : i + 1, :, :] = old_conv_layer.weight.data[:, idx : idx + 1, :, :]

            # Scale the weights
            new_conv_layer.weight.mul_(scaling_factor_conv)

            # Copy bias if it exists
            if old_conv_layer.bias is not None:
                new_conv_layer.bias.data = old_conv_layer.bias.data

        # Replace the old convolutional layer with the new one
        self.pos_embed.proj = new_conv_layer

        # Output layer modification
        old_linear_layer = self.proj_out  # Get the original Linear layer

        # Calculate the new output features for the Linear layer
        new_out_features = self.patch_size * self.patch_size * new_in_channels

        # Calculate scaling factor for the linear layer
        scaling_factor_linear = (old_linear_layer.out_features / new_out_features) ** 0.5

        # Create a new Linear layer with the desired output channels
        new_linear_layer = nn.Linear(
            old_linear_layer.in_features,  # Keep the input features the same
            new_out_features,  # New number of output features
            bias=old_linear_layer.bias is not None,
        )

        with torch.no_grad():
            features_to_copy = min(old_linear_layer.out_features, new_out_features)

            # Copy existing weights
            new_linear_layer.weight.data[:features_to_copy, :] = old_linear_layer.weight.data[:features_to_copy, :]

            # Copy existing weights to new outputs via modulo indexing
            for i in range(features_to_copy, new_out_features):
                idx = i % old_linear_layer.out_features
                new_linear_layer.weight.data[i, :] = old_linear_layer.weight.data[idx, :]

            # Scale the weights
            new_linear_layer.weight.mul_(scaling_factor_linear)

            # Copy existing biases
            if old_linear_layer.bias is not None:
                new_linear_layer.bias.data[:features_to_copy] = old_linear_layer.bias.data[:features_to_copy]

                # Copy biases via modulo indexing for new outputs
                for i in range(features_to_copy, new_out_features):
                    idx = i % old_linear_layer.out_features
                    new_linear_layer.bias.data[i] = old_linear_layer.bias.data[idx]

        # Replace the old Linear layer with the new one
        self.proj_out = new_linear_layer

    # Copied from diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.enable_forward_chunking
    def enable_forward_chunking(self, chunk_size: Optional[int] = None, dim: int = 0) -> None:
        """
        Sets the attention processor to use [feed forward
        chunking](https://huggingface.co/blog/reformer#2-chunked-feed-forward-layers).

        Parameters:
            chunk_size (`int`, *optional*):
                The chunk size of the feed-forward layers. If not specified, will run feed-forward layer individually
                over each tensor of dim=`dim`.
            dim (`int`, *optional*, defaults to `0`):
                The dimension over which the feed-forward computation should be chunked. Choose between dim=0 (batch)
                or dim=1 (sequence length).
        """
        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")

        # By default chunk size is 1
        chunk_size = chunk_size or 1

        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, chunk_size, dim)

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor(return_deprecated_lora=True)

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        block_controlnet_hidden_states: List = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`SD3Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        b, view, channel, height, width = hidden_states.shape
        # 2Dfy
        hidden_states = hidden_states.reshape(b * view, channel, height, width)

        hidden_states = self.pos_embed(hidden_states)

        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # Multi-view Patch
        temb = temb.unsqueeze(1).repeat(1, view, 1)
        encoder_hidden_states = encoder_hidden_states.unsqueeze(1).repeat(1, view, 1, 1)

        # 2dfy
        temb = temb.reshape(b * view, temb.shape[-1])
        encoder_hidden_states = encoder_hidden_states.reshape(
            b * view, encoder_hidden_states.shape[-2], encoder_hidden_states.shape[-1]
        )

        for index_block, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    **ckpt_kwargs,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb
                )

            # controlnet residual
            if block_controlnet_hidden_states is not None and block.context_pre_only is False:
                interval_control = len(self.transformer_blocks) // len(block_controlnet_hidden_states)
                hidden_states = hidden_states + block_controlnet_hidden_states[index_block // interval_control]

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # unpatchify
        patch_size = self.config.patch_size
        height = height // patch_size
        width = width // patch_size

        hidden_states = hidden_states.reshape(
            shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
        )

        output = output.reshape(b, view, self.out_channels, height * patch_size, width * patch_size)
        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
