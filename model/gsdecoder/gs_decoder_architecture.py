from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2D, get_up_block
from einops import rearrange, repeat

from .camera_embedding import get_plucker_rays
from .decoder_splatting import DecoderSplatting


class AttnProcessor2_0_modified:
    r"""
    We copied the AttnProcessor2_0 class from the diffusers library
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        num_views: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        hidden_states = rearrange(hidden_states, "(b v) t c -> b (v t) c", v=num_views)
        batch_size = hidden_states.shape[0]
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        hidden_states = rearrange(hidden_states, "b (v t) c -> (b v) t c", v=num_views)
        batch_size = hidden_states.shape[0]

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class GSDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 41,
        out_channels: int = 14,
        up_block_types: Tuple[str, ...] = (
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
        ),
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        cfg=None,
    ):
        super(GSDecoder, self).__init__()
        self.cfg = cfg
        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.gn = nn.GroupNorm(norm_num_groups, block_out_channels[-1], affine=True)
        self.attn_block_pre = nn.ModuleList(
            [
                Attention(
                    block_out_channels[-1],
                    heads=1,
                    dim_head=block_out_channels[-1],
                    rescale_output_factor=1,
                    eps=1e-6,
                    norm_num_groups=32,
                    spatial_norm_dim=None,
                    residual_connection=True,
                    processor=AttnProcessor2_0_modified(),
                    bias=True,
                    upcast_softmax=True,
                    _from_deprecated_attn_block=True,
                    dropout=0.0,
                )
                for _ in range(3)
            ]
        )

        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn="silu",
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=None,
            add_attention=False,
        )

        self.attn_block_post = nn.ModuleList(
            [
                Attention(
                    block_out_channels[-1],
                    heads=1,
                    dim_head=block_out_channels[-1],
                    rescale_output_factor=1,
                    eps=1e-6,
                    norm_num_groups=32,
                    spatial_norm_dim=None,
                    residual_connection=True,
                    processor=AttnProcessor2_0_modified(),
                    bias=True,
                    upcast_softmax=True,
                    _from_deprecated_attn_block=True,
                    dropout=0.0,
                )
                for _ in range(3)
            ]
        )

        # up
        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                prev_output_channel=None,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn="silu",
                resnet_groups=norm_num_groups,
                attention_head_dim=output_channel,
                temb_channels=None,
                resnet_time_scale_shift="group",
            )
            self.up_blocks.append(up_block)

        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

        self.splatter = DecoderSplatting()

    def forward(
        self, latent_image, latent_depth, target_extrinsics, source_extrinsics, intrinsics, near, far, plucker_ray=None
    ):
        # calculate plucker rays
        lh, lw = latent_image.shape[-2:]
        if plucker_ray is None:
            ray_embeddings = get_plucker_rays(source_extrinsics, intrinsics, h=lh, w=lw, stride=8)
        else:
            ray_embeddings = plucker_ray

        if self.training:
            ray_embeddings += torch.randn_like(ray_embeddings) * 0.0001  # simulating some noises

        if latent_depth is not None:
            x = torch.cat([latent_image, latent_depth, ray_embeddings], dim=2)  # channels: 16 + 16 + 9
        else:
            x = torch.cat([latent_image, ray_embeddings], dim=2)

        b, v, _, h, w = x.size()
        x = rearrange(x, "b v c h w -> (b v) c h w")  # TODO: patchify?
        x = self.conv_in(x)
        x = self.gn(x)

        for attn_block in self.attn_block_pre:
            x = attn_block(x, num_views=v)

        x = self.mid_block(x)

        for attn_block in self.attn_block_post:
            x = attn_block(x, num_views=v)

        # up
        for up_block in self.up_blocks:
            x = up_block(x)

        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)

        x = rearrange(x, "(b v) c h w -> b v c h w", b=b, v=v)  # (b v c_out h w)

        with torch.amp.autocast("cuda", enabled=False):
            if self.training:
                x_img, x_depth, b_loss, s_loss = self.splatter(
                    x.float(), target_extrinsics, source_extrinsics, intrinsics, near, far
                )
                return x_img, repeat(x_depth, "b c h w -> b (c r) h w", r=3), b_loss, s_loss
            else:
                x = self.splatter.gaussian_adapter(
                    x.float(), source_extrinsics, intrinsics, near, far
                )  # return encoded gaussian primitives

                return x

    def eval_rendering(self, means, covariance, opacity, rgb, target_extrinsics, target_intrinsics, near, far):
        # TODO: I should merge this eval rendering loop for this
        return self.splatter.eval_forward(
            means, covariance, opacity, rgb, target_extrinsics, target_intrinsics, near, far
        )
