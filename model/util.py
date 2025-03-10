from collections import OrderedDict

import torch
from diffusers import AutoencoderKL, SD3Transformer2DModel
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


def create_vae(cfg):
    vae = AutoencoderKL.from_pretrained(
        cfg.mv_rf_model.hf_path,
        subfolder="vae",
    )
    return vae


def create_sd3_transformer():
    sd3 = SD3Transformer2DModel.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        subfolder="transformer",
        low_cpu_mem_usage=True,
    )
    return sd3


def create_depth(cfg):
    """
    How to use depth models

    # prepare image for the model
    inputs = depth_image_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
    """
    depth_image_processor = AutoImageProcessor.from_pretrained(cfg.depth_encoder.hf_path)
    depth_model = AutoModelForDepthEstimation.from_pretrained(cfg.depth_encoder.hf_path)
    return depth_image_processor, depth_model


def convert_to_buffer(module: torch.nn.Module, persistent: bool = True):
    # Recurse over child modules.
    for name, child in list(module.named_children()):
        convert_to_buffer(child, persistent)

    # Also re-save buffers to change persistence.
    for name, parameter_or_buffer in (
        *module.named_parameters(recurse=False),
        *module.named_buffers(recurse=False),
    ):
        value = parameter_or_buffer.detach().clone()
        delattr(module, name)
        module.register_buffer(name, value, persistent=persistent)


def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """

    model_context_manager = model.summon_full_params(model)
    with model_context_manager:
        ema_params = OrderedDict(ema_model.named_parameters())
        model_params = OrderedDict(model.named_parameters())
        for name, param in model_params.items():
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def stack_depth_images(depth_in):
    """
    Crawled from https://github.com/prs-eth/Marigold/blob/main/src/trainer/marigold_trainer.py#L395
    """
    if 4 == len(depth_in.shape):
        stacked = depth_in.repeat(1, 3, 1, 1)
    elif 3 == len(depth_in.shape):
        stacked = depth_in.unsqueeze(1)
        stacked = depth_in.repeat(1, 3, 1, 1)
    return stacked
