import os
from pathlib import Path

import hydra
import torch
import torchvision
from diffusers import FlowMatchEulerDiscreteScheduler
from einops import rearrange, repeat
from model.gsdecoder.camera_embedding import get_plucker_rays, optimize_plucker_ray
from model.gsdecoder.load_gsdecoder import create_gsdecoder
from model.multiview_rf.load_mv_sd3 import create_sd_multiview_rf_model
from model.multiview_rf.text_embedding import compute_text_embeddings
from model.refiner.camera_util import export_mv, export_ply_for_gaussians, export_video, load_ply_for_gaussians
from model.refiner.sds_pp_refiner import GSRefinerSDSPlusPlus
from model.util import create_sd3_transformer, create_vae
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from util import dist_util, camera_visualization
from pytorch3d.utils import cameras_from_opencv_projection
import matplotlib.pyplot as plt

def visualize_camera_pose(cameras, path):
    # visualize cameras
    extrinsic = cameras[0][:, :, 3:]
    device = cameras.device
    num_view = len(extrinsic)
    homo_extrinsic = repeat(torch.eye(4).to(device, dtype=extrinsic.dtype), 'i j -> v i j', v=num_view).clone()
    homo_extrinsic[:, :3] = extrinsic
    w2c = homo_extrinsic.inverse()
    image_size = repeat(torch.tensor([256, 256], device=device), "i -> v i", v=num_view)
    cameras_ours = cameras_from_opencv_projection(
        R=w2c[:, :3, :3],
        tvec=w2c[:, :3, -1],
        camera_matrix=cameras[0, :, :, :3],
        image_size=image_size,
    )
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.clear()
    points = camera_visualization.plot_cameras(ax, cameras_ours)
    cc = points[:, -1]
    max_scene = cc.max(dim=0)[0].cpu()
    min_scene = cc.min(dim=0)[0].cpu()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xlim3d([min_scene[0] - 0.1, max_scene[0] + 0.3])
    ax.set_ylim3d([min_scene[2] - 0.1, max_scene[2] + 0.1])
    ax.set_zlim3d([min_scene[1] - 0.1, max_scene[1] + 0.1])
    ax.invert_yaxis()
    plt.savefig(os.path.join(path, "pose.pdf"), bbox_inches="tight", pad_inches=0, transparent=True)

def save_results(path, output, poses, Ks):
    means, covariance, opacity, rgb, rotation, scale = output
    _, v, h, w, _ = means.shape

    means = rearrange(means, "() v h w xyz -> (v h w) xyz")
    opacity = rearrange(opacity, "() v h w o -> (v h w) o")
    rgb = rearrange(rgb, "() v h w rgb -> (v h w) rgb")
    rotation = rearrange(rotation, "() v h w q -> (v h w) q")
    scale = rearrange(scale, "() v h w s -> (v h w) s")
    source_rotations = repeat(poses[..., :3, :3], "() v i j -> (v h w) i j", h=h, w=w)

    cam_rotation_matrix = R.from_quat(rotation.detach().cpu().numpy()).as_matrix()
    world_rotation_matrix = source_rotations.detach().cpu().numpy() @ cam_rotation_matrix
    world_rotations = R.from_matrix(world_rotation_matrix).as_quat()
    world_rotations = torch.from_numpy(world_rotations).to(source_rotations.device)

    export_ply_for_gaussians(os.path.join(path, 'gaussian'), (means, rgb[:, None], opacity, scale, world_rotations))
    cameras = torch.cat([Ks[0], poses[0, :, :3]], dim=-1)
    torch.save(cameras, path / "cameras.pt")


def generate_sampling(
    text_prompt,
    text_encoders,
    tokenizers,
    model,
    gs_decoder,
    vae,
    noise_scheduler,
    stable_diffusion3_transformer,
    cfg,
    device,
    dtype,
):
    """
    Generating a sample, supposing that batch size is 1
    """
    batch_size = 1

    if isinstance(text_prompt, str):
        text_prompt = [text_prompt]
    negative_prompt = [""]

    noise_scheduler.set_timesteps(cfg.inference.sample.num_steps)
    timesteps = noise_scheduler.timesteps
    latents = torch.randn(batch_size, 8, 38, 32, 32, device=device, dtype=dtype)

    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            text_prompt,
            text_encoders,
            tokenizers,
            77,
            device,
        )
        if cfg.inference.sample.cfg:
            negative_prompt_embeds, pooled_negative_prompt_embeds = compute_text_embeddings(
                negative_prompt, text_encoders, tokenizers, 77, device=device
            )
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([pooled_negative_prompt_embeds, pooled_prompt_embeds], dim=0)
            cfg_scales = []
            cfg_scales.append([cfg.inference.sample.cfg_scale[0]] * 16)  # Image
            cfg_scales.append([cfg.inference.sample.cfg_scale[1]] * 16)  # Depth
            cfg_scales.append([cfg.inference.sample.cfg_scale[2]] * 6)  # Pose
            cfg_scales = torch.tensor(sum(cfg_scales, []), dtype=latents.dtype, device=latents.device)
            cfg_scales = repeat(cfg_scales, "i -> () () i () ()")

        for i, t in tqdm(enumerate(timesteps)):
            latent_model_input = torch.cat([latents] * 2) if cfg.inference.sample.cfg else latents
            timestep = t.expand(latent_model_input.shape[0]).to(device)

            noise_pred = model(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            )[0]

            if cfg.inference.sample.cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + cfg_scales * (noise_pred_text - noise_pred_uncond)

                if noise_scheduler.step_index is None:
                    noise_scheduler._init_step_index(t)

                if noise_scheduler.step_index <= cfg.inference.sample.stop_ray:
                    with torch.enable_grad():
                        original_latent = latents - noise_pred * noise_scheduler.sigmas[noise_scheduler.step_index]
                        poses, Ks, inv_poses = optimize_plucker_ray(original_latent[:, :, -6:])
                        clean_ray_latent = get_plucker_rays(poses[:, :, :3], Ks, is_diffusion=True)

            if cfg.inference.sample.sd3_guidance and (i % 3 == 0) and (i < cfg.inference.sample.stop_ray):
                image_2d_flatten_latent = rearrange(latent_model_input, "b v c h w -> (b v) c h w")
                image_timestep = timestep.repeat(latent_model_input.shape[1])

                if cfg.inference.sample.sd3_cfg:
                    positive_prompt_embed = prompt_embeds[1].repeat(latent_model_input.shape[1], 1, 1)
                    negative_prompt_embed = prompt_embeds[0].repeat(latent_model_input.shape[1], 1, 1)
                    image_prompt_embed = torch.cat([negative_prompt_embed, positive_prompt_embed], dim=0)

                    positive_pooled_prompt_embed = pooled_prompt_embeds[1].repeat(latent_model_input.shape[1], 1)
                    negative_pooled_prompt_embed = pooled_prompt_embeds[0].repeat(latent_model_input.shape[1], 1)
                    image_pooled_prompt_embed = torch.cat(
                        [negative_pooled_prompt_embed, positive_pooled_prompt_embed], dim=0
                    )
                else:
                    image_prompt_embed = prompt_embeds.repeat(latent_model_input.shape[1], 1, 1)
                    image_pooled_prompt_embed = pooled_prompt_embeds.repeat(latent_model_input.shape[1], 1)

                sd_noise_pred = stable_diffusion3_transformer(
                    hidden_states=image_2d_flatten_latent[:, :16, :, :],
                    timestep=image_timestep,
                    encoder_hidden_states=image_prompt_embed,
                    pooled_projections=image_pooled_prompt_embed,
                    return_dict=False,
                )[0]

                if cfg.inference.sample.sd3_cfg:
                    sd_noise_pred_uncond, sd_noise_pred_text = sd_noise_pred.chunk(2)
                    sd_noise_pred = sd_noise_pred_uncond + cfg.inference.sample.sd3_scale * (
                        sd_noise_pred_text - sd_noise_pred_uncond
                    )
                sd_noise_pred = rearrange(sd_noise_pred, "(b v) c h w -> b v c h w", v=latent_model_input.shape[1])
                noise_pred[:, :, :16, :, :] = sd_noise_pred

            latents = noise_scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            # update ray latent
            sigmas = noise_scheduler.sigmas[noise_scheduler.step_index]
            noise = torch.randn_like(latents[:, :, 32:])
            noisy_ray_latent = (1.0 - sigmas) * clean_ray_latent + sigmas * noise
            latents[:, :, -6:] = noisy_ray_latent

    with torch.inference_mode():
        img_latent, depth_latent, _ = latents.split([16, 16, 6], dim=2)
        output = gs_decoder(img_latent, depth_latent, None, poses[:, :, :3], Ks, near=0.05, far=20)
        means, covariance, opacity, rgb, rotation, scale = output
        # save results
        path = Path(f"{cfg.inference.generate.save_path}/{text_prompt[0]}")
        os.makedirs(path, exist_ok=True)

        text_prompt_save = path / "text_prompt.txt"
        with open(text_prompt_save, "w", encoding="utf-8") as f:
            f.write(text_prompt[0])

        # Save 3DGS and camera parameters
        save_results(path, output, poses, Ks)

        ### Save images
        rendered_images = gs_decoder.eval_rendering(
            means.float(),
            covariance.float(),
            opacity.float(),
            rgb.float(),
            poses[:, :, :3].float(),
            Ks.float(),
            near=0.05,
            far=20,
        )

        rendered_images = rendered_images.clamp(min=0, max=1)
        transform = torchvision.transforms.ToPILImage()

        mv_results_path = path / "mv_results"
        os.makedirs(mv_results_path, exist_ok=True)

        for view, img in enumerate(rendered_images):
            transform(img).save(mv_results_path / f"render_img_{view}.png")


@hydra.main(config_path="../config", config_name="base_config.yaml", version_base="1.1")
def main(cfg):
    dist_util.setup_dist(cfg.general)
    device = dist_util.device()
    dtype = torch.float16
    print(f"Device: {device}")

    path = Path(f"{cfg.inference.generate.save_path}/{cfg.inference.generate.prompt}")

    if os.path.exists(path / "cameras.pt") and os.path.exists(path / "gaussian.ply"):
        print("Step 1 is already done")  # Skip generating First step
    else:
        """
        First Step: Generate a initial 3DGS
        """
        # create vae part
        vae = create_vae(cfg)
        decoder = create_gsdecoder(cfg)
        vae, decoder = vae.to(device=device, dtype=dtype), decoder.to(device=device)
        decoder.load_state_dict(torch.load(cfg.inference.gsdecoder_ckpt, map_location="cpu", weights_only=False))
        vae.eval(), decoder.eval()

        # MV RF
        model, tokenizer, text_encoders = create_sd_multiview_rf_model()
        model = model.to(device=device, dtype=dtype)
        model.load_state_dict(torch.load(cfg.inference.mv_rf_ckpt, map_location="cpu", weights_only=False))
        model.eval()

        # Text encoders
        text_encoders_list = []
        for i, text_encoder in enumerate(text_encoders):
            text_encoder.requires_grad_(False)
            text_encoder.to(device, dtype=dtype)
            text_encoders_list.append(text_encoder.to(device, dtype=dtype))
        text_encoders = text_encoders_list

        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            cfg.mv_rf_model.hf_path,
            subfolder="scheduler",
            force_download=False,
            shift=1.0,
        )

        if cfg.inference.sample.sd3_guidance:
            stable_diffusion3_transformer = create_sd3_transformer()
            stable_diffusion3_transformer = stable_diffusion3_transformer.to(device, dtype=dtype)
            stable_diffusion3_transformer.eval()
        else:
            stable_diffusion3_transformer = None

        generate_sampling(
            cfg.inference.generate.prompt,
            text_encoders,
            tokenizer,
            model,
            decoder,
            vae,
            noise_scheduler,
            stable_diffusion3_transformer,
            cfg,
            device,
            dtype,
        )

    """
    Second Step: Refine the 3DGS
    """
    refiner = GSRefinerSDSPlusPlus(**cfg.inference.generate.refiner.args)
    refiner.to(device)
    gaussians = load_ply_for_gaussians(path / "gaussian.ply", device=device)
    cameras = torch.load(path / "cameras.pt", weights_only=False).unsqueeze(0)
    refined_gaussians = refiner.refine_gaussians(gaussians, cfg.inference.generate.prompt, dense_cameras=cameras)

    visualize_camera_pose(cameras, path)
    export_ply_for_gaussians(os.path.join(path, 'refined_gaussian'), [p[0] for p in refined_gaussians])

    def render_fn(cameras, h=512, w=512):
        return refiner.renderer(cameras, refined_gaussians, h=h, w=w, bg=None)[:2]

    export_mv(
        render_fn,
        path,
        cameras,
    )

    export_video(render_fn, path, "refined_video", cameras, device=device)


if __name__ == "__main__":
    main()
