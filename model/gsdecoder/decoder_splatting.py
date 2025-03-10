import torch
from einops import einsum, rearrange, repeat
from torch import nn

from .camera_embedding import meshgrid
from .cuda_splatting import render_cuda

# Reference: https://github.com/dcharatan/pixelsplat/blob/main/src/model/encoder/common/gaussian_adapter.py
# We slightly modify the official code to fit in our setting.


# https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
def quaternion_to_matrix(quaternions, eps: float = 1e-8):
    # Order changed to match scipy format!
    i, j, k, r = torch.unbind(quaternions, dim=-1)
    two_s = 2 / ((quaternions * quaternions).sum(dim=-1) + eps)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return rearrange(o, "... (i j) -> ... i j", i=3, j=3)


def build_covariance(scale, rotation_xyzw):
    scale = scale.diag_embed()
    rotation = quaternion_to_matrix(rotation_xyzw)
    return rotation @ scale @ rearrange(scale, "... i j -> ... j i") @ rearrange(rotation, "... i j -> ... j i")


class DecoderSplatting(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("background_color", torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32), persistent=False)
        self.act_scale = nn.Softplus()
        self.act_rgb = nn.Softplus()

    def get_scale_multiplier(self, intrinsics):
        pixel_size = torch.ones((2,), dtype=torch.float32, device=intrinsics.device)
        xy_multipliers = einsum(
            intrinsics[..., :2, :2].inverse(),
            pixel_size,
            "... i j, j -> ... i",
        )
        return xy_multipliers.sum(dim=-1)

    def gaussian_adapter(self, raw_gaussians, extrinsics, intrinsics, near=1, far=100):
        b, v, c, h, w = raw_gaussians.shape
        if len(intrinsics.shape) == 3:
            intrinsics = repeat(intrinsics, "b i j -> b v i j", v=v)
        extrinsics = repeat(extrinsics, "b v i j -> b v () () i j")
        intrinsics = repeat(intrinsics, "b v i j -> b v () () i j")
        raw_gaussians = rearrange(raw_gaussians, "b v c h w -> b v h w c")

        rgb, disp, opacity, scales, rotations, xy_offset = raw_gaussians.split((3, 1, 1, 3, 4, 2), dim=-1)

        # calculate xy_offset and origin/direction for each view.
        pixel_coords = meshgrid((w, h), normalized=False, indexing="xy", device=raw_gaussians.device)
        pixel_coords = repeat(pixel_coords, "h w c -> b v h w c", b=b, v=v)

        coordinates = pixel_coords + (xy_offset.sigmoid() - 0.5)
        coordinates = torch.cat([coordinates, torch.ones_like(coordinates[..., :1])], dim=-1)

        directions = einsum(intrinsics.inverse(), coordinates, "... i j, ... j -> ... i")
        directions = directions / directions.norm(dim=-1, keepdim=True)
        # directions = directions / directions[..., -1:]
        directions = torch.cat([directions, torch.zeros_like(directions[..., :1])], dim=-1)
        directions = einsum(extrinsics, directions, "... i j, ... j -> ... i")
        origins = extrinsics[..., -1].broadcast_to(directions.shape)

        # calculate depth from disparity
        depths = 1.0 / (disp.sigmoid() * (1.0 / near - 1.0 / far) + 1.0 / far)

        # calculate all parameters of gaussian splats
        means = origins + directions * depths

        multiplier = self.get_scale_multiplier(intrinsics)
        scales = self.act_scale(scales) * multiplier[..., None]

        if len(torch.where(scales > 0.05)[0]) > 0:
            big_gaussian_reg_loss = torch.mean(scales[torch.where(scales > 0.05)])
        else:
            big_gaussian_reg_loss = 0

        if len(torch.where(scales < 1e-6)[0]) > 0:
            small_gaussian_reg_loss = torch.mean(-torch.log(scales[torch.where(scales < 1e-6)]) * 0.1)
        else:
            small_gaussian_reg_loss = 0

        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + 1e-8)
        covariances = build_covariance(scales, rotations)
        c2w_rotations = extrinsics[..., :3, :3]
        covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)

        opacity = opacity.sigmoid()
        rgb = self.act_rgb(rgb)
        if self.training:
            return means, covariances, opacity, rgb, big_gaussian_reg_loss, small_gaussian_reg_loss
        else:
            return means, covariances, opacity, rgb, rotations, scales

    def forward(self, raw_gaussians, target_extrinsics, source_extrinsics, intrinsics, near=1, far=100):
        b, v = target_extrinsics.shape[:2]
        h, w = raw_gaussians.shape[-2:]

        means, covariance, opacity, rgb, b_loss, s_loss = self.gaussian_adapter(
            raw_gaussians, source_extrinsics, intrinsics, near, far
        )
        lower_ext = torch.tensor([0.0, 0.0, 0.0, 1.0], device=target_extrinsics.device, dtype=torch.float32)
        lower_ext = repeat(lower_ext, "i -> b v () i", b=b, v=v)
        homo_extrinsics = torch.cat([target_extrinsics, lower_ext], dim=2)
        output_image, output_depth = render_cuda(
            rearrange(homo_extrinsics, "b v i j -> (b v) i j"),
            repeat(intrinsics, "b i j -> (b v) i j", v=v),
            near,
            far,
            (h, w),
            v,
            repeat(self.background_color, "i -> bv i", bv=b * v),
            rearrange(means, "b v h w xyz -> b (v h w) xyz"),
            rearrange(covariance, "b v h w i j -> b (v h w) i j"),
            rearrange(rgb, "b v h w c -> b (v h w) c"),
            rearrange(opacity, "b v h w o -> b (v h w) o"),
        )
        return output_image, output_depth, b_loss, s_loss

    def eval_forward(self, means, covariance, opacity, rgb, target_extrinsics, target_intrinsics, near, far):
        # TODO: merge eval and training forward logics
        _, _, h, w, _ = means.shape
        b, target_view_num, _, _ = target_extrinsics.shape

        if len(target_intrinsics.shape) == 3:
            target_intrinsics = repeat(target_intrinsics, "b i j -> (b v) i j", v=target_view_num)
        else:
            target_intrinsics = repeat(target_intrinsics, "b v i j -> (b v) i j")

        lower_ext = torch.tensor([0.0, 0.0, 0.0, 1.0], device=target_extrinsics.device, dtype=torch.float32)
        lower_ext = repeat(lower_ext, "i -> b v () i", b=b, v=target_view_num)
        homo_extrinsics = torch.cat([target_extrinsics, lower_ext], dim=2)
        output_image, output_depth = render_cuda(
            rearrange(homo_extrinsics, "b v i j -> (b v) i j"),
            target_intrinsics,
            near,
            far,
            (h, w),
            target_view_num,
            repeat(self.background_color, "i -> bv i", bv=b * target_view_num),
            rearrange(means, "b v h w xyz -> b (v h w) xyz"),
            rearrange(covariance, "b v h w i j -> b (v h w) i j"),
            rearrange(rgb, "b v h w rgb -> b (v h w) rgb"),
            rearrange(opacity, "b v h w o -> b (v h w) o"),
        )
        return output_image
