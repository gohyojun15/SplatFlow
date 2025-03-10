import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from einops import einsum, rearrange, repeat


def get_fov(intrinsics):
    intrinsics_inv = intrinsics.inverse()

    def process_vector(vector):
        vector = torch.tensor(vector, dtype=torch.float32, device=intrinsics.device)
        vector = einsum(intrinsics_inv, vector, "b i j, j -> b i")
        return vector / vector.norm(dim=-1, keepdim=True)

    left = process_vector([0, 0.5, 1])
    right = process_vector([1, 0.5, 1])
    top = process_vector([0.5, 0, 1])
    bottom = process_vector([0.5, 1, 1])
    fov_x = (left * right).sum(dim=-1).acos()
    fov_y = (top * bottom).sum(dim=-1).acos()
    return torch.stack((fov_x, fov_y), dim=-1)


def get_projection_matrix(near, far, fov_x, fov_y):
    """Maps points in the viewing frustum to (-1, 1) on the X/Y axes and (0, 1) on the Z
    axis. Differs from the OpenGL version in that Z doesn't have range (-1, 1) after
    transformation and that Z is flipped.
    """
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    top = tan_fov_y * near
    bottom = -top
    right = tan_fov_x * near
    left = -right

    (b,) = fov_x.shape
    result = torch.zeros((b, 4, 4), dtype=torch.float32, device=fov_x.device)
    result[:, 0, 0] = 2 * near / (right - left)
    result[:, 1, 1] = 2 * near / (top - bottom)
    result[:, 0, 2] = (right + left) / (right - left)
    result[:, 1, 2] = (top + bottom) / (top - bottom)
    result[:, 3, 2] = 1
    result[:, 2, 2] = far / (far - near)
    result[:, 2, 3] = -(far * near) / (far - near)
    return result


def render_cuda(
    extrinsics,
    intrinsics,
    near,
    far,
    image_shape,
    num_views,
    background_color,
    gaussian_means,
    gaussian_covariances,
    gaussians_rgb,
    gaussian_opacities,
):
    b, _, _ = extrinsics.shape
    h, w = image_shape

    update_intrinsics = intrinsics.clone()
    update_intrinsics[:, 0] *= 1 / w
    update_intrinsics[:, 1] *= 1 / h

    fov_x, fov_y = get_fov(update_intrinsics).unbind(dim=-1)
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    projection_matrix = get_projection_matrix(near, far, fov_x, fov_y)
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
    full_projection = view_matrix @ projection_matrix

    gaussian_means = repeat(gaussian_means, "b n c -> (b v) n c", v=num_views)
    gaussian_covariances = repeat(gaussian_covariances, "b n i j -> (b v) n i j", v=num_views)
    gaussians_rgb = repeat(gaussians_rgb, "b n c -> (b v) n c", v=num_views)
    gaussian_opacities = repeat(gaussian_opacities, "b n c -> (b v) n c", v=num_views)

    all_images = []
    all_depths = []
    for i in range(b):
        # Set up a tensor for the gradients of the screen-space means.
        mean_gradients = torch.zeros_like(gaussian_means[i], requires_grad=True)
        try:
            mean_gradients.retain_grad()
        except Exception:
            pass

        settings = GaussianRasterizationSettings(
            image_height=h,
            image_width=w,
            tanfovx=tan_fov_x[i].item(),
            tanfovy=tan_fov_y[i].item(),
            bg=background_color[i],
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],
            projmatrix=full_projection[i],
            sh_degree=0,
            campos=extrinsics[i, :3, 3],
            prefiltered=False,  # This matches the original usage.
            debug=False,
        )
        rasterizer = GaussianRasterizer(settings)

        row, col = torch.triu_indices(3, 3)

        image, _, depth, _ = rasterizer(
            means3D=gaussian_means[i],
            means2D=mean_gradients,
            shs=None,
            colors_precomp=gaussians_rgb[i],
            opacities=gaussian_opacities[i],
            cov3D_precomp=gaussian_covariances[i, :, row, col],
        )
        all_images.append(image)
        all_depths.append(depth)

    return torch.stack(all_images), torch.stack(all_depths)
