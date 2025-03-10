import cv2
import torch
from einops import rearrange, repeat
from torch import nn

# Reference: https://github.com/valeoai/LaRa/blob/main/semanticbev/models/components/LaRa_embeddings.py
# We slightly modify the official code to fit in our setting.


def meshgrid(spatial_shape, normalized=True, indexing="ij", device=None):
    """Create evenly spaced position coordinates for self.spatial_shape with values in [v_min, v_max].
    :param v_min: minimum coordinate value per dimension.
    :param v_max: maximum coordinate value per dimension.
    :return: position coordinates tensor of shape (*shape, len(shape)).
    """
    if normalized:
        axis_coords = [torch.linspace(-1.0, 1.0, steps=s, device=device) for s in spatial_shape]
    else:
        axis_coords = [torch.linspace(0, s - 1, steps=s, device=device) for s in spatial_shape]

    grid_coords = torch.meshgrid(*axis_coords, indexing=indexing)

    return torch.stack(grid_coords, dim=-1)


def get_plucker_rays(extrinsics, intrinsics, h=32, w=32, stride=8, is_diffusion=False):
    b, v = extrinsics.shape[:2]

    # Adjust intrinsics scale due to downsizing by input_stride (we take feature maps as input not the raw images)

    updated_intrinsics = intrinsics.clone().unsqueeze(1) if len(intrinsics.shape) == 3 else intrinsics.clone()
    updated_intrinsics[..., 0, 0] *= 1 / stride
    updated_intrinsics[..., 0, 2] *= 1 / stride
    updated_intrinsics[..., 1, 1] *= 1 / stride
    updated_intrinsics[..., 1, 2] *= 1 / stride

    # create positionnal encodings
    pixel_coords = meshgrid((w, h), normalized=False, indexing="xy", device=extrinsics.device)
    ones = torch.ones((h, w, 1), device=extrinsics.device)

    pixel_coords = torch.cat([pixel_coords, ones], dim=-1)  # [x, y, 1] vectors of pixel coordinates
    pixel_coords = rearrange(pixel_coords, "h w c -> c (h w)")
    pixel_coords = repeat(pixel_coords, "... -> b v ...", b=b, v=v)

    # Split extrinsics into rots and trans, rots == c2w, trans == camera center at world coordinate
    rots, trans = extrinsics.split([3, 1], dim=-1)

    # pixel_coords.shape = [B, N, 3, K] | N # of cams, K # of pixels
    directions = rots @ updated_intrinsics.inverse() @ pixel_coords
    directions = directions / directions.norm(dim=2, keepdim=True)
    directions = rearrange(directions, "b v c (h w) -> b v c h w", h=h, w=w)
    cam_origins = repeat(trans.squeeze(-1), "b v c -> b v c h w", h=h, w=w)
    moments = torch.cross(cam_origins, directions, dim=2)

    output = [] if is_diffusion else [cam_origins]
    output.append(directions)
    output.append(moments)

    return torch.cat(output, dim=2)


def optimize_plucker_ray(ray_latent):
    b, v, _, h, w = ray_latent.shape
    ray_latent = ray_latent.float()  # this function does not support bfloat16 type..
    directions, moments = ray_latent.split(3, dim=2)

    # Reverse Process
    c = torch.linalg.norm(directions, dim=2, keepdim=True)
    origins = torch.cross(directions, moments / c, dim=2)

    new_trans = intersect_skew_lines_high_dim(
        rearrange(origins, "b n c h w -> b n (h w) c"), rearrange(directions, "b n c h w -> b n (h w) c")
    )

    # Retrieve target rays
    I_intrinsic_ = torch.tensor([[1, 0, h // 2], [0, 1, w // 2], [0, 0, 1]], dtype=ray_latent.dtype, device=c.device)
    I_intrinsic = repeat(I_intrinsic_, "i j -> b v i j", b=b, v=v)
    I_rot = repeat(torch.eye(3, dtype=ray_latent.dtype, device=c.device), "i j -> b v i j", b=b, v=v)

    # create positionnal encodings
    pixel_coords = meshgrid((w, h), normalized=False, indexing="xy", device=ray_latent.device)
    ones = torch.ones((h, w, 1), device=ray_latent.device)

    pixel_coords = torch.cat([pixel_coords, ones], dim=-1)  # [x, y, 1] vectors of pixel coordinates
    pixel_coords = rearrange(pixel_coords, "h w c -> c (h w)")
    pixel_coords = repeat(pixel_coords, "... -> b v ...", b=b, v=v)

    I_directions = I_rot @ I_intrinsic.inverse() @ pixel_coords
    I_directions = I_directions / I_directions.norm(dim=2, keepdim=True)
    I_directions = rearrange(I_directions, "b v c (h w) -> b v c h w", h=h, w=w)

    new_rots, new_intrinsics = [], []
    for bb in range(b):
        Rs, Ks = [], []
        for vv in range(v):
            R, f, pp = compute_optimal_rotation_intrinsics(
                I_directions[bb, vv], directions[bb, vv], reproj_threshold=0.2
            )
            Rs.append(R)
            K = I_intrinsic_.clone()
            K[:2, :2] = torch.diag(1 / f)
            K[:, -1][:2] += pp
            Ks.append(K)

        new_rots.append(torch.stack(Rs))
        new_intrinsics.append(torch.stack(Ks))

    new_rots = torch.stack(new_rots)
    new_intrinsics = torch.stack(new_intrinsics)

    ff = nn.Parameter(new_intrinsics[..., [0, 1], [0, 1]].mean(1)[0], requires_grad=True)
    optimizer = torch.optim.Adam([ff], lr=0.01)
    X = torch.tensor(
        [[1, 0, h // 2], [0, 1, w // 2], [0, 0, 1]], dtype=ray_latent.dtype, device=c.device, requires_grad=True
    )

    # Optimization loop
    num_iterations = 10
    for i in range(num_iterations + 1):
        optimizer.zero_grad()
        scale_matrix = torch.diag(torch.cat([ff, torch.ones_like(ff[:1])], dim=0))

        triu_intrinsics = X @ scale_matrix
        re_X = repeat(triu_intrinsics, "i j -> b v i j", b=b, v=v)

        I_directions = I_rot @ re_X.inverse() @ pixel_coords
        I_directions = I_directions / I_directions.norm(dim=2, keepdim=True)
        I_directions = rearrange(I_directions, "b v c (h w) -> b v c h w", h=h, w=w)

        new_rots = []
        for bb in range(b):
            Rs = []
            for vv in range(v):
                R = compute_optimal_rotation_alignment(I_directions[bb, vv], directions[bb, vv])
                Rs.append(R)

            new_rots.append(torch.stack(Rs))

        new_rots = torch.stack(new_rots)

        if i == num_iterations:
            break

        I_directions = new_rots.clone() @ re_X.inverse() @ pixel_coords.clone()
        I_directions = I_directions / I_directions.norm(dim=2, keepdim=True)
        I_directions = rearrange(I_directions, "b v c (h w) -> b v c h w", h=h, w=w)
        loss = torch.norm(I_directions - directions, dim=2).mean()

        loss.backward()
        optimizer.step()

    new_rots = new_rots.detach().clone()
    new_intrinsics = re_X.detach().clone()

    stride = 8
    new_intrinsics[..., 0, 0] *= stride
    new_intrinsics[..., 0, 2] *= stride
    new_intrinsics[..., 1, 1] *= stride
    new_intrinsics[..., 1, 2] *= stride

    # normalize camera pose
    new_poses = torch.cat([new_rots, new_trans.unsqueeze(-1)], dim=-1)
    bottom = torch.tensor([0, 0, 0, 1], dtype=new_poses.dtype, device=new_poses.device)
    homo_poses = torch.cat([new_poses, repeat(bottom, "i -> b v () i", b=b, v=v)], dim=-2)
    inv_poses = torch.cat([homo_poses.inverse()[:, :, :3], repeat(bottom, "i -> b v () i", b=b, v=v)], dim=-2)
    return homo_poses, new_intrinsics, inv_poses


# Refer to RayDiffusion
def intersect_skew_lines_high_dim(p, r):
    # p : num views x 3 x num points
    # Implements https://en.wikipedia.org/wiki/Skew_lines In more than two dimensions
    dim = p.shape[-1]

    eye = torch.eye(dim, device=p.device, dtype=p.dtype)[None, None, None]
    I_min_cov = eye - (r[..., None] * r[..., None, :])
    sum_proj = I_min_cov.matmul(p[..., None]).sum(dim=-3)

    p_intersect = torch.linalg.lstsq(I_min_cov.sum(dim=-3), sum_proj).solution[..., 0]

    return p_intersect


def compute_optimal_rotation_intrinsics(rays_origin, rays_target, z_threshold=1e-8, reproj_threshold=0.2):
    """
    Note: for some reason, f seems to be 1/f.

    Args:
        rays_origin (torch.Tensor): (3, H, W)
        rays_target (torch.Tensor): (3, H, W)
        z_threshold (float): Threshold for z value to be considered valid.

    Returns:
        R (torch.tensor): (3, 3)
        focal_length (torch.tensor): (2,)
        principal_point (torch.tensor): (2,)
    """
    device = rays_origin.device
    _, h, w = rays_origin.shape

    rays_origin = rearrange(rays_origin, "c h w -> (h w) c")
    rays_target = rearrange(rays_target, "c h w -> (h w) c")

    z_mask = torch.logical_and(torch.abs(rays_target) > z_threshold, torch.abs(rays_origin) > z_threshold)[:, 2]
    rays_target = rays_target[z_mask]
    rays_origin = rays_origin[z_mask]
    rays_origin = rays_origin[:, :2] / rays_origin[:, -1:]
    rays_target = rays_target[:, :2] / rays_target[:, -1:]

    A, _ = cv2.findHomography(
        rays_origin.cpu().numpy(),
        rays_target.cpu().numpy(),
        cv2.RANSAC,
        reproj_threshold,
    )
    A = torch.from_numpy(A).float().to(device)

    if torch.linalg.det(A) < 0:
        A = -A

    R, L = ql_decomposition(A)
    L = L / L[2][2]

    f = torch.stack((L[0][0], L[1][1]))
    pp = torch.stack((L[2][0], L[2][1]))
    return R, f, pp


def ql_decomposition(A):
    P = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]], device=A.device).float()
    A_tilde = torch.matmul(A, P)
    Q_tilde, R_tilde = torch.linalg.qr(A_tilde)
    Q = torch.matmul(Q_tilde, P)
    L = torch.matmul(torch.matmul(P, R_tilde), P)
    d = torch.diag(L)
    Q[:, 0] *= torch.sign(d[0])
    Q[:, 1] *= torch.sign(d[1])
    Q[:, 2] *= torch.sign(d[2])
    L[0] *= torch.sign(d[0])
    L[1] *= torch.sign(d[1])
    L[2] *= torch.sign(d[2])
    return Q, L


def compute_optimal_rotation_alignment(A, B):
    """
    Compute optimal R that minimizes: || A - B @ R ||_F

    Args:
        A (torch.Tensor): (3, H, W)
        B (torch.Tensor): (3, H, W)

    Returns:
        R (torch.tensor): (3, 3)
    """
    A = rearrange(A, "c h w -> (h w) c")
    B = rearrange(B, "c h w -> (h w) c")

    # normally with R @ B, this would be A @ B.T
    H = B.T @ A
    U, _, Vh = torch.linalg.svd(H, full_matrices=True)
    s = torch.linalg.det(U @ Vh)
    S_prime = torch.diag(torch.tensor([1, 1, torch.sign(s)], device=A.device))
    return U @ S_prime @ Vh
