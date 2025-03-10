import math
import os
from io import BytesIO

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from einops import einsum, rearrange
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R
from torchvision.utils import save_image


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

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
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def matrix_to_quaternion(M: torch.Tensor) -> torch.Tensor:
    """
    Matrix-to-quaternion conversion method. Equation taken from
    https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
    Args:
        M: rotation matrices, (... x 3 x 3)
    Returns:
        q: quaternion of shape (... x 4)
    """
    prefix_shape = M.shape[:-2]
    Ms = M.reshape(-1, 3, 3)

    trs = 1 + Ms[:, 0, 0] + Ms[:, 1, 1] + Ms[:, 2, 2]

    Qs = []

    for i in range(Ms.shape[0]):
        M = Ms[i]
        tr = trs[i]
        if tr > 0:
            r = torch.sqrt(tr) / 2.0
            x = (M[2, 1] - M[1, 2]) / (4 * r)
            y = (M[0, 2] - M[2, 0]) / (4 * r)
            z = (M[1, 0] - M[0, 1]) / (4 * r)
        elif (M[0, 0] > M[1, 1]) and (M[0, 0] > M[2, 2]):
            S = torch.sqrt(1.0 + M[0, 0] - M[1, 1] - M[2, 2]) * 2  # S=4*qx
            r = (M[2, 1] - M[1, 2]) / S
            x = 0.25 * S
            y = (M[0, 1] + M[1, 0]) / S
            z = (M[0, 2] + M[2, 0]) / S
        elif M[1, 1] > M[2, 2]:
            S = torch.sqrt(1.0 + M[1, 1] - M[0, 0] - M[2, 2]) * 2  # S=4*qy
            r = (M[0, 2] - M[2, 0]) / S
            x = (M[0, 1] + M[1, 0]) / S
            y = 0.25 * S
            z = (M[1, 2] + M[2, 1]) / S
        else:
            S = torch.sqrt(1.0 + M[2, 2] - M[0, 0] - M[1, 1]) * 2  # S=4*qz
            r = (M[1, 0] - M[0, 1]) / S
            x = (M[0, 2] + M[2, 0]) / S
            y = (M[1, 2] + M[2, 1]) / S
            z = 0.25 * S
        Q = torch.stack([r, x, y, z], dim=-1)
        Qs += [Q]

    return torch.stack(Qs, dim=0).reshape(*prefix_shape, 4)


@torch.amp.autocast("cuda", enabled=False)
def quaternion_slerp(q0, q1, fraction, spin: int = 0, shortestpath: bool = True):
    """Return spherical linear interpolation between two quaternions.
    Args:
        quat0: first quaternion
        quat1: second quaternion
        fraction: how much to interpolate between quat0 vs quat1 (if 0, closer to quat0; if 1, closer to quat1)
        spin: how much of an additional spin to place on the interpolation
        shortestpath: whether to return the short or long path to rotation
    """
    d = (q0 * q1).sum(-1)
    if shortestpath:
        # invert rotation
        d[d < 0.0] = -d[d < 0.0]
        q1[d < 0.0] = q1[d < 0.0]

    d = d.clamp(0, 1.0)

    angle = torch.acos(d) + spin * math.pi
    isin = 1.0 / (torch.sin(angle) + 1e-10)
    q0_ = q0 * torch.sin((1.0 - fraction) * angle) * isin
    q1_ = q1 * torch.sin(fraction * angle) * isin

    q = q0_ + q1_
    q[angle < 1e-5, :] = q0

    return q


def sample_from_two_pose(pose_a, pose_b, fraction, noise_strengths=[0, 0]):
    """
    Args:
        pose_a: first pose
        pose_b: second pose
        fraction
    """

    quat_a = matrix_to_quaternion(pose_a[..., :3, :3])
    quat_b = matrix_to_quaternion(pose_b[..., :3, :3])

    quaternion = quaternion_slerp(quat_a, quat_b, fraction)
    quaternion = torch.nn.functional.normalize(quaternion + torch.randn_like(quaternion) * noise_strengths[0], dim=-1)

    R = quaternion_to_matrix(quaternion)
    T = (1 - fraction) * pose_a[..., :3, 3] + fraction * pose_b[..., :3, 3]
    T = T + torch.randn_like(T) * noise_strengths[1]

    new_pose = pose_a.clone()
    new_pose[..., :3, :3] = R
    new_pose[..., :3, 3] = T
    return new_pose


def sample_from_dense_cameras(dense_cameras, t, noise_strengths=[0, 0, 0, 0]):
    _, N, A, B = dense_cameras.shape
    _, M = t.shape

    t = t.to(dense_cameras.device)
    left = torch.floor(t * (N - 1)).long().clamp(0, N - 2)
    right = left + 1
    fraction = t * (N - 1) - left
    a = torch.gather(dense_cameras, 1, left[..., None].repeat(1, 1, A, B))
    b = torch.gather(dense_cameras, 1, right[..., None].repeat(1, 1, A, B))

    new_pose = sample_from_two_pose(a[:, :, :3, 3:], b[:, :, :3, 3:], fraction, noise_strengths=noise_strengths[:2])

    new_ins = (1 - fraction) * a[:, :, :3, :3] + fraction * b[:, :, :3, :3]

    return torch.cat([new_ins, new_pose], dim=-1)


def export_ply_for_gaussians(path, gaussians):
    xyz, features, opacity, scales, rotations = gaussians

    means3D = xyz.contiguous().float()
    opacity = opacity.contiguous().float()
    scales = scales.contiguous().float()
    rotations = rotations.contiguous().float()
    shs = features.contiguous().float()  # [N, 1, 3]

    SH_C0 = 0.28209479177387814
    means3D, rotations, shs = adjust_gaussians(means3D, rotations, shs, SH_C0, inverse=False)

    opacity = inverse_sigmoid(opacity)
    scales = torch.log(scales + 1e-8)

    xyzs = means3D.detach().cpu().numpy()
    f_dc = shs.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = opacity.detach().cpu().numpy()
    scales = scales.detach().cpu().numpy()
    rotations = rotations.detach().cpu().numpy()

    l = ["x", "y", "z"]  # noqa: E741
    # All channels except the 3 DC
    for i in range(f_dc.shape[1]):
        l.append("f_dc_{}".format(i))
    l.append("opacity")
    for i in range(scales.shape[1]):
        l.append("scale_{}".format(i))
    for i in range(rotations.shape[1]):
        l.append("rot_{}".format(i))

    dtype_full = [(attribute, "f4") for attribute in l]

    elements = np.empty(xyzs.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyzs, f_dc, opacities, scales, rotations), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, "vertex")

    PlyData([el]).write(path + '.ply')

    plydata = PlyData([el])

    vert = plydata["vertex"]
    sorted_indices = np.argsort(
        -np.exp(vert["scale_0"] + vert["scale_1"] + vert["scale_2"]) / (1 + np.exp(-vert["opacity"]))
    )
    buffer = BytesIO()
    for idx in sorted_indices:
        v = plydata["vertex"][idx]
        position = np.array([v["x"], v["y"], v["z"]], dtype=np.float32)
        scales = np.exp(
            np.array(
                [v["scale_0"], v["scale_1"], v["scale_2"]],
                dtype=np.float32,
            )
        )
        rot = np.array(
            [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]],
            dtype=np.float32,
        )
        color = np.array(
            [
                0.5 + SH_C0 * v["f_dc_0"],
                0.5 + SH_C0 * v["f_dc_1"],
                0.5 + SH_C0 * v["f_dc_2"],
                1 / (1 + np.exp(-v["opacity"])),
            ]
        )
        buffer.write(position.tobytes())
        buffer.write(scales.tobytes())
        buffer.write((color * 255).clip(0, 255).astype(np.uint8).tobytes())
        buffer.write(((rot / np.linalg.norm(rot)) * 128 + 128).clip(0, 255).astype(np.uint8).tobytes())

    with open(path + '.splat', "wb") as f:
        f.write(buffer.getvalue())


def load_ply_for_gaussians(path, device="cpu"):
    plydata = PlyData.read(path)

    xyz = np.stack(
        (
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"]),
        ),
        axis=1,
    )
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    print("Number of points at loading : ", xyz.shape[0])

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    xyz = torch.tensor(xyz, dtype=torch.float, device=device)[None]
    features = torch.tensor(features_dc, dtype=torch.float, device=device).transpose(1, 2)[None]
    opacity = torch.tensor(opacities, dtype=torch.float, device=device)[None]
    scales = torch.tensor(scales, dtype=torch.float, device=device)[None]
    rotations = torch.tensor(rots, dtype=torch.float, device=device)[None]

    opacity = torch.sigmoid(opacity)
    scales = torch.exp(scales)

    SH_C0 = 0.28209479177387814
    xyz, rotations, features = adjust_gaussians(xyz, rotations, features, SH_C0, inverse=True)

    return xyz, features, opacity, scales, rotations


def adjust_gaussians(means, rotations, shs, SH_C0, inverse):
    rot_adjust = torch.tensor(
        [
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0],
        ],
        dtype=torch.float32,
        device=means.device,
    )

    adjustment = torch.tensor(
        R.from_rotvec([0, 0, -45], True).as_matrix(),
        dtype=torch.float32,
        device=means.device,
    )

    rot_adjust = adjustment @ rot_adjust

    if inverse:  # load: convert wxyz --> xyzw (rotation), convert shs to precomputed color
        rot_adjust = rot_adjust.inverse()
        means = einsum(rot_adjust, means, "i j, ... j -> ... i")
        rotations = R.from_quat(rotations[0].detach().cpu().numpy(), scalar_first=True).as_matrix()
        rotations = rot_adjust.detach().cpu().numpy() @ rotations
        rotations = R.from_matrix(rotations).as_quat()
        rotations = torch.from_numpy(rotations)[None].to(dtype=torch.float32, device=means.device)
        shs = 0.5 + shs * SH_C0

    else:  # export: convert xyzw --> wxyz (rotation), convert precomputed color to shs
        means = einsum(rot_adjust, means, "i j, ... j -> ... i")
        rotations = R.from_quat(rotations.detach().cpu().numpy()).as_matrix()
        rotations = rot_adjust.detach().cpu().numpy() @ rotations
        rotations = R.from_matrix(rotations).as_quat()
        x, y, z, w = rearrange(rotations, "g xyzw -> xyzw g")
        rotations = torch.from_numpy(np.stack((w, x, y, z), axis=-1)).to(torch.float32)
        shs = (shs - 0.5) / SH_C0

    return means, rotations, shs


@torch.no_grad()
def export_video(render_fn, save_path, name, dense_cameras, fps=60, num_frames=720, size=512, device="cuda:0"):
    images = []
    depths = []

    for i in tqdm.trange(num_frames, desc="Rendering video..."):
        t = torch.full((1, 1), fill_value=i / num_frames, device=device)

        camera = sample_from_dense_cameras(dense_cameras, t)

        image, depth = render_fn(camera, size, size)

        images.append(process_image(image.reshape(3, size, size)))
        depths.append(process_image(depth.reshape(1, size, size)))

    imageio.mimwrite(os.path.join(save_path, f"{name}.mp4"), images, fps=fps, quality=8, macro_block_size=1)


def process_image(image):
    return image.permute(1, 2, 0).detach().cpu().mul(1 / 2).add(1 / 2).clamp(0, 1).mul(255).numpy().astype(np.uint8)


@torch.no_grad()
def export_mv(render_fn, save_path, dense_cameras, size=256):
    num_views = dense_cameras.shape[1]
    imgs = []
    for i in tqdm.trange(num_views, desc="Rendering images..."):
        image = render_fn(dense_cameras[:, i].unsqueeze(1), size, size)[0]
        path = os.path.join(save_path, "mv_results")
        os.makedirs(path, exist_ok=True)

        path = os.path.join(path, f"refined_render_img_{i}.png")
        image = image.reshape(3, size, size).clamp(-1, 1).add(1).mul(1 / 2)
        imgs.append(image)
        save_image(image, path)

    cmap = plt.get_cmap("hsv")
    num_frames = 8
    num_rows = 2
    num_cols = 4
    figsize = (num_cols * 2, num_rows * 2)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)
    axs = axs.flatten()
    for i in range(num_rows * num_cols):
        if i < num_frames:
            axs[i].imshow((imgs[i].cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8))
            for s in ["bottom", "top", "left", "right"]:
                axs[i].spines[s].set_color(cmap(i / (num_frames)))
                axs[i].spines[s].set_linewidth(5)
            axs[i].set_xticks([])
            axs[i].set_yticks([])
        else:
            axs[i].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "refined_mv_images.pdf"), transparent=True)
