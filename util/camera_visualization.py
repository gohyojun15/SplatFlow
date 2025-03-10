# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import matplotlib.pyplot as plt

import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


def get_camera_wireframe(scale: float = 0.3):
    """
    Returns a wireframe of a 3D line-plot of a camera symbol.
    """
    a = 0.5 * torch.tensor([-1, 1, 2])
    b = 0.5 * torch.tensor([1, 1, 2])
    c = 0.5 * torch.tensor([-1, -1, 2])
    d = 0.5 * torch.tensor([1, -1, 2])
    C = torch.zeros(3)
    camera_points = [a, b, d, c, a, C, b, d, C, c, C]
    lines = torch.stack([x.float() for x in camera_points]) * scale
    return lines


def plot_cameras(ax, cameras, color=None):
    """
    Plots a set of `cameras` objects into the maplotlib axis `ax` with
    color `color`.
    """
    cam_wires_canonical = get_camera_wireframe(scale=0.05).cuda()[None]
    cam_trans = cameras.get_world_to_view_transform().inverse()
    cam_wires_trans = cam_trans.transform_points(cam_wires_canonical)
    plot_handles = []
    cmap = plt.get_cmap("hsv")

    for i, wire in enumerate(cam_wires_trans):
        # the Z and Y axes are flipped intentionally here!
        color_ = cmap(i / len(cameras))[:-1] if color is None else color
        x_, z_, y_ = wire.detach().cpu().numpy().T.astype(float)
        (h,) = ax.plot(x_, y_, z_, color=color_, linewidth=0.5)
        plot_handles.append(h)
    return cam_wires_trans


def plot_camera_scene(cameras, cameras_gt, path):
    """
    Plots a set of predicted cameras `cameras` and their corresponding
    ground truth locations `cameras_gt`. The plot is named with
    a string passed inside the `status` argument.
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.clear()

    points = plot_cameras(ax, cameras)
    points_gt = plot_cameras(ax, cameras_gt, color=(0.1, 0.1, 0.1))
    tot_pts = torch.cat([points[:, -1], points_gt[:, -1]], dim=0)

    max_scene = tot_pts.max(dim=0)[0].cpu()
    min_scene = tot_pts.min(dim=0)[0].cpu()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xlim3d([min_scene[0] - 0.1, max_scene[0] + 0.3])
    ax.set_ylim3d([min_scene[2] - 0.1, max_scene[2] + 0.1])
    ax.set_zlim3d([min_scene[1] - 0.1, max_scene[1] + 0.1])

    ax.invert_yaxis()

    plt.savefig(os.path.join(path, "pose.pdf"), bbox_inches="tight", pad_inches=0, transparent=True)