import os
import math

import torch
from torch.nn import functional as F
import numpy as np
from PIL import Image

from nerf.camera import PerspectiveCamera


def imread(path):
    return torch.as_tensor(
        np.array(Image.open(path).convert("RGB")) / 255, dtype=torch.float32
    )


def load_data(path, downsample=1):
    poses_ar = torch.from_numpy(np.load(os.path.join(path, "poses_bounds.npy")))
    poses = poses_ar[:, :-2].reshape(-1, 3, 5)
    bounds = poses_ar[:, -2:]

    if downsample > 1:
        imgdir = os.path.join(path, f"images_{downsample}")

    else:
        imgdir = os.path.join(path, "images")

    imgfiles = [
        os.path.join(imgdir, f)
        for f in sorted(os.listdir(imgdir))
        if os.path.splitext(f)[-1].lower() in (".jpg", ".png")
    ]

    if poses.shape[0] != len(imgfiles):
        raise RuntimeError("Number of images and poses does not match")

    shape = imread(imgfiles[0]).shape
    poses[:, :2, 4] = torch.as_tensor(shape[:2])  # .reshape(2, 1)
    poses[:, 2, 4] = poses[:, 2, 4] / downsample

    imgs = [imread(f) for f in imgfiles]
    imgs = torch.stack(imgs, 0)

    # 3, 5, batch_size
    poses = torch.cat((poses[:, :, 1:2], -poses[:, :, 0:1], poses[:, :, 2:]), 2).to(
        torch.float32
    )

    poses = PerspectiveCamera.from_compact_representation(poses)

    return imgs, poses, bounds


def normalize(x):
    return F.normalize(x, dim=0)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(torch.cross(vec1_avg, vec2))
    vec1 = normalize(torch.cross(vec2, vec0))
    mat = torch.stack((vec0, vec1, vec2, pos), 1)

    return mat


def poses_avg(poses):
    center = poses.T.mean(0)
    vec2 = normalize(poses.R[:, :, 2].sum(0))
    up = poses.R[:, :, 1].sum(0)
    hwf = torch.cat(
        (poses.image_size[0], poses.focal_length[0].unsqueeze(0)), 0
    ).unsqueeze(-1)
    cam2world = torch.cat((viewmatrix(vec2, up, center), hwf), 1)

    return cam2world


def recenter_poses(poses):
    poses_ = poses.clone()
    bottom = torch.tensor((0, 0, 0, 1), dtype=torch.float32).unsqueeze(0)
    cam2world = poses_avg(poses)
    cam2world = torch.cat((cam2world[:3, :4], bottom), -2)

    poses = torch.inverse(cam2world) @ poses.camera_to_world()
    poses_.R[:] = poses[:, :3, :3]
    poses_.T = poses[:, :3, 3]

    return poses_


def spiral_render_path(cam2world, up, rads, focal, zdelta, zrate, rots, n_views):
    render_poses = []
    rads = torch.tensor(list(rads) + [1.0])
    hwf = cam2world[:, 4:5]

    for theta in torch.linspace(0, 2 * math.pi * rots, n_views + 1).tolist()[:-1]:
        c = cam2world[:3, :4] @ (
            torch.tensor(
                (math.cos(theta), -math.sin(theta), -math.sin(theta * zrate), 1)
            )
            * rads
        )
        z = normalize(
            c
            - (cam2world[:3, :4] @ torch.tensor((0, 0, -focal, 1), dtype=torch.float32))
        )
        render_poses.append(torch.cat((viewmatrix(z, up, c), hwf), 1))

    return render_poses


class LLFF:
    def __init__(
        self,
        path,
        downsample,
        recenter=True,
        bounds_factor=0.75,
        spherify=False,
        path_zflat=False,
        llff_hold=0,
    ):
        self.path_zflat = path_zflat

        self.images, self.poses, self.bounds = load_data(path, downsample=downsample)

        if bounds_factor is not None:
            scale = 1 / (self.bounds.min() * bounds_factor)
            self.poses.T *= scale
            self.bounds *= scale

        if recenter:
            self.poses = recenter_poses(self.poses)

        self.render_poses = self.make_render_poses()

        if llff_hold > 0:
            self.holdout_index = torch.arange(self.images.shape[0]).tolist()[
                ::llff_hold
            ]

        else:
            cam2world = poses_avg(self.poses)
            self.holdout_index = [
                torch.sum(torch.square(cam2world[:3, 3] - self.poses.T), -1)
                .argmin()
                .item()
            ]

    def __len__(self):
        return self.images.shape[0]

    def make_render_poses(self):
        cam2world = poses_avg(self.poses)
        up = normalize(self.poses.R[:, :, 1].sum(0))
        close_depth, inf_depth = self.bounds.min() * 0.9, self.bounds.max() * 5
        dt = 0.75
        mean_dz = 1 / (((1 - dt) / close_depth + dt / inf_depth))
        focal = mean_dz

        zdelta = close_depth * 0.2
        tt = self.poses.T
        rads = torch.quantile(torch.abs(tt), 0.9, dim=0)
        cam2world_path = cam2world
        n_views = 120
        n_rots = 2

        if self.path_zflat:
            zloc = -close_depth * 0.1
            cam2world_path[:3, :3] = (
                cam2world_path[:3, 3] + zloc * cam2world_path[:3, 2]
            )
            rads[2] = 0
            n_rots = 1
            n_views /= 2

        render_poses = torch.stack(
            spiral_render_path(
                cam2world_path,
                up,
                rads,
                focal,
                zdelta,
                zrate=0.5,
                rots=n_rots,
                n_views=n_views,
            ),
            0,
        )

        render_poses = PerspectiveCamera.from_compact_representation(render_poses)

        return render_poses

    def __getitem__(self, index):
        return self.images[index], self.poses[index], self.bounds[index]

    def get_all(self):
        return self.images, self.poses, self.bounds

    def get_train(self):
        index = torch.arange(len(self)).tolist()
        index = [i for i in index if i not in self.holdout_index]

        return self.images[index], self.poses[index], self.bounds[index]

    def get_valid(self):
        index = torch.arange(len(self)).tolist()
        index = [i for i in index if i in self.holdout_index]

        return self.images[index], self.poses[index], self.bounds[index]
