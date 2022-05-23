import numpy as np
import torch
from torch.utils import data

from nerf import render


class CachedDataset(data.Dataset):
    def __init__(self, datasource, split="train"):
        self.datasource = datasource

        if split == "train":
            images, poses, _ = datasource.get_train()

        elif split == "valid":
            images, poses, _ = datasource.get_valid()

        rays = torch.stack([render.get_rays(p).stack() for p in poses])
        height, width = rays.shape[2], rays.shape[3]
        rays_rgb = torch.cat((rays, images[:, None]), 1)  # N, ro + rd + rgb, H, W, 3
        rays_rgb = rays_rgb.permute(0, 2, 3, 1, 4)  # N, H, W, ro + rd + rgb, 3
        rays_rgb = rays_rgb.reshape(-1, 3, 3)  # N * H * W, ro + rd + rgb, 3
        origin, direction, rgb = rays_rgb.unbind(1)

        height, width = poses.image_size[0].tolist()

        self.rays = render.Ray(origin, direction, height, width)
        self.rgb = rgb
        self.poses = poses

    def __getitem__(self, index):
        return self.rgb[index], self.rays[index]

    def __len__(self):
        return self.rgb.shape[0]


def collate_data(batch):
    rgbs = []
    rays = []

    for rgb, ray in batch:
        rgbs.append(rgb)
        rays.append(ray)

    rgbs = torch.stack(rgbs, 0)
    rays = render.stack_rays(rays)

    return rgbs, rays
