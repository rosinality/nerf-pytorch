from functools import partial
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils import data
from tqdm import tqdm

import datasource
import render
from dataset import CachedDataset, collate_data
import model


def sample_data(loader):
    while True:
        for data in loader:
            yield data


def mse_to_psnr(mse):
    return -10 * torch.log(mse) / torch.log(torch.tensor(10.0))


if __name__ == "__main__":
    device = "cuda"
    data_path = "ramen"
    ndc = True
    downsample = 4

    llff_data = datasource.LLFF(data_path, downsample, llff_hold=8)

    if ndc:
        bounds = (0, 1)

    else:
        bounds = (llff_data.bounds.min() * 0.1, llff_data.bounds.max() * 0.9)
        print("bounds:", bounds)

    dset = CachedDataset(llff_data)

    point_embed = model.SinusoidalEncoding(3, True, 9, 10, True)
    viewdir_embed = model.SinusoidalEncoding(3, True, 3, 4, True)
    feedforward = model.FeedForward(63, 256, 8, [4])
    nerf_coarse = model.NeRF(63, 27, 256, 5, feedforward, use_viewdirs=True)
    feedforward = model.FeedForward(63, 256, 8, [4])
    nerf_fine = model.NeRF(63, 27, 256, 5, feedforward, use_viewdirs=True)
    nerf = model.ImplicitRepresentation(
        nerf_coarse, nerf_fine, point_embed, viewdir_embed
    )
    nerf = nerf.to(device)

    optimizer = optim.Adam(nerf.parameters(), lr=5e-4)

    render_rays = partial(
        render.render_rays, n_samples=64, n_importance=64, noise_std=1, perturb=1
    )

    train_loader = data.DataLoader(
        dset, batch_size=1024, shuffle=True, num_workers=1, collate_fn=collate_data
    )

    pbar = tqdm(range(200000 + 1), dynamic_ncols=True)

    train_loader = sample_data(train_loader)
    focal_length = dset.poses[0].K[0, 0, 0].item()
    decay_step = 250_000
    decay_rate = 0.1

    for i in pbar:
        rgbs, rays = next(train_loader)
        rgbs = rgbs.to(device)
        rays = rays.to(device)

        out = render.render(
            nerf,
            rays,
            focal_length,
            render_rays,
            use_viewdirs=True,
            ndc=ndc,
            bounds=bounds,
        )
        optimizer.zero_grad()
        loss = F.mse_loss(out["rgb"], rgbs)
        psnr = mse_to_psnr(loss.detach())

        if "rgb0" in out:
            loss0 = F.mse_loss(out["rgb0"], rgbs)
            psnr0 = mse_to_psnr(loss0.detach())
            loss = loss + loss0

        loss.backward()
        optimizer.step()

        lr = 5e-4 * (decay_rate ** (i / decay_step))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if "rgb0" in out:
            pbar.set_description(
                f"lr: {lr:.6f}; psnr coarse: {psnr0.item():.3f}; psnr fine: {psnr.item():.3f}"
            )
        else:
            pbar.set_description(f"lr: {lr:.6f}; psnr: {psnr.item():.3f}")

        if i % 10000 == 0:
            torch.save({"model": nerf.state_dict()}, f"ckpt-{str(i).zfill(6)}.pt")
