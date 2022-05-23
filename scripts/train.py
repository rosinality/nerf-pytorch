import torch
from torch.nn import functional as F
from tensorfn import load_arg_config, get_logger
from tensorfn.config import instantiate

from nerf.config import NeRFConfig
from nerf.dataset import CachedDataset, collate_data
from nerf import render


def sample_data(loader):
    while True:
        for data in loader:
            yield data


def mse_to_psnr(mse):
    return -10 * torch.log(mse) / torch.log(torch.tensor(10.0))


if __name__ == "__main__":
    device = "cuda"

    conf = load_arg_config(NeRFConfig)

    logger = get_logger(mode=conf.logger)
    logger.info(conf.dict())

    datasource = instantiate(conf.training.datasource)
    bounds = (datasource.bounds.min() * 0.1, datasource.bounds.max() * 0.9)

    if conf.training.ndc:
        bounds = (0, 1)

    dset = CachedDataset(datasource, split="train")
    img_v, pose_v, bound_v = datasource.get_valid()

    model = instantiate(conf.model)
    model = model.to(device)

    optimizer = instantiate(conf.training.optimizer, model.parameters())
    scheduler = instantiate(conf.training.scheduler, optimizer)
    train_loader = conf.training.loader.make(dset, collate_fn=collate_data)
    checker = instantiate(conf.checker)
    checker.catalog(conf)

    render_rays = instantiate(conf.training.render_rays)
    render_rays_eval = instantiate(conf.evaluate.render_rays)

    train_loader = sample_data(train_loader)
    focal_length = dset.poses[0].K[0, 0, 0].item()

    for i in range(conf.training.n_iter + 1):
        rgbs, rays = next(train_loader)
        rgbs = rgbs.to(device)
        rays = rays.to(device)

        out = render.render(
            model,
            rays,
            focal_length,
            render_rays,
            use_viewdirs=True,
            ndc=conf.training.ndc,
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
        scheduler.step()

        if i % conf.log_freq == 0:
            lr = optimizer.param_groups[0]["lr"]

            if "rgb0" in out:
                losses = {
                    "lr": lr,
                    "psnr coarse": psnr0.item(),
                    "psnr fine": psnr.item(),
                }
                checker.log(**losses, step=i)

            else:
                losses = {"lr": lr, "psnr": psnr.item()}
                checker.log(**losses, step=i)

        if i % conf.evaluate.eval_freq == 0:
            valid_rgb, valid_disp = render.render_path(
                model,
                pose_v,
                render_rays_eval,
                ndc=conf.training.ndc,
                bounds=bounds,
                use_viewdirs=True,
                device=device,
                logger=logger,
            )
            psnr = mse_to_psnr(F.mse_loss(valid_rgb, img_v)).item()

            checker.log(**{"valid/psnr": psnr}, step=i)

            checker.checkpoint(
                {
                    "model": model.state_dict(),
                    "conf": conf.dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": i,
                },
                f"ckpt-{str(i).zfill(6)}.pt",
            )
