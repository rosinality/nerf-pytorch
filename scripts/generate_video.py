import argparse

import imageio
import torch
from torch.nn import functional as F
from tensorfn import load_arg_config, get_logger
from tensorfn.config import instantiate
import numpy as np
from matplotlib import pyplot as plt

from nerf.config import NeRFConfig
from nerf import render


def to_rgb(x):
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)


if __name__ == "__main__":
    device = "cuda"

    conf = load_arg_config(NeRFConfig)

    logger = get_logger(mode=conf.logger)

    datasource = instantiate(conf.training.datasource)
    bounds = (datasource.bounds.min() * 0.1, datasource.bounds.max() * 0.9)

    if conf.training.ndc:
        bounds = (0, 1)

    render_poses = datasource.make_render_poses()

    model = instantiate(conf.model)
    model = model.to(device)

    ckpt = torch.load(conf.ckpt, map_location=lambda storage, loc: storage)
    model.load_state_dict(ckpt["model"])

    render_rays_eval = instantiate(conf.evaluate.render_rays)

    rgbs, disps = render.render_path(
        model,
        render_poses,
        render_rays_eval,
        ndc=conf.training.ndc,
        bounds=bounds,
        use_viewdirs=True,
        device=device,
        logger=logger,
    )
    rgbs = rgbs.cpu().numpy()
    disps = disps.cpu().numpy()

    imageio.mimwrite(
        conf.out,
        np.concatenate(
            (to_rgb(rgbs), to_rgb(plt.cm.viridis(disps / disps.max())[..., :3])), 2
        ),
        fps=30,
        quality=8,
    )
