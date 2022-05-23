import torch
from torch.nn import functional as F


def check_same(x):
    return all(x[0] == elem for elem in x)


def stack_rays(rays):
    assert check_same([r.height for r in rays])
    assert check_same([r.width for r in rays])

    origin = torch.stack([r.origin for r in rays], 0)
    direction = torch.stack([r.direction for r in rays], 0)

    return Ray(origin, direction, rays[0].height, rays[0].width)


class Ray:
    def __init__(self, origin, direction, height, width, ndc=False):
        self.origin = origin
        self.direction = direction
        self.height = height
        self.width = width
        self.ndc = ndc

    def __len__(self):
        return self.origin.shape[0]

    def stack(self):
        return torch.stack((self.origin, self.direction), 0)

    def split(self, size):
        origins = self.origin.split(size, dim=0)
        directions = self.direction.split(size, dim=0)

        return [
            Ray(origin, direction, self.height, self.width, self.ndc)
            for origin, direction in zip(origins, directions)
        ]

    def flatten(self):
        return Ray(
            self.origin.reshape(-1, 3),
            self.direction.reshape(-1, 3),
            self.height,
            self.width,
            self.ndc,
        )

    def __getitem__(self, index):
        return Ray(
            self.origin[index], self.direction[index], self.height, self.width, self.ndc
        )

    def to(self, *args, **kwargs):
        return Ray(
            self.origin.to(*args, **kwargs),
            self.direction.to(*args, **kwargs),
            self.height,
            self.width,
            ndc=self.ndc,
        )

    def to_ndc(self, focal, near):
        if self.ndc:
            return self

        t = -(near + self.origin[..., 2]) / self.direction[..., 2]
        origin = self.origin + t[..., None] * self.direction

        o0 = -1 / (self.width / (2 * focal)) * origin[..., 0] / origin[..., 2]
        o1 = -1 / (self.height / (2 * focal)) * origin[..., 1] / origin[..., 2]
        o2 = 1 + 2 * near / origin[..., 2]

        d0 = (
            -1
            / (self.width / (2 * focal))
            * (
                self.direction[..., 0] / self.direction[..., 2]
                - origin[..., 0] / origin[..., 2]
            )
        )
        d1 = (
            -1
            / (self.height / (2 * focal))
            * (
                self.direction[..., 1] / self.direction[..., 2]
                - origin[..., 1] / origin[..., 2]
            )
        )
        d2 = -2 * near / origin[..., 2]

        origin = torch.stack((o0, o1, o2), -1)
        direction = torch.stack((d0, d1, d2), -1)

        return Ray(origin, direction, self.height, self.width, ndc=True)


def get_rays(poses):
    height, width = poses.image_size[0].tolist()

    i, j = torch.meshgrid(
        torch.arange(width, dtype=torch.float32),
        torch.arange(height, dtype=torch.float32),
        indexing="xy",
    )

    K = poses.K[0]

    # K:
    # [focal_length,            0, center_x]
    # [           0, focal_length, center_y]
    # [           0,            0,        1]

    dirs = torch.stack(
        ((i - K[0, 2]) / K[0, 0], -(j - K[1, 2]) / K[1, 1], -torch.ones_like(i)), -1
    )
    rays_d = torch.sum(dirs.unsqueeze(-2) * poses.R, -1)
    rays_o = poses.T[0].expand_as(rays_d)

    return Ray(rays_o, rays_d, height, width)


def render(
    query_fn,
    rays,
    focal_length,
    render_rays,
    bounds=(0, 1),
    chunk_size=1024 * 32,
    ndc=True,
    use_viewdirs=False,
):
    viewdirs = None
    if use_viewdirs:
        viewdirs = rays.direction
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = viewdirs.reshape(-1, 3)

    shape = rays.direction.shape

    if ndc:
        rays = rays.to_ndc(focal_length, 1)

    rays = rays.flatten()

    near, far = bounds
    multiplier = torch.ones_like(rays.direction[..., :1])
    near = near * multiplier
    far = far * multiplier

    ray_chunks = rays.split(chunk_size)
    near_chunks = near.split(chunk_size, dim=0)
    far_chunks = far.split(chunk_size, dim=0)

    if use_viewdirs:
        viewdirs = viewdirs.split(chunk_size, dim=0)

    else:
        viewdirs = [None] * len(ray_chunks)

    all_outputs = {}
    for ray_c, near_c, far_c, viewdir in zip(
        ray_chunks, near_chunks, far_chunks, viewdirs
    ):
        outputs = render_rays(query_fn, ray_c, viewdir, near_c, far_c)

        for k, v in outputs.items():
            if k not in all_outputs:
                all_outputs[k] = []

            all_outputs[k].append(v)

    all_outputs = {k: torch.cat(v, 0) for k, v in all_outputs.items()}

    for k, v in all_outputs.items():
        v_shape = list(shape[:-1]) + list(v.shape[1:])
        all_outputs[k] = v.reshape(v_shape)

    return all_outputs


def render_rays(
    query_fn,
    ray_c,
    viewdirs,
    near_c,
    far_c,
    n_samples,
    n_importance,
    inv_depth=False,
    perturb=0,
    noise_std=0,
    white_background=False,
):
    points, interp = ray_sampling(ray_c, near_c, far_c, n_samples, inv_depth, perturb)
    out = query_fn(points, viewdirs)
    rgb_map, disp_map, acc_map, weights, depth_map = postprocess(
        out, interp, ray_c.direction, noise_std, white_background
    )

    if n_importance > 0:
        rgb_map0, disp_map0, acc_map0 = rgb_map, disp_map, acc_map

        points, interp = importance_sampling(
            interp, weights, ray_c, n_importance, perturb == 0
        )
        out = query_fn(points, viewdirs, use_fine_if_avail=True)

        rgb_map, disp_map, acc_map, weights, depth_map = postprocess(
            out, interp, ray_c.direction, noise_std, white_background
        )

    outputs = {"rgb": rgb_map, "disp": disp_map, "acc": acc_map, "out": out}

    if n_importance > 0:
        outputs.update(
            {
                "rgb0": rgb_map0,
                "disp0": disp_map0,
                "acc0": acc_map0,
                "interp_std": torch.std(interp, dim=-1, unbiased=False),
            }
        )

    return outputs


def ray_sampling(rays, near, far, n_samples, inv_depth=False, perturb=0):
    interval = torch.linspace(0, 1, n_samples, device=rays.origin.device)

    if inv_depth:
        interp = 1 / (1 / near * (1 - interval) + 1 / far * (interval))

    else:
        interp = near * (1 - interval) + far * interval

    n_rays = rays.origin.shape[0]

    interp = interp.expand(n_rays, n_samples)

    if perturb > 0:
        mids = 0.5 * (interp[..., 1:] + interp[..., :-1])
        upper = torch.cat((mids, interp[..., -1:]), -1)
        lower = torch.cat((interp[..., :1], mids), -1)
        interp_rand = torch.rand(interp.shape, device=interp.device)
        interp = lower + (upper - lower) * interp_rand

    points = (
        rays.origin[..., None, :] + rays.direction[..., None, :] * interp[..., :, None]
    )

    return points, interp


def to_alpha(out, dist, activation=F.relu):
    return 1 - torch.exp(-activation(out) * dist)


def postprocess(out, interpolation, direction, noise_std=0, white_background=False):
    dists = interpolation[..., 1:] - interpolation[..., :-1]
    dists = torch.cat(
        (dists, torch.tensor([1e10], device=dists.device).expand(dists[..., :1].shape)),
        -1,
    )
    dists = dists * torch.norm(direction[..., None, :], dim=-1)

    rgb = torch.sigmoid(out[..., :3])

    noise = 0
    if noise_std > 0:
        noise = torch.randn(out[..., 3].shape, device=out.device) * noise_std

    alpha = to_alpha(out[..., 3] + noise, dists)
    weights = alpha * torch.cumprod(
        torch.cat(
            (torch.ones((alpha.shape[0], 1), device=out.device), 1 - alpha + 1e-10),
            -1,
        ),
        -1,
    )[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)
    depth_map = torch.sum(weights * interpolation, -1)
    disp_map = 1 / (depth_map / torch.sum(weights, -1)).clamp(min=1e-10)
    acc_map = torch.sum(weights, -1)

    if white_background:
        rgb_map = rgb_map + (1 - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map


@torch.no_grad()
def importance_sampling(interpolation, weights, rays, n_samples, deterministic):
    mids = 0.5 * (interpolation[..., 1:] + interpolation[..., :-1])
    samples = sample_pdf(
        mids, weights[..., 1:-1], n_samples, deterministic=deterministic
    )
    interp, _ = torch.sort(torch.cat((interpolation, samples), -1), -1)
    points = (
        rays.origin[..., None, :] + rays.direction[..., None, :] * interp[..., :, None]
    )

    return points, interp


def sample_pdf(bins, weights, n_samples, deterministic=False):
    weights = weights + 1e-5
    pdf = weights / weights.sum(-1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat((torch.zeros_like(cdf[..., :1]), cdf), -1)

    if deterministic:
        u = torch.linspace(0, 1, steps=n_samples, device=weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])

    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device=weights.device)

    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = (inds - 1).clamp(min=0)
    above = inds.clamp(max=cdf.shape[-1] - 1)
    inds_g = torch.stack((below, above), -1)  # batch, n_samples, 2

    matched_shape = (inds_g.shape[0], inds_g.shape[1], cdf.shape[-1])
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


@torch.no_grad()
def render_path(
    query_fn,
    poses,
    render_rays,
    use_viewdirs=False,
    bounds=(0, 1),
    ndc=True,
    device="cuda",
    logger=None,
):
    focal = poses.focal_length[0].item()

    rgbs = []
    disps = []

    for i, pose in enumerate(poses):
        if logger is not None:
            logger.info(f"rendering... [{i + 1}/{len(poses)}]")

        rays = get_rays(pose).to(device)
        out = render(
            query_fn,
            rays,
            focal,
            render_rays,
            use_viewdirs=use_viewdirs,
            bounds=bounds,
            ndc=ndc,
        )
        rgbs.append(out["rgb"].cpu())
        disps.append(out["disp"].cpu())

    rgbs = torch.stack(rgbs, 0)
    disps = torch.stack(disps, 0)

    return rgbs, disps
