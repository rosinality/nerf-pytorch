from tensorfn.config.builder import F, L, field

from nerf import model, render, datasource

conf = field()

n_freq_rgb = 10
in_dim_rgb = n_freq_rgb * 3 * 2 + 3
n_freq_dir = 4
in_dim_dir = n_freq_dir * 3 * 2 + 3
dim = 256
out_dim = 5
n_layer = 8

point_embed = L[model.SinusoidalEncoding](3, True, n_freq_rgb - 1, n_freq_rgb, True)
dir_embed = L[model.SinusoidalEncoding](3, True, n_freq_dir - 1, n_freq_dir, True)
feedforward = L[model.FeedForward](in_dim_rgb, dim, n_layer, [4])
nerf = L[model.NeRF](
    in_dim_rgb, in_dim_dir, dim, out_dim, feedforward, use_viewdirs=True
)
conf.model = L[model.ImplicitRepresentation](nerf, nerf, point_embed, dir_embed)

lr = 5e-4
n_iter = 200_000
schedule_iter = 250_000
batch_size = 1024
n_samples = 64
n_importance = 64

datasource = L[datasource.LLFF]("data/fern", downsample=8, llff_hold=8)
render_rays = F[render.render_rays](
    n_samples=n_samples, n_importance=n_importance, noise_std=1, perturb=1
)
optimizer = field(type="adam", lr=lr)
scheduler = field(
    type="cycle",
    lr=lr,
    n_iter=schedule_iter,
    final_multiplier=0.1,
    warmup=0,
    decay=("linear", "exp"),
)
loader = field(batch_size=batch_size, shuffle=True, num_workers=1)

conf.training = field(
    datasource=datasource,
    loader=loader,
    optimizer=optimizer,
    scheduler=scheduler,
    render_rays=render_rays,
    ndc=True,
    n_iter=n_iter,
)

conf.evaluate = field(
    eval_freq=10000,
    render_rays=F[render.render_rays](
        n_samples=n_samples, n_importance=n_importance, noise_std=0, perturb=False
    ),
)

conf.checker = field(
    storage=[field(type="local", path="checkpoints")],
    reporter=[
        field(type="logger"),
    ],
)
