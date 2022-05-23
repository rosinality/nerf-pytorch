import torch
from torch import nn
from torch.nn import functional as F


class SinusoidalEncoding(nn.Module):
    def __init__(self, in_dim, include_input, max_freq_log2, n_freq, log_sampling):
        super().__init__()

        self.include_input = include_input

        self.out_dim = 0
        if self.include_input:
            self.out_dim += in_dim

        if log_sampling:
            freq_bands = 2 ** torch.linspace(0, max_freq_log2, steps=n_freq)

        else:
            freq_bands = torch.linspace(2 ** 0, 2 ** max_freq_log2, steps=n_freq)

        self.out_dim += freq_bands.shape[0] * in_dim * 2

        self.register_buffer("freq_bands", freq_bands)

    def forward(self, input):
        shape = input.shape
        input = input.unsqueeze(-1)  # [..., in_dim, 1]
        sin = torch.sin(self.freq_bands * input)  # [..., in_dim, n_freq]
        cos = torch.cos(self.freq_bands * input)

        sin = sin.reshape(*shape[:-1], -1)
        cos = cos.reshape(*shape[:-1], -1)

        if self.include_input:
            out = torch.cat((input.squeeze(-1), sin, cos), -1)

        else:
            out = torch.cat((sin, cos), -1)

        return out


class FeedForward(nn.Module):
    def __init__(self, in_dim, dim, n_layers, skips, activation=F.relu):
        super().__init__()

        self.skips = skips
        self.activation = activation

        self.layers = nn.ModuleList(
            [nn.Linear(in_dim, dim)]
            + [
                nn.Linear(dim, dim)
                if i not in self.skips
                else nn.Linear(dim + in_dim, dim)
                for i in range(n_layers - 1)
            ]
        )

    def forward(self, input):
        out = input

        for i, layer in enumerate(self.layers):
            out = layer(out)
            out = self.activation(out)

            if i in self.skips:
                out = torch.cat((input, out), -1)

        return out


class NeRF(nn.Module):
    def __init__(
        self, in_dim, in_view_dim, dim, out_dim, feedforward, use_viewdirs=False
    ):
        super().__init__()

        self.in_dim = in_dim
        self.in_view_dim = in_view_dim
        self.use_viewdirs = use_viewdirs

        self.feedforward = feedforward

        self.view_linear = nn.Linear(self.in_view_dim + dim, dim // 2)

        if use_viewdirs:
            self.feature_linear = nn.Linear(dim, dim)
            self.alpha_linear = nn.Linear(dim, 1)
            self.rgb_linear = nn.Linear(dim // 2, 3)

        else:
            self.out_linear = nn.Linear(dim, out_dim)

    def forward(self, input):
        input_points, input_views = torch.split(
            input, (self.in_dim, self.in_view_dim), -1
        )
        out = self.feedforward(input_points)

        if self.use_viewdirs:
            alpha = self.alpha_linear(out)
            feature = self.feature_linear(out)
            out = torch.cat((feature, input_views), -1)
            out = F.relu(self.view_linear(out))

            rgb = self.rgb_linear(out)
            out = torch.cat((rgb, alpha), -1)

        else:
            out = self.out_linear(out)

        return out


def batchify(fn, input, chunk):
    if chunk is None:
        return fn(input)

    return torch.cat([fn(inp) for inp in input.split(chunk, 0)], 0)


class ImplicitRepresentation(nn.Module):
    def __init__(self, network, network_fine, embed, embed_dir):
        super().__init__()

        self.network = network
        self.network_fine = network_fine
        self.embed = embed
        self.embed_dir = embed_dir

    def forward(self, points, view_dirs, use_fine_if_avail=False, chunk_size=1024 * 64):
        points_flat = points.reshape(-1, points.shape[-1])
        embeds = self.embed(points_flat)

        if view_dirs is not None:
            view_dirs = view_dirs[:, None].expand(points.shape)
            view_dirs_flat = view_dirs.reshape(-1, view_dirs.shape[-1])
            embed_dirs = self.embed_dir(view_dirs_flat)
            embeds = torch.cat((embeds, embed_dirs), -1)

        network = self.network
        if use_fine_if_avail and self.network_fine is not None:
            network = self.network_fine

        outputs_flat = batchify(network, embeds, chunk_size)
        outputs = outputs_flat.reshape(*points.shape[:-1], outputs_flat.shape[-1])

        return outputs
