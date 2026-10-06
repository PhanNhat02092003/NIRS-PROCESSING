"""B-spline Kolmogorov-Arnold Network, ported from ../MT-SMART-NIR (kan.py
there) for the CSSE food head (model/mt_smartnir_model.py). Separate from
model/classification_model.py's KANClassifier/KANRegressor, which use a
different (Gaussian-RBF) basis -- the two are not interchangeable, this one
exists to match MT-SMART-NIR's own architecture exactly.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        grid_range: tuple = (-1.0, 1.0),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        step = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32) * step + grid_range[0]
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        self.base_activation = nn.SiLU()

        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(
            self.spline_weight,
            mean=0.0,
            std=scale_noise / math.sqrt((grid_size + spline_order) * in_features),
        )
        self.scale_base = scale_base
        self.scale_spline = scale_spline

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)  # (batch, in, 1)
        grid = self.grid

        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()
        for k in range(1, self.spline_order + 1):
            left_num = x - grid[:-(k + 1)]
            left_den = grid[k:-1] - grid[:-(k + 1)]
            right_num = grid[k + 1:] - x
            right_den = grid[k + 1:] - grid[1:-k]

            left = torch.where(left_den != 0, left_num / left_den, torch.zeros_like(left_num))
            right = torch.where(right_den != 0, right_num / right_den, torch.zeros_like(right_num))

            bases = left * bases[..., :-1] + right * bases[..., 1:]
        return bases.contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(self.base_activation(x), self.base_weight * self.scale_base)

        splines = self.b_splines(x)
        splines_flat = splines.view(x.size(0), -1)
        spline_w_flat = self.spline_weight.view(self.out_features, -1) * self.scale_spline
        spline_out = F.linear(splines_flat, spline_w_flat)

        return base_out + spline_out


class KAN(nn.Module):
    """Multi-layer KAN network: stacks KANLinear by a layer_sizes list."""

    def __init__(self, layer_sizes: list, grid_size: int = 5, spline_order: int = 3,
                 grid_range: tuple = (-3.0, 3.0)):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(KANLinear(layer_sizes[i], layer_sizes[i + 1],
                                    grid_size=grid_size, spline_order=spline_order,
                                    grid_range=grid_range))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
