"""NirMACNet: A Multi-Scale Adaptive Convolutional Network for NIR
spectroscopy, reimplemented from the paper
(baseline-information/regression/NirMACNet/J.UCS Article Template V5.tex,
architecture read off Figures 2-4) as a Stage 2 (concentration prediction)
baseline against SMART-NIR.

Architecture (Fig. 4): a shared stem (Conv1d k=7@64 -> BN+ReLU -> MaxPool k=2)
feeds three parallel branches, each a stack of 3 residual "Subblocks"
(channels 64->128->256) at a fixed kernel size (3, 5, 7 respectively, one per
branch) -- multi-scale feature extraction, no pooling inside the branches
themselves so spatial detail is preserved (paper Section 3.1). Each branch's
final feature map is global-average-pooled to a 256-dim vector; the three are
concatenated (768-dim) and fed to a KAN head (768->256->128->64->32->1,
reusing this project's existing KANLayer from model/regression_model.py).

The Subblock's residual wiring (Fig. 2/3) is the standard He et al. BasicBlock
-- Conv-BN+ReLU-Conv-BN-[+shortcut]-ReLU, with a 1x1-conv+BN projection
shortcut when a Subblock changes channel count (every Subblock in this model
except the first in each branch, per Fig. 4's 64@64 -> 64@128 -> 128@256).
"""
from dataclasses import dataclass, field
from typing import Tuple

import torch
import torch.nn as nn

from model.regression_model import KANLayer


class BasicConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=1, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Subblock(nn.Module):
    """Conv-BN+ReLU-Conv-BN-[+shortcut]-ReLU (Fig. 2/3): a standard residual
    BasicBlock. The shortcut is identity when in_ch == out_ch, otherwise a
    1x1 conv + BN projection (paper: "shortcut connections can be augmented
    with a linear projection layer W_s" when dimensions differ).
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int):
        super().__init__()
        self.conv1 = BasicConv1d(in_ch, out_ch, kernel_size)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.shortcut = (
            nn.Identity() if in_ch == out_ch
            else nn.Sequential(nn.Conv1d(in_ch, out_ch, kernel_size=1), nn.BatchNorm1d(out_ch))
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.bn2(self.conv2(h))
        return self.act(h + self.shortcut(x))


class Branch(nn.Module):
    """One multi-scale branch: 3 stacked Subblocks (channels stem_ch ->
    channels[0] -> channels[1] -> channels[2]) at a fixed kernel size, then
    global average pooling to a channels[-1]-dim feature vector.
    """

    def __init__(self, kernel_size: int, stem_ch: int = 64,
                 channels: Tuple[int, int, int] = (64, 128, 256)):
        super().__init__()
        chs = [stem_ch] + list(channels)
        self.subblocks = nn.ModuleList([
            Subblock(chs[i], chs[i + 1], kernel_size) for i in range(len(channels))
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for sb in self.subblocks:
            x = sb(x)
        return self.pool(x).squeeze(-1)


class NirMACNetKANRegressor(nn.Module):
    """KAN head matching the paper's stated dims (768 -> 256 -> 128 -> 64 ->
    32 -> 1 for the milk dataset's 3-branch x 256 concat), built from this
    project's existing Gaussian-RBF KANLayer.
    """

    def __init__(self, in_dim: int, hidden: Tuple[int, ...], num_targets: int, n_basis: int = 8):
        super().__init__()
        dims = [in_dim] + list(hidden)
        self.layers = nn.ModuleList([
            KANLayer(dims[i], dims[i + 1], n_basis=n_basis) for i in range(len(hidden))
        ])
        self.act = nn.GELU()
        self.out = nn.Linear(dims[-1], num_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = self.act(layer(x))
        return self.out(x)


@dataclass
class NirMACNetConfig:
    stem_ch: int = 64
    branch_channels: Tuple[int, int, int] = (64, 128, 256)
    kernel_sizes: Tuple[int, int, int] = (3, 5, 7)
    kan_hidden: Tuple[int, ...] = field(default_factory=lambda: (256, 128, 64, 32))
    kan_basis: int = 8
    num_targets: int = 1


class NirMACNet(nn.Module):
    def __init__(self, cfg: NirMACNetConfig):
        super().__init__()
        self.cfg = cfg
        self.stem = nn.Sequential(
            nn.Conv1d(1, cfg.stem_ch, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(cfg.stem_ch),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.branches = nn.ModuleList([
            Branch(k, stem_ch=cfg.stem_ch, channels=cfg.branch_channels)
            for k in cfg.kernel_sizes
        ])
        concat_dim = cfg.branch_channels[-1] * len(cfg.kernel_sizes)
        self.head = NirMACNetKANRegressor(concat_dim, cfg.kan_hidden, cfg.num_targets, n_basis=cfg.kan_basis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.stem(x)
        feats = [branch(x) for branch in self.branches]
        feat = torch.cat(feats, dim=-1)
        return self.head(feat)
