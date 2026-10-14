"""Single-task re-export of the ResNet-1D baseline from MT-SMART-NIR
(ResNet1DMultiTask in model_multitask.py there), adapted for benchmarks
with one classification OR regression head instead of the three pesticide
pipeline heads (food/detection/concentration).

Built in the spirit of Pan et al. 2025, "ResNet + multi-task-learning
architecture for near-infrared spectroscopy", Spectrochimica Acta Part A
333:125866, DOI: 10.1016/j.saa.2025.125866 -- a shared ResNet backbone with
multiple task-specific output heads. The paper doesn't release source code
or specify exact layer/channel counts, so the stem + four residual stages
(doubling channels each stage) below is this repo's own ResNet-1D design,
following the same general shared-backbone-plus-multiple-outputs principle.
"""

import torch
import torch.nn as nn


class _ResBlock1D(nn.Module):
    """Basic 1D residual block: two Conv1d(k=3)+BN+ReLU with a skip connection."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + identity)


class ResNet1DBackbone(nn.Module):
    def __init__(self, base_channels: int = 64, blocks_per_stage: int = 2, dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        stages, in_ch = [], base_channels
        for stage_idx in range(4):
            out_ch = base_channels * (2 ** stage_idx)
            for b in range(blocks_per_stage):
                stride = 2 if (b == 0 and stage_idx > 0) else 1
                stages.append(_ResBlock1D(in_ch, out_ch, stride=stride))
                in_ch = out_ch
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.out_channels = in_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x if x.dim() == 2 else x.squeeze(1)  # (B, L)
        h = h.unsqueeze(1)          # (B, 1, L)
        h = self.stem(h)
        h = self.stages(h)
        h = self.pool(h).flatten(1)  # (B, C)
        return self.dropout(h)


class ResNet1DClassifier(nn.Module):
    def __init__(self, num_classes: int, base_channels: int = 64, blocks_per_stage: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.backbone = ResNet1DBackbone(base_channels, blocks_per_stage, dropout)
        self.head = nn.Sequential(
            nn.Linear(self.backbone.out_channels, 64), nn.ReLU(), nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class ResNet1DRegressor(nn.Module):
    def __init__(self, num_targets: int = 1, base_channels: int = 64, blocks_per_stage: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.backbone = ResNet1DBackbone(base_channels, blocks_per_stage, dropout)
        # Plain linear head (no Softplus): targets are log1p+z-score normalized
        # by RegressionNIRSDataset, so negative values are valid in this space.
        self.head = nn.Linear(self.backbone.out_channels, num_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))
