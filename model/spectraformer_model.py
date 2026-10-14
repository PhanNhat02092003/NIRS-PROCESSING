"""Single-task re-export of the Spectraformer baseline from MT-SMART-NIR
(SpectraformerMultiTask in model_multitask.py there), adapted for benchmarks
with one classification OR regression head instead of the three pesticide
pipeline heads (food/detection/concentration).

Reimplementation of Chen, Zhou & Ren 2024, "Spectraformer: deep learning
model for grain spectral qualitative analysis based on transformer
structure", RSC Advances 14(12):8053-8066, DOI: 10.1039/d3ra07708j. The
published model is single-task (grain-variety classification) and its
authors' source code was not released; per the paper's architecture
description it is a CNN front-end (four kernel-3 Conv1d layers followed by
four kernel-1 Conv1d layers) feeding a standard Transformer encoder that
treats the spectrum as a token sequence. Exact hyperparameters (encoder
depth/width, any downsampling) are not specified in the paper and are this
repo's own choice. An AdaptiveAvgPool1d compresses the token sequence to a
fixed length before the Transformer encoder so self-attention cost is
bounded consistently regardless of a dataset's native wavelength count (the
paper evaluates only one fixed-resolution instrument and doesn't address
this).
"""

import torch
import torch.nn as nn


class SpectraformerBackbone(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 8, n_layers: int = 4,
                 n_tokens: int = 128, dropout: float = 0.1):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_model = d_model

        cnn = []
        in_ch = 1
        for _ in range(4):
            cnn += [nn.Conv1d(in_ch, d_model, kernel_size=3, padding=1), nn.ReLU()]
            in_ch = d_model
        for _ in range(4):
            cnn += [nn.Conv1d(in_ch, d_model, kernel_size=1), nn.ReLU()]
        self.cnn_stem = nn.Sequential(*cnn)
        self.pool = nn.AdaptiveAvgPool1d(n_tokens)

        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x if x.dim() == 2 else x.squeeze(1)  # (B, L)
        h = h.unsqueeze(1)                        # (B, 1, L)
        h = self.cnn_stem(h)                      # (B, D, L)
        h = self.pool(h)                          # (B, D, n_tokens)
        h = h.transpose(1, 2) + self.pos_embed     # (B, n_tokens, D)
        h = self.transformer(h)                    # (B, n_tokens, D)
        return h.mean(dim=1)                       # (B, D) mean-pooled representation


class SpectraformerClassifier(nn.Module):
    def __init__(self, num_classes: int, d_model: int = 128, n_heads: int = 8,
                 n_layers: int = 4, n_tokens: int = 128, dropout: float = 0.1):
        super().__init__()
        self.backbone = SpectraformerBackbone(d_model, n_heads, n_layers, n_tokens, dropout)
        self.head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class SpectraformerRegressor(nn.Module):
    def __init__(self, num_targets: int = 1, d_model: int = 128, n_heads: int = 8,
                 n_layers: int = 4, n_tokens: int = 128, dropout: float = 0.1):
        super().__init__()
        self.backbone = SpectraformerBackbone(d_model, n_heads, n_layers, n_tokens, dropout)
        # Plain linear head (no Softplus): targets are log1p+z-score normalized
        # by RegressionNIRSDataset, so negative values are valid in this space.
        self.head = nn.Linear(d_model, num_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))
