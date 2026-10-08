"""Cross-Scale Spectral Encoder (CSSE), ported from ../MT-SMART-NIR
(model_multitask.py there), with the pesticide-detection head removed for
datasets like Mango where the regression target is always measured (no
presence/absence concept) -- only a food-classification head and a single
continuous-regression head remain.

Architecture (unchanged from MT-SMART-NIR):
  NIR spectrum (L-dim)
      -> 4 parallel Conv2d branches (kernels 4/8/16/32, stride 4), one token
         stream per scale
      -> factorised positional + scale embeddings
      -> N x CrossScaleEncoderLayer (shared weights across all 4 scales):
           within-scale MHSA -> ScaleInteractionModule (4x4 learned mixing)
           -> LocalGatedFFN (SwiGLU gate + depthwise Conv1d)
      -> Attentive Scale Fusion (softmax-weighted sum of 4 scale summaries)
      -> LayerNorm + KAN([D, 32, 16, n_food])  -> food logits
      -> Linear(D, 1)                           -> regression target
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.kan_bspline import KAN

KERNEL_CONFIGS = [(4, 0), (8, 3), (16, 7), (32, 15)]  # (kernel_h, padding_h)
N_SCALES = 4


def _compute_h_out(seq_len: int) -> int:
    k, p = KERNEL_CONFIGS[0]
    return (seq_len + 2 * p - k) // 4 + 1


class LocalGatedFFN(nn.Module):
    """SwiGLU gate + depthwise Conv1d local spectral mixing (see
    ../MT-SMART-NIR/model_multitask.py for the full rationale)."""

    def __init__(self, d_model: int, h_hidden: int, kernel_size: int = 3, expand_ratio: float = 0.5):
        super().__init__()
        E = max(int(h_hidden * expand_ratio), 8)
        self.gate_up = nn.Linear(d_model, 2 * E, bias=False)
        self.dw_conv = nn.Conv1d(E, E, kernel_size=kernel_size, padding=kernel_size // 2, groups=E, bias=False)
        self.down = nn.Linear(E, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v, g = self.gate_up(x).chunk(2, dim=-1)
        v = self.dw_conv(v.transpose(1, 2)).transpose(1, 2)
        return self.down(v * F.silu(g))


class ScaleInteractionModule(nn.Module):
    """4-way cross-scale mixing of mean-pooled per-scale summaries, broadcast
    back as a residual (see ../MT-SMART-NIR/model_multitask.py, Eq. 3-5)."""

    def __init__(self, d_model: int, n_scales: int = N_SCALES):
        super().__init__()
        self.n_scales = n_scales
        self.mix = nn.Parameter(torch.eye(n_scales))
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, scale_tokens: list) -> list:
        summaries = torch.stack([z.mean(dim=1) for z in scale_tokens], dim=1)  # [B, 4, D]
        summaries = self.ln(summaries)
        mix = torch.softmax(self.mix, dim=-1)
        mixed = torch.einsum("sk,bkd->bsd", mix, summaries)
        mixed = self.proj(mixed)
        return [scale_tokens[k] + mixed[:, k:k + 1, :] for k in range(self.n_scales)]


class CrossScaleEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, h_hidden: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln_attn = nn.LayerNorm(d_model)
        self.si = ScaleInteractionModule(d_model)
        self.ffn = LocalGatedFFN(d_model, h_hidden)
        self.ln_ffn = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, scale_tokens: list) -> list:
        after_sa = []
        for z in scale_tokens:
            z_n = self.ln_attn(z)
            out, _ = self.attn(z_n, z_n, z_n)
            after_sa.append(z + self.drop(out))

        after_si = self.si(after_sa)

        after_ffn = []
        for z in after_si:
            after_ffn.append(z + self.drop(self.ffn(self.ln_ffn(z))))
        return after_ffn


class CrossScaleSpectralEncoder(nn.Module):
    def __init__(self, c_out: int = 64, n_layers: int = 6, n_heads: int = 6,
                 h_hidden: int = 128, seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.conv_branches = nn.ModuleList([
            nn.Conv2d(1, c_out, kernel_size=(k, 1), stride=(4, 1), padding=(p, 0))
            for k, p in KERNEL_CONFIGS
        ])

        self.d_model = math.ceil(4 * c_out / n_heads) * n_heads
        D = self.d_model
        self.scale_proj = nn.Linear(c_out, D)

        h_out = _compute_h_out(seq_len)
        self.pos_embed = nn.Parameter(torch.zeros(h_out, D))
        self.scale_embed = nn.Parameter(torch.zeros(N_SCALES, D))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.scale_embed, std=0.02)

        self.layers = nn.ModuleList([
            CrossScaleEncoderLayer(D, n_heads, h_hidden, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(D)
        self.scale_gate = nn.Linear(D * N_SCALES, N_SCALES, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2d = x.unsqueeze(1).unsqueeze(-1)  # [B, 1, L, 1]

        scale_tokens = []
        for k, conv in enumerate(self.conv_branches):
            feat = conv(x2d).squeeze(-1).permute(0, 2, 1)  # [B, H_out, c_out]
            z = self.scale_proj(feat) + self.pos_embed + self.scale_embed[k]
            scale_tokens.append(z)

        for layer in self.layers:
            scale_tokens = layer(scale_tokens)
        scale_tokens = [self.norm(z) for z in scale_tokens]

        summaries = [z.mean(dim=1) for z in scale_tokens]
        weights = torch.softmax(self.scale_gate(torch.cat(summaries, dim=-1)), dim=-1)
        stacked = torch.stack(summaries, dim=1)
        return (stacked * weights.unsqueeze(-1)).sum(dim=1)


class MangoCSSEModel(nn.Module):
    """CSSE encoder with two heads: food (cultivar) classification and a
    single continuous regression target (dry_matter) -- no detection head,
    since dry_matter is always measured for every sample."""

    def __init__(self, n_food_classes: int, c_out: int = 64, n_layers: int = 6,
                 n_heads: int = 6, h_hidden: int = 128, seq_len: int = 512,
                 use_kan: bool = True, dropout: float = 0.1):
        super().__init__()
        self.encoder = CrossScaleSpectralEncoder(c_out, n_layers, n_heads, h_hidden, seq_len, dropout)
        D = self.encoder.d_model

        if use_kan:
            self.food_head = nn.Sequential(nn.LayerNorm(D), KAN([D, 32, 16, n_food_classes]))
        else:
            self.food_head = nn.Sequential(
                nn.Linear(D, 32), nn.GELU(), nn.Linear(32, 16), nn.GELU(), nn.Linear(16, n_food_classes)
            )
        # Plain linear head: the regression target is log1p+z-score normalized
        # by the dataset (see MangoMultiTaskDataset), so negative values are
        # valid in this space -- no Softplus.
        self.reg_head = nn.Linear(D, 1)

    def forward(self, x: torch.Tensor):
        cls = self.encoder(x)
        food_logits = self.food_head(cls)
        reg_pred = self.reg_head(cls).squeeze(-1)  # (B,)
        return food_logits, reg_pred


class GrainitCSSEModel(nn.Module):
    """CSSE encoder with three heads: cereal-type classification plus two
    independent continuous regression targets (Moisture, Protein) -- still
    no detection head, since both targets are always-measured grain
    properties (see dataset/grainit_multitask_dataset.py)."""

    def __init__(self, n_food_classes: int, c_out: int = 64, n_layers: int = 6,
                 n_heads: int = 6, h_hidden: int = 128, seq_len: int = 512,
                 use_kan: bool = True, dropout: float = 0.1):
        super().__init__()
        self.encoder = CrossScaleSpectralEncoder(c_out, n_layers, n_heads, h_hidden, seq_len, dropout)
        D = self.encoder.d_model

        if use_kan:
            self.food_head = nn.Sequential(nn.LayerNorm(D), KAN([D, 32, 16, n_food_classes]))
        else:
            self.food_head = nn.Sequential(
                nn.Linear(D, 32), nn.GELU(), nn.Linear(32, 16), nn.GELU(), nn.Linear(16, n_food_classes)
            )
        # Plain linear heads: both targets are log1p+z-score normalized by
        # the dataset, so negative values are valid in this space -- no Softplus.
        self.moisture_head = nn.Linear(D, 1)
        self.protein_head = nn.Linear(D, 1)

    def forward(self, x: torch.Tensor):
        cls = self.encoder(x)
        food_logits = self.food_head(cls)
        moisture_pred = self.moisture_head(cls).squeeze(-1)  # (B,)
        protein_pred = self.protein_head(cls).squeeze(-1)  # (B,)
        return food_logits, moisture_pred, protein_pred


class RapeseedCSSEModel(nn.Module):
    """CSSE encoder with four heads: tissue-type classification, two
    independent continuous regression targets (N_content, C_content), and
    N-regime (N-/N+) classification -- the latter used as the "quality"
    task instead of a guessed %DM threshold, since N_content has no single
    cutoff valid across tissue types or growth stages, but the dataset
    already carries the real N-/N+ fertilization-treatment ground truth
    (see dataset/rapeseed_multitask_dataset.py). No detection head, as with
    Mango/Grainit: neither regression target has a presence/absence concept."""

    def __init__(self, n_food_classes: int, n_regime_classes: int = 2, c_out: int = 64,
                 n_layers: int = 6, n_heads: int = 6, h_hidden: int = 128, seq_len: int = 512,
                 use_kan: bool = True, dropout: float = 0.1):
        super().__init__()
        self.encoder = CrossScaleSpectralEncoder(c_out, n_layers, n_heads, h_hidden, seq_len, dropout)
        D = self.encoder.d_model

        if use_kan:
            self.food_head = nn.Sequential(nn.LayerNorm(D), KAN([D, 32, 16, n_food_classes]))
        else:
            self.food_head = nn.Sequential(
                nn.Linear(D, 32), nn.GELU(), nn.Linear(32, 16), nn.GELU(), nn.Linear(16, n_food_classes)
            )
        # Plain linear heads: both targets are log1p+z-score normalized by
        # the dataset, so negative values are valid in this space -- no Softplus.
        self.n_content_head = nn.Linear(D, 1)
        self.c_content_head = nn.Linear(D, 1)
        self.regime_head = nn.Linear(D, n_regime_classes)

    def forward(self, x: torch.Tensor):
        cls = self.encoder(x)
        food_logits = self.food_head(cls)
        n_content_pred = self.n_content_head(cls).squeeze(-1)  # (B,)
        c_content_pred = self.c_content_head(cls).squeeze(-1)  # (B,)
        regime_logits = self.regime_head(cls)
        return food_logits, n_content_pred, c_content_pred, regime_logits


class MultiTaskSMARTNIRModel(nn.Module):
    """Original 3-head MT-SMART-NIR design (ported verbatim from
    MultiTaskSMARTNIR in ../MT-SMART-NIR/model_multitask.py), kept for the
    Danang pesticide dataset: food classification + per-substance detection
    (multi-label) + per-substance concentration regression, restricted to
    an explicit substance subset (see dataset/pesticide_multitask_dataset.py)
    rather than all 19. Unlike Mango/Grainit/Rapeseed, the detection head is
    kept here -- pesticide presence/absence is a real concept on this
    dataset (Stage 1), not something to strip out.

    reg_head ends in Softplus (concentration is always >= 0) and is
    soft-gated by the (detached) detection logits during training -- the
    gradient for "how confident is this present" and "what's the
    concentration" stay separate; predict() applies a hard threshold mask
    instead of the soft gate for clean inference-time estimates.
    """

    def __init__(self, n_food_classes: int, n_pesticides: int, c_out: int = 64,
                 n_layers: int = 6, n_heads: int = 6, h_hidden: int = 128, seq_len: int = 512,
                 use_kan: bool = True, dropout: float = 0.1):
        super().__init__()
        self.n_pesticides = n_pesticides
        self.encoder = CrossScaleSpectralEncoder(c_out, n_layers, n_heads, h_hidden, seq_len, dropout)
        D = self.encoder.d_model

        if use_kan:
            self.food_head = nn.Sequential(nn.LayerNorm(D), KAN([D, 32, 16, n_food_classes]))
        else:
            self.food_head = nn.Sequential(
                nn.Linear(D, 32), nn.GELU(), nn.Linear(32, 16), nn.GELU(), nn.Linear(16, n_food_classes)
            )
        self.det_head = nn.Linear(D, n_pesticides)
        self.reg_head = nn.Sequential(nn.Linear(D, n_pesticides), nn.Softplus())

    def forward(self, x: torch.Tensor, return_raw_conc: bool = False):
        cls = self.encoder(x)
        food_logits = self.food_head(cls)
        pest_logits = self.det_head(cls)
        c_raw = self.reg_head(cls)
        # Soft gate -- detach so the regression gradient can't corrupt det_head.
        pest_conc = c_raw * torch.sigmoid(pest_logits.detach())
        if return_raw_conc:
            return food_logits, pest_logits, pest_conc, c_raw
        return food_logits, pest_logits, pest_conc

    def predict(self, x: torch.Tensor, detection_threshold: float = 0.5):
        """Inference with a hard binary mask instead of the soft gate:
        detected compounds get the full Softplus estimate, absent ones are
        exactly 0."""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            cls = self.encoder(x)
            food_logits = self.food_head(cls)
            pest_logits = self.det_head(cls)
            c_raw = self.reg_head(cls)
        food_pred = food_logits.argmax(-1)
        pest_detected = torch.sigmoid(pest_logits) >= detection_threshold
        pest_conc = c_raw * pest_detected.float()
        if was_training:
            self.train()
        return food_pred, pest_detected, pest_conc
