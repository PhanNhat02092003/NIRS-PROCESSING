"""XSpecMamba / SpectralMamba, ported from the reference implementation
(git@github.com:phuongnguyen90/MambaNIR.git, mamba_architecture_v3.py) and
adapted for this project's data:

  - The reference repo (and its paper's "Data Preprocessing and Input
    Construction" section) builds GAF/RP/Corr at the raw spectrum's native
    length N, then bilinear-resizes each NxN matrix down to img_size --
    `spectrum_to_image` below matches this order (min-max scale to [-1,1] ->
    full-resolution GAF/RP/Corr -> `scipy.ndimage.zoom` bilinear downsample),
    not the earlier PAA-resize-then-transform shortcut this file used to
    take for compute/memory reasons. This is real O(N^2) work per sample
    (tens of ms at this project's wavelength counts), so callers should
    precompute it once per fold rather than lazily per __getitem__ -- see
    SpectralImageDataset in regression.py.
  - The reference repo's three channels pass through an 8-bit colormap +
    ToTensor + normalize(mean=std=0.5) pipeline, which end-to-end just maps
    each channel to a comparable centered range. `spectrum_to_image` reaches
    the same end state directly -- per-channel min-max rescale to [-1, 1] --
    without the colormap round-trip (which only loses precision for a model
    that consumes floats anyway) or the false assumption that our raw GAF/
    Corr float outputs already sit in [0, 1] the way an 8-bit image would.
  - The five architecture modules (A-E) are ported close to verbatim -- they
    are generic over input size/channels and were already well-designed for
    this kind of data. The only real change is `n_outputs`: the reference
    model predicts 2 targets (Nitrogen, Carbon) jointly from one backbone;
    here, matching SMART-NIR/NirMACNet/EBAR's per-substance-at-a-time Stage 2
    setup, n_outputs=1.
  - The backbone is the `mamba_ssm` package's own `Mamba` layer directly
    (not HuggingFace's `transformers.MambaModel` wrapper, which pulls in
    `torchvision` transitively through an unrelated import chain and fails
    at import time whenever the installed torch/torchvision pair is
    mismatched -- true in this project's environment, independent of
    whether the compiled `mamba_ssm`/`causal_conv1d` CUDA kernels
    themselves are installed and working). `mamba_ssm.Mamba` requires a CUDA
    tensor (its selective-scan kernel has no CPU path), so `backbone="auto"`
    trial-runs a forward pass on `cuda` and falls back to GRU on any
    failure -- no GPU, package not installed, or a genuine kernel error --
    rather than crashing at training time.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import zoom

try:
    from mamba_ssm import Mamba as MambaBlock
    _MAMBA_IMPORT_OK = True
except Exception:
    _MAMBA_IMPORT_OK = False


# ---------------------------------------------------------------------------
# 1D spectrum -> 3-channel spectral image (GAF / RP / Corr), adapted from
# data_augmentation.py -- direct float computation, no pyts/PNG round-trip.
# ---------------------------------------------------------------------------
def _rescale_minus1_1(x: np.ndarray) -> np.ndarray:
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-8:
        return np.zeros_like(x)
    return 2 * (x - mn) / (mx - mn) - 1


def compute_gaf(x: np.ndarray) -> np.ndarray:
    """Gramian Angular Summation Field."""
    xr = _rescale_minus1_1(x)
    phi = np.arccos(np.clip(xr, -1.0, 1.0))
    return np.cos(phi[:, None] + phi[None, :])


def compute_rp(x: np.ndarray, percentage: float = 20.0) -> np.ndarray:
    """Recurrence Plot, point threshold at the given percentile of pairwise
    distances (matches pyts' RecurrencePlot(threshold='point', percentage=20))."""
    d = np.abs(x[:, None] - x[None, :])
    thr = np.percentile(d, percentage)
    return (d <= thr).astype(np.float64)


def compute_corr(x: np.ndarray) -> np.ndarray:
    """Outer-product (Gram matrix) approximation of the correlation matrix,
    per the reference paper: Corr ~= G = X X^T, G_{i,j} = X_i * X_j. No
    rescale here -- the paper defines this for a spectrum already in
    [-1, 1]^T (`spectrum_to_image` guarantees that via `_rescale_minus1_1`
    before calling this), so G_{i,j} is already bounded in [-1, 1] by
    construction (a product of two [-1, 1] values). Rescaling per-sample
    here would stretch each sample's actual (often much narrower) range out
    to fill [-1, 1], distorting the true relative magnitude between
    samples that the paper's formula preserves as-is.
    """
    return np.outer(x, x)


def spectrum_to_image(x_1d: np.ndarray, img_size: int = 64) -> np.ndarray:
    """(L,) raw/normalized spectrum -> (3, img_size, img_size) float32 image:
    channel 0 = GAF, 1 = RP, 2 = Corr.

    Matches the reference paper's order: min-max scale the raw spectrum to
    [-1, 1] (its one and only stated per-spectrum normalization step), build
    each LxL representation at the spectrum's native length (not
    pre-downsampled), then bilinear-resize each to img_size x img_size --
    so the pairwise structure is computed from the full spectrum, and only
    the resulting image is compressed.

    No further *per-sample-adaptive* rescale is applied to GAF or Corr: both
    are already bounded in [-1, 1] by construction once the input spectrum
    is (GAF via cos(.), Corr as an outer product of [-1, 1] values -- see
    compute_gaf/compute_corr), and an adaptive min-max stretch here would
    inflate each sample to fill [-1, 1] regardless of its true magnitude,
    destroying exactly the inter-sample amplitude differences the paper's
    formulas leave intact. RP is natively {0, 1}-valued (a different native
    range, not a differently-scaled version of the same signal), so it gets
    a single *fixed* affine map `2x - 1` to land on the same [-1, 1] scale
    as the other two channels without depending on any per-sample statistic.
    """
    x = _rescale_minus1_1(np.asarray(x_1d, dtype=np.float64))
    n = len(x)
    zoom_factor = img_size / n

    def _resize(full: np.ndarray) -> np.ndarray:
        resized = full if n == img_size else zoom(full, zoom_factor, order=1)
        assert resized.shape == (img_size, img_size), (
            f"zoom produced {resized.shape}, expected ({img_size}, {img_size}) "
            f"for input length {n} -- rounding edge case, pad/crop needed"
        )
        return resized

    gaf = _resize(compute_gaf(x))
    rp = 2.0 * _resize(compute_rp(x)) - 1.0
    corr = _resize(compute_corr(x))
    img = np.stack([gaf, rp, corr], axis=0)
    return img.astype(np.float32)


# ---------------------------------------------------------------------------
# 2-D sinusoidal positional embedding (verbatim port)
# ---------------------------------------------------------------------------
def build_2d_sincos_pos_embed(h: int, w: int, dim: int, device) -> torch.Tensor:
    assert dim % 4 == 0, "pos-embed dim must be divisible by 4"
    half = dim // 4
    omega = 1.0 / (10000 ** (torch.arange(half, device=device, dtype=torch.float32) / half))
    y = torch.arange(h, device=device, dtype=torch.float32)
    x = torch.arange(w, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    out_y = torch.einsum("hw,d->hwd", yy, omega)
    out_x = torch.einsum("hw,d->hwd", xx, omega)
    pe = torch.cat([torch.sin(out_y), torch.cos(out_y), torch.sin(out_x), torch.cos(out_x)], dim=-1)
    return pe.view(h * w, dim)


# ---------------------------------------------------------------------------
# (A) NIR Gradient Enhancement -- verbatim port
# ---------------------------------------------------------------------------
class NIRGradientEnhancement(nn.Module):
    def __init__(self, in_chans: int = 3):
        super().__init__()
        self.in_chans = in_chans
        sx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3).repeat(in_chans, 1, 1, 1)
        sy = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3).repeat(in_chans, 1, 1, 1)
        self.register_buffer("sobel_x", sx)
        self.register_buffer("sobel_y", sy)
        self.fuse = nn.Sequential(
            nn.Conv2d(in_chans * 3, in_chans, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_chans),
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(in_chans, in_chans, bias=False), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(x, self.sobel_x, padding=1, groups=self.in_chans)
        gy = F.conv2d(x, self.sobel_y, padding=1, groups=self.in_chans)
        enhanced = self.fuse(torch.cat([x, gx, gy], dim=1))
        gate = self.se(x).view(x.size(0), self.in_chans, 1, 1)
        return x + gate * enhanced


# ---------------------------------------------------------------------------
# (B) Multi-Scale Spectral Patch Embedding -- verbatim port
# ---------------------------------------------------------------------------
class MultiScaleSpectralPatchEmbed(nn.Module):
    def __init__(self, in_chans: int = 3, emb_dim: int = 256, img_size: int = 64):
        super().__init__()
        self.fine_proj = nn.Conv2d(in_chans, emb_dim, kernel_size=4, stride=4, padding=0)
        self.fine_norm = nn.LayerNorm(emb_dim)
        self.coarse_proj = nn.Conv2d(in_chans, emb_dim, kernel_size=16, stride=16, padding=0)
        self.coarse_norm = nn.LayerNorm(emb_dim)
        self.scale_embed = nn.Embedding(2, emb_dim)
        nn.init.trunc_normal_(self.scale_embed.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        dev = x.device
        f = self.fine_proj(x)
        Hf, Wf = f.shape[-2:]
        f = self.fine_norm(f.flatten(2).transpose(1, 2))
        f = f + self.scale_embed(torch.zeros(1, dtype=torch.long, device=dev))
        c = self.coarse_proj(x)
        c = self.coarse_norm(c.flatten(2).transpose(1, 2))
        c = c + self.scale_embed(torch.ones(1, dtype=torch.long, device=dev))
        return f, c, Hf, Wf


# ---------------------------------------------------------------------------
# (C) Spectral Band Bias -- verbatim port
# ---------------------------------------------------------------------------
class SpectralBandBias(nn.Module):
    def __init__(self, n_tokens: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(n_tokens))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        scale = (1.0 + self.gate).unsqueeze(0).unsqueeze(-1)
        return tokens * scale


# ---------------------------------------------------------------------------
# (D) Cross-Directional Scanning Mamba -- ported with a more robust
# Mamba-vs-GRU backbone check (see module docstring).
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _mamba_backbone_usable(emb_dim: int, depth: int) -> bool:
    # Cached: this is a real construct+forward trial (see docstring above),
    # not a cheap check, and "auto" gets re-evaluated on every fold/substance
    # (potentially hundreds of times per run) with the same (emb_dim, depth).
    if not _MAMBA_IMPORT_OK or not torch.cuda.is_available():
        return False
    try:
        m = nn.Sequential(*[MambaBlock(d_model=emb_dim) for _ in range(depth)]).cuda()
        with torch.no_grad():
            m(torch.randn(1, 4, emb_dim, device="cuda"))
        return True
    except Exception:
        return False


class CrossScanMamba(nn.Module):
    def __init__(self, emb_dim: int = 256, depth: int = 2, n_dirs: int = 4, backbone: str = "auto"):
        super().__init__()
        assert n_dirs in (1, 2, 4), "n_dirs must be 1, 2, or 4"
        self.n_dirs = n_dirs

        if backbone == "mamba":
            use_mamba = _MAMBA_IMPORT_OK
        elif backbone == "gru":
            use_mamba = False
        else:  # auto
            use_mamba = _mamba_backbone_usable(emb_dim, depth)

        if use_mamba:
            self.backbone = nn.Sequential(*[MambaBlock(d_model=emb_dim) for _ in range(depth)])
            self._use_mamba = True
            print("[CrossScanMamba] Using mamba_ssm.Mamba backbone (CUDA selective-scan kernel)")
        else:
            self.backbone = nn.GRU(emb_dim, emb_dim, num_layers=depth, batch_first=True, bidirectional=False)
            self._use_mamba = False
            print("[CrossScanMamba] Using GRU backbone (cuDNN-accelerated)")

        self.scan_weights = nn.Parameter(torch.zeros(n_dirs))
        self.norm = nn.LayerNorm(emb_dim)

    @staticmethod
    def _row_to_col(t: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, L, C = t.shape
        return t.view(B, H, W, C).permute(0, 2, 1, 3).reshape(B, L, C)

    @staticmethod
    def _col_to_row(t: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, L, C = t.shape
        return t.view(B, W, H, C).permute(0, 2, 1, 3).reshape(B, L, C)

    def forward(self, tokens: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B = tokens.size(0)
        scans = [tokens]
        if self.n_dirs >= 2:
            scans.append(tokens.flip(1))
        if self.n_dirs == 4:
            col = self._row_to_col(tokens, H, W)
            scans.append(col)
            scans.append(col.flip(1))

        batched = torch.cat(scans, dim=0)
        if self._use_mamba:
            hs_all = self.backbone(batched)
        else:
            hs_all, _ = self.backbone(batched)

        hs_list = []
        for i in range(self.n_dirs):
            hs = hs_all[i * B: (i + 1) * B]
            if i == 1:
                hs = hs.flip(1)
            elif i == 2:
                hs = self._col_to_row(hs, H, W)
            elif i == 3:
                hs = self._col_to_row(hs.flip(1), H, W)
            hs_list.append(hs)

        w = torch.softmax(self.scan_weights, dim=0)
        fused = sum(w[i] * hs_list[i] for i in range(self.n_dirs))
        return self.norm(fused)


# ---------------------------------------------------------------------------
# (E) Target-Specific Attention Pooling -- verbatim port
# ---------------------------------------------------------------------------
class TargetSpecificAttentionPooling(nn.Module):
    def __init__(self, emb_dim: int = 256, n_outputs: int = 1, n_heads: int = 8):
        super().__init__()
        assert emb_dim % n_heads == 0, "emb_dim must be divisible by n_heads"
        self.queries = nn.Parameter(torch.empty(n_outputs, emb_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.attn = nn.MultiheadAttention(emb_dim, n_heads, batch_first=True, dropout=0.0)
        self.norm_q = nn.LayerNorm(emb_dim)
        self.norm_kv = nn.LayerNorm(emb_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B = tokens.size(0)
        q = self.norm_q(self.queries.unsqueeze(0).expand(B, -1, -1))
        kv = self.norm_kv(tokens)
        pooled, _ = self.attn(q, kv, kv)
        return pooled


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
@dataclass
class XSpecMambaConfig:
    img_size: int = 64
    in_chans: int = 3
    emb_dim: int = 128
    depth: int = 1
    n_outputs: int = 1
    n_dirs: int = 4  # matches the reference paper's validated default (N_d=4); see its
    # own ablation: "Moving from N_d=1... to N_d=4 consistently improves performance...
    # critical for capturing the symmetric structure of 2D spectral representations."
    n_heads_pool: int = 8
    # Default stays "gru", not "auto": the real backbone (mamba_ssm.Mamba)
    # needs its compiled CUDA kernels (causal_conv1d + mamba_ssm) installed,
    # which most environments running this file (e.g. Kaggle) won't have --
    # "auto"'s trial-and-fallback handles that fine either way, but a static
    # default of "gru" keeps behavior identical and reproducible across
    # environments without silently upgrading wherever the packages happen
    # to be present. Pass backbone="mamba" (or "auto") explicitly once
    # they're installed -- see baseline-information/regression/XSpecMamba
    # for the exact nvcc/CUDA_HOME setup this took to get working on an
    # sm_120 (Blackwell) GPU.
    backbone: str = "gru"


class XSpecMamba(nn.Module):
    def __init__(self, cfg: XSpecMambaConfig):
        super().__init__()
        assert cfg.emb_dim % 4 == 0, "emb_dim must be divisible by 4 (for pos-embed)"
        assert cfg.emb_dim % cfg.n_heads_pool == 0, "emb_dim must be divisible by n_heads_pool"
        self.cfg = cfg

        self.grad_enhance = NIRGradientEnhancement(cfg.in_chans)
        self.patch_embed = MultiScaleSpectralPatchEmbed(cfg.in_chans, cfg.emb_dim, cfg.img_size)

        n_fine = (cfg.img_size // 4) ** 2
        self.band_bias = SpectralBandBias(n_fine)
        self.cross_scan = CrossScanMamba(cfg.emb_dim, cfg.depth, cfg.n_dirs, cfg.backbone)
        self.pool = TargetSpecificAttentionPooling(cfg.emb_dim, cfg.n_outputs, cfg.n_heads_pool)

        mid = cfg.emb_dim // 4
        self.heads = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(cfg.emb_dim), nn.Linear(cfg.emb_dim, mid), nn.GELU(), nn.Linear(mid, 1))
            for _ in range(cfg.n_outputs)
        ])
        for head in self.heads:
            nn.init.trunc_normal_(head[1].weight, std=0.02)
            nn.init.zeros_(head[1].bias)
            nn.init.trunc_normal_(head[3].weight, std=0.02)
            nn.init.zeros_(head[3].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, in_chans, img_size, img_size] -> [B, n_outputs]"""
        x = self.grad_enhance(x)
        fine, coarse, Hf, Wf = self.patch_embed(x)
        pe = build_2d_sincos_pos_embed(Hf, Wf, fine.size(-1), fine.device)
        fine = fine + pe.unsqueeze(0)
        fine = self.band_bias(fine)
        fine_hs = self.cross_scan(fine, Hf, Wf)
        tokens = torch.cat([fine_hs, coarse], dim=1)
        pooled = self.pool(tokens)
        out = torch.cat([head(pooled[:, i, :]) for i, head in enumerate(self.heads)], dim=-1)
        return out
