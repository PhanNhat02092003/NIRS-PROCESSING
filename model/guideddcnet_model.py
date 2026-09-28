"""GuidedDCNet: a diffusion-integrated classification baseline, reimplemented
from the paper (baseline-information/classification/GuidedDCNet/cas-sc-template.tex)
since the authors' own reference repo has no code published yet.

Two deliberate deviations from the paper, agreed on before implementing:
  - The Local stream's "SHAP" region selection uses a gradient-based saliency
    proxy (`saliency_mask`, one backward pass) instead of literal
    KernelSHAP/DeepSHAP, which would need thousands of forward passes per
    sample per batch and make K-Fold training infeasible.
  - The diffusion process uses T=100 steps instead of a larger value like the
    1000 typical of image DDPMs -- the label vector here is only
    `num_classes`-dim, and T=100 keeps the per-epoch reverse-sampling
    validation pass fast across 5-fold x 2-machine training.
"""
import math
from dataclasses import dataclass, field
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Shared 1D ResNet backbone (Data / Global / Local encoders each get their
# own instance -- separate weights, same architecture, per the paper's
# "distinct" encoder roles).
# --------------------------------------------------------------------------
class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + identity)


class ResNet1DBackbone(nn.Module):
    def __init__(self, in_ch: int = 1, stem_ch: int = 32,
                 stage_channels: Tuple[int, ...] = (32, 64, 128, 256)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, stem_ch, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(stem_ch),
            nn.GELU(),
        )
        stages = []
        prev_ch = stem_ch
        for ch in stage_channels:
            stages.append(ResidualBlock1D(prev_ch, ch, stride=2))
            prev_ch = ch
        self.stages = nn.ModuleList(stages)
        self.out_channels = prev_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, L) -> (B, out_channels, L')
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x


class DataEncoder(nn.Module):
    """pi(x): global-average-pooled ResNet1D feature, used purely as a
    conditioning embedding for the Conditional UNet (not a classifier)."""

    def __init__(self, stage_channels: Tuple[int, ...] = (32, 64, 128, 256)):
        super().__init__()
        self.backbone = ResNet1DBackbone(stage_channels=stage_channels)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.embed_dim = stage_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        feat = self.backbone(x)
        return self.pool(feat).squeeze(-1)


# --------------------------------------------------------------------------
# Multi-scale Conditional Guidance Mechanism (MCGM)
# --------------------------------------------------------------------------
class GlobalStream(nn.Module):
    """f_g -> 1x1 conv class-activation map -> average response = y_g_hat."""

    def __init__(self, num_classes: int, stage_channels: Tuple[int, ...] = (32, 64, 128, 256)):
        super().__init__()
        self.backbone = ResNet1DBackbone(stage_channels=stage_channels)
        self.cam_conv = nn.Conv1d(stage_channels[-1], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        feat = self.backbone(x)          # (B, C, L')
        cam = self.cam_conv(feat)        # (B, num_classes, L')
        y_g_hat = cam.mean(dim=-1)       # (B, num_classes)
        return y_g_hat, cam


def saliency_mask(global_stream: GlobalStream, x: torch.Tensor, top_frac: float = 0.3) -> torch.Tensor:
    """Fast SHAP-approximation for the Local stream's region selection:
    |d(sum(y_g_hat))/dx| via a single backward pass, keeping the top
    `top_frac` wavelength positions per sample (zero elsewhere). This is NOT
    literal SHAP -- see module docstring for why.
    """
    was_training = global_stream.training
    global_stream.eval()  # avoid a second BN running-stat update from this side pass
    x_flat = x if x.dim() == 2 else x.squeeze(1)
    with torch.enable_grad():  # needed even when called from inside torch.no_grad() (reverse_sample)
        x_req = x_flat.detach().clone().requires_grad_(True)
        y_g_hat, _ = global_stream(x_req)
        score = y_g_hat.sum()
        grad = torch.autograd.grad(score, x_req)[0]
    if was_training:
        global_stream.train()

    importance = grad.abs()
    L = importance.shape[-1]
    k = max(1, int(L * top_frac))
    threshold = importance.topk(k, dim=-1).values[:, -1:]
    mask = (importance >= threshold).float()
    return (x_flat * mask).detach()


class GatedAttentionPool(nn.Module):
    """Ilse et al. gated-attention MIL pooling over a token sequence."""

    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.V = nn.Linear(dim, hidden)
        self.U = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, N, dim)
        a = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h)))  # (B, N, 1)
        a = torch.softmax(a, dim=1)
        return (a * h).sum(dim=1)


class LocalStream(nn.Module):
    def __init__(self, num_classes: int, top_frac: float = 0.3,
                 stage_channels: Tuple[int, ...] = (32, 64, 128, 256)):
        super().__init__()
        self.top_frac = top_frac
        self.backbone = ResNet1DBackbone(stage_channels=stage_channels)
        self.pool = GatedAttentionPool(stage_channels[-1])
        self.head = nn.Linear(stage_channels[-1], num_classes)

    def forward(self, x: torch.Tensor, global_stream: GlobalStream) -> torch.Tensor:
        x_masked = saliency_mask(global_stream, x, top_frac=self.top_frac)
        feat = self.backbone(x_masked.unsqueeze(1))   # (B, C, L')
        tokens = feat.transpose(1, 2)                  # (B, L', C)
        pooled = self.pool(tokens)
        return self.head(pooled)


class MCGM(nn.Module):
    def __init__(self, num_classes: int, top_frac: float = 0.3,
                 stage_channels: Tuple[int, ...] = (32, 64, 128, 256)):
        super().__init__()
        self.global_stream = GlobalStream(num_classes, stage_channels)
        self.local_stream = LocalStream(num_classes, top_frac, stage_channels)

    def forward(self, x: torch.Tensor):
        y_g_hat, _ = self.global_stream(x)
        y_l_hat = self.local_stream(x, self.global_stream)
        return y_g_hat, y_l_hat


# --------------------------------------------------------------------------
# Conditional UNet (denoiser over the num_classes-dim label vector)
# --------------------------------------------------------------------------
def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / half)
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ConditionalUNet(nn.Module):
    """MLP-based denoiser: reconciles the paper's abstract
    `Dec(Enc(Proj(Concat(y_t, y_g, y_l))), pi(x), t)` with its detailed
    Conditional-Linear-Block walkthrough (timestep re-injected via
    element-wise modulation at every block).
    """

    def __init__(self, num_classes: int, embed_dim: int, d_hidden: int = 128,
                 K: int = 4, time_dim: int = 64):
        super().__init__()
        self.num_classes = num_classes
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_hidden)
        )
        self.in_proj = nn.Linear(num_classes * 3, d_hidden)
        self.pi_proj = nn.Linear(embed_dim, d_hidden)
        self.blocks = nn.ModuleList([nn.Linear(d_hidden, d_hidden) for _ in range(K - 1)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(d_hidden) for _ in range(K - 1)])
        self.act = nn.Softplus()
        self.out = nn.Linear(d_hidden, num_classes)

    def forward(self, pi_x: torch.Tensor, y_t: torch.Tensor, y_g_hat: torch.Tensor,
                y_l_hat: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        e_t = self.time_mlp(sinusoidal_embedding(t, self.time_dim))
        inp = torch.cat([y_t, y_g_hat, y_l_hat], dim=-1)
        z = self.in_proj(inp) * e_t
        z = z * self.pi_proj(pi_x)
        for lin, bn in zip(self.blocks, self.bns):
            z = self.act(bn(lin(z) * e_t))
        return self.out(z)


def mmd_loss(a: torch.Tensor, b: torch.Tensor,
             sigmas: Tuple[float, ...] = (0.2, 0.5, 1.0, 2.0, 4.0)) -> torch.Tensor:
    """Biased MMD^2 between two equal-shape batches of vectors, RBF kernel
    mixture (Eq. 1 of the paper). Cheap here since vectors are num_classes-dim.
    """
    def kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x2 = (x ** 2).sum(dim=-1, keepdim=True)
        y2 = (y ** 2).sum(dim=-1, keepdim=True)
        dist = (x2 + y2.t() - 2 * x @ y.t()).clamp(min=0.0)
        total = sum(torch.exp(-dist / (2 * s * s)) for s in sigmas)
        return total / len(sigmas)

    kxx = kernel(a, a).mean()
    kyy = kernel(b, b).mean()
    kxy = kernel(a, b).mean()
    return kxx - 2 * kxy + kyy


# --------------------------------------------------------------------------
# Top-level model
# --------------------------------------------------------------------------
@dataclass
class GuidedDCNetConfig:
    num_classes: int = 9
    stage_channels: Tuple[int, ...] = field(default_factory=lambda: (32, 64, 128, 256))
    top_frac: float = 0.3
    T: int = 100
    beta1: float = 1e-4
    betaT: float = 0.02
    d_hidden: int = 128
    K: int = 4


class GuidedDCNet(nn.Module):
    def __init__(self, cfg: GuidedDCNetConfig):
        super().__init__()
        self.cfg = cfg
        self.num_classes = cfg.num_classes
        self.T = cfg.T

        self.data_encoder = DataEncoder(stage_channels=cfg.stage_channels)
        self.mcgm = MCGM(cfg.num_classes, top_frac=cfg.top_frac, stage_channels=cfg.stage_channels)
        self.unet = ConditionalUNet(cfg.num_classes, embed_dim=cfg.stage_channels[-1],
                                     d_hidden=cfg.d_hidden, K=cfg.K)

        betas = torch.linspace(cfg.beta1, cfg.betaT, cfg.T)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def encode(self, x: torch.Tensor):
        pi_x = self.data_encoder(x)
        y_g_hat, y_l_hat = self.mcgm(x)
        return pi_x, y_g_hat, y_l_hat

    def pretrain_logits(self, x: torch.Tensor, data_head: nn.Module):
        """Stage-1 CE pretraining logits: sum of the two guidance vectors plus
        a temporary head on the Data Encoder embedding (discarded after
        pretraining) -- the paper pretrains MCGM *and* Data Encoder jointly
        but only gives MCGM's outputs a class-shaped meaning, so the Data
        Encoder needs its own head to receive a classification gradient too.
        """
        pi_x, y_g_hat, y_l_hat = self.encode(x)
        return y_g_hat + y_l_hat + data_head(pi_x)

    def forward_noise(self, y0: torch.Tensor, guidance: torch.Tensor,
                       t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        ab = self.alpha_bars[t].unsqueeze(-1)
        sqrt_ab = ab.sqrt()
        sqrt_1mab = (1 - ab).sqrt()
        return sqrt_ab * y0 + sqrt_1mab * eps + (1 - sqrt_ab) * guidance

    def training_losses(self, x: torch.Tensor, y0: torch.Tensor):
        B = x.shape[0]
        device = x.device
        pi_x, y_g_hat, y_l_hat = self.encode(x)
        t = torch.randint(0, self.T, (B,), device=device)
        eps = torch.randn(B, self.num_classes, device=device)
        zeros = torch.zeros_like(y_g_hat)

        y_t_dual = self.forward_noise(y0, y_g_hat + y_l_hat, t, eps)
        eps_pred_dual = self.unet(pi_x, y_t_dual, y_g_hat, y_l_hat, t)
        loss_eps = F.mse_loss(eps_pred_dual, eps)

        y_t_g = self.forward_noise(y0, y_g_hat, t, eps)
        eps_pred_g = self.unet(pi_x, y_t_g, y_g_hat, zeros, t)

        y_t_l = self.forward_noise(y0, y_l_hat, t, eps)
        eps_pred_l = self.unet(pi_x, y_t_l, zeros, y_l_hat, t)

        loss_mmd_g = mmd_loss(eps, eps_pred_g)
        loss_mmd_l = mmd_loss(eps, eps_pred_l)
        return loss_eps, loss_mmd_g, loss_mmd_l

    @torch.no_grad()
    def reverse_sample(self, x: torch.Tensor) -> torch.Tensor:
        """Full T-step DDPM reverse loop starting from N((y_g+y_l)/2, I).
        Returns y0_hat (B, num_classes); argmax gives the predicted class.
        """
        B = x.shape[0]
        device = x.device
        pi_x, y_g_hat, y_l_hat = self.encode(x)
        mean_init = (y_g_hat + y_l_hat) / 2
        y_t = mean_init + torch.randn_like(mean_init)

        for t_idx in reversed(range(self.T)):
            t = torch.full((B,), t_idx, device=device, dtype=torch.long)
            eps_pred = self.unet(pi_x, y_t, y_g_hat, y_l_hat, t)
            alpha_t = self.alphas[t_idx]
            alpha_bar_t = self.alpha_bars[t_idx]
            beta_t = self.betas[t_idx]
            mean = (y_t - (beta_t / (1 - alpha_bar_t).sqrt()) * eps_pred) / alpha_t.sqrt()
            if t_idx > 0:
                y_t = mean + beta_t.sqrt() * torch.randn_like(y_t)
            else:
                y_t = mean
        return y_t
