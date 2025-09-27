# smart_nir.py
import math
from dataclasses import dataclass
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# 1) Multi-kernel 1D block
# ---------------------------
class MultiKernelBlock(nn.Module):
    """
    Input:  (B, 1, L)  with L=512
    Branches: conv1d kernels [4, 8, 16, 32], stride=4, paddings [0,3,7,15]
    Output: (B, 4*C_out, H_out) with H_out ~= 128 for L=512
    """
    def __init__(self, in_ch: int = 1, out_ch_per_branch: int = 64):
        super().__init__()
        C = out_ch_per_branch
        self.conv4  = nn.Conv1d(in_ch, C,  kernel_size=4,  stride=4, padding=0)
        self.conv8  = nn.Conv1d(in_ch, C,  kernel_size=8,  stride=4, padding=3)
        self.conv16 = nn.Conv1d(in_ch, C,  kernel_size=16, stride=4, padding=7)
        self.conv32 = nn.Conv1d(in_ch, C,  kernel_size=32, stride=4, padding=15)
        self.bn4, self.bn8, self.bn16, self.bn32 = nn.BatchNorm1d(C), nn.BatchNorm1d(C), nn.BatchNorm1d(C), nn.BatchNorm1d(C)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,1,L)
        o1 = self.act(self.bn4(self.conv4(x)))
        o2 = self.act(self.bn8(self.conv8(x)))
        o3 = self.act(self.bn16(self.conv16(x)))
        o4 = self.act(self.bn32(self.conv32(x)))
        return torch.cat([o1, o2, o3, o4], dim=1)  # (B, 4*C, H_out)


# ---------------------------
# 2) Positional embedding + projection
# ---------------------------
class PatchProjector(nn.Module):
    """
    Project channel depth to d_model, add CLS token and learnable positional bias.
    Input:  (B, C_tot, H_out)
    Output: (B, H_out+1, d_model)
    """
    def __init__(self, c_in: int, d_model: int, max_tokens: int):
        super().__init__()
        self.proj = nn.Linear(c_in, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens + 1, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, H)
        x = x.transpose(1, 2)            # (B, H, C_in)
        x = self.proj(x)                 # (B, H, d_model)
        B, H, D = x.shape
        cls = self.cls_token.expand(B, -1, -1)   # (B,1,D)
        x = torch.cat([cls, x], dim=1)           # (B, H+1, D)
        pos = self.pos_embed[:, :H+1, :]
        return x + pos


# ---------------------------
# 3) Transformer with Dual-MLP
# ---------------------------
class DualMLP(nn.Module):
    """
    Split last dim in half: (.., D/2 | D/2)
    Each branch: Linear -> GELU -> Linear, then concat.
    """
    def __init__(self, d_model: int, hidden_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for Dual-MLP."
        half = d_model // 2
        h = int(half * hidden_ratio)

        self.fc1_a = nn.Linear(half, h)
        self.fc2_a = nn.Linear(h, half)
        self.fc1_b = nn.Linear(half, h)
        self.fc2_b = nn.Linear(h, half)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = torch.split(x, x.shape[-1] // 2, dim=-1)
        a = self.drop(self.fc2_a(self.act(self.fc1_a(a))))
        b = self.drop(self.fc2_b(self.act(self.fc1_b(b))))
        return torch.cat([a, b], dim=-1)


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0, attn_drop: float = 0.0, drop: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=attn_drop, batch_first=True)
        self.drop_path1 = nn.Dropout(drop)

        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = DualMLP(d_model, hidden_ratio=mlp_ratio, drop=drop)
        self.drop_path2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN -> MHSA -> residual
        y = self.ln1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + self.drop_path1(y)

        # Pre-LN -> Dual-MLP -> residual
        y = self.mlp(self.ln2(x))
        x = x + self.drop_path2(y)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, depth: int, d_model: int, n_heads: int, mlp_ratio: float = 4.0, attn_drop: float = 0.0, drop: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            EncoderBlock(d_model, n_heads, mlp_ratio, attn_drop, drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)   # (B, N+1, D)


# ---------------------------
# 4) KAN (đơn giản, dựa trên cơ sở RBF)
#    y_j = sum_i sum_k W[j,i,k] * exp(-beta * (x_i - c_k)^2) + b_j
# ---------------------------
class KANLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_basis: int = 8, beta: float = 4.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_basis = n_basis
        self.beta = nn.Parameter(torch.tensor(float(beta)))
        # centers cố định đều trên [-1,1]
        centers = torch.linspace(-1.0, 1.0, n_basis).view(1, 1, n_basis)  # (1,1,K)
        self.register_buffer("centers", centers, persistent=False)
        # trọng số (out, in, K)
        self.weights = nn.Parameter(torch.zeros(out_dim, in_dim, n_basis))
        nn.init.xavier_uniform_(self.weights)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        # chuẩn hoá đầu vào từng chiều (affine learnable)
        self.in_scale = nn.Parameter(torch.ones(in_dim))
        self.in_shift = nn.Parameter(torch.zeros(in_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim)
        x = (x + self.in_shift) * self.in_scale
        x = x.unsqueeze(-1)  # (B, in_dim, 1)
        # RBF
        # broadcast x vs centers: (B, in, K)
        phi = torch.exp(-(self.beta) * (x - self.centers) ** 2)
        # (B, in, K) -> (B, out)
        # einsum: b i k , o i k -> b o
        y = torch.einsum('bik,oik->bo', phi, self.weights) + self.bias
        return y


class KANClassifier(nn.Module):
    """
    Hai lớp ẩn: 32 -> 16 (như mô tả), nhưng mỗi lớp là KANLayer.
    """
    def __init__(self, in_dim: int, num_classes: int, n_basis: int = 8):
        super().__init__()
        self.l1 = KANLayer(in_dim, 32, n_basis=n_basis)
        self.act1 = nn.GELU()
        self.l2 = KANLayer(32, 16, n_basis=n_basis)
        self.act2 = nn.GELU()
        self.out = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.l1(x))
        x = self.act2(self.l2(x))
        return self.out(x)


class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32), nn.GELU(),
            nn.Linear(32, 16), nn.GELU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------
# 5) SMART-NIR model
# ---------------------------
@dataclass
class SmartNIRClassificationConfig:
    signal_len: int = 512
    out_ch_per_branch: int = 64      
    d_model: int = 64                
    depth: int = 6                   
    n_heads: int = 8
    mlp_ratio: float = 4.0
    attn_drop: float = 0.0
    drop: float = 0.0
    classifier: Literal["mlp", "kan"] = "kan"
    num_classes: int = 2
    kan_basis: int = 8


class SMARTNIR(nn.Module):
    def __init__(self, cfg: SmartNIRClassificationConfig):
        super().__init__()
        self.cfg = cfg
        self.mk = MultiKernelBlock(in_ch=1, out_ch_per_branch=cfg.out_ch_per_branch)

        # Tính số token H_out theo conv1d (xấp xỉ 128 cho L=512)
        # Dùng forward để suy ra động trong build projector.
        dummy = torch.zeros(1, 1, cfg.signal_len)
        with torch.no_grad():
            h = self.mk(dummy).shape[-1]
        c_tot = cfg.out_ch_per_branch * 4

        self.proj = PatchProjector(c_in=c_tot, d_model=cfg.d_model, max_tokens=h)
        self.encoder = TransformerEncoder(
            depth=cfg.depth, d_model=cfg.d_model, n_heads=cfg.n_heads,
            mlp_ratio=cfg.mlp_ratio, attn_drop=cfg.attn_drop, drop=cfg.drop
        )

        # Head: lấy CLS token
        in_dim = cfg.d_model
        if cfg.classifier == "mlp":
            self.head = MLPClassifier(in_dim, cfg.num_classes)
        else:
            self.head = KANClassifier(in_dim, cfg.num_classes, n_basis=cfg.kan_basis)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        r: (B, L) hoặc (B, 1, L)
        """
        if r.dim() == 2:
            r = r.unsqueeze(1)  # (B,1,L)
        x0 = self.mk(r)                    # (B, C_tot, H_out)
        x1 = self.proj(x0)                 # (B, H_out+1, D)
        z  = self.encoder(x1)              # (B, H_out+1, D)
        cls = z[:, 0, :]                   # (B, D)
        logits = self.head(cls)            # (B, num_classes)
        return logits
