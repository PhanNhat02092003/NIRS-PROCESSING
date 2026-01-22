import math
from dataclasses import dataclass
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiKernelBlock(nn.Module):
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
        o1 = self.act(self.bn4(self.conv4(x)))
        o2 = self.act(self.bn8(self.conv8(x)))
        o3 = self.act(self.bn16(self.conv16(x)))
        o4 = self.act(self.bn32(self.conv32(x)))
        return torch.cat([o1, o2, o3, o4], dim=1)

class PatchProjector(nn.Module):
    def __init__(self, c_in: int, d_model: int, max_tokens: int):
        super().__init__()
        self.proj = nn.Linear(c_in, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens + 1, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)           
        x = self.proj(x)                 
        B, H, D = x.shape
        cls = self.cls_token.expand(B, -1, -1)   
        x = torch.cat([cls, x], dim=1)           
        pos = self.pos_embed[:, :H+1, :]
        return x + pos

class DualMLP(nn.Module):
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
        y = self.ln1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + self.drop_path1(y)

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
        return self.norm(x) 

class KANLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_basis: int = 8, beta: float = 4.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_basis = n_basis
        self.beta = nn.Parameter(torch.tensor(float(beta)))
        centers = torch.linspace(-1.0, 1.0, n_basis).view(1, 1, n_basis) 
        self.register_buffer("centers", centers, persistent=False)
        self.weights = nn.Parameter(torch.zeros(out_dim, in_dim, n_basis))
        nn.init.xavier_uniform_(self.weights)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.in_scale = nn.Parameter(torch.ones(in_dim))
        self.in_shift = nn.Parameter(torch.zeros(in_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x + self.in_shift) * self.in_scale
        x = x.unsqueeze(-1)
        phi = torch.exp(-(self.beta) * (x - self.centers) ** 2)
        y = torch.einsum('bik,oik->bo', phi, self.weights) + self.bias
        return y


class KANRegressor(nn.Module):
    def __init__(self, in_dim: int, num_targets: int, n_basis: int = 8):
        super().__init__()
        self.l1 = KANLayer(in_dim, 32, n_basis=n_basis)
        self.act1 = nn.GELU()
        self.l2 = KANLayer(32, 16, n_basis=n_basis)
        self.act2 = nn.GELU()
        self.out = nn.Linear(16, num_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.l1(x))
        x = self.act2(self.l2(x))
        return self.out(x)


class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, num_targets: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32), nn.GELU(),
            nn.Linear(32, 16), nn.GELU(),
            nn.Linear(16, num_targets)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

@dataclass
class SmartNIRRegressionConfig:
    signal_len: int = 512
    out_ch_per_branch: int = 64      
    d_model: int = 64                
    depth: int = 6                   
    n_heads: int = 8
    mlp_ratio: float = 4.0
    attn_drop: float = 0.0
    drop: float = 0.0
    classifier: Literal["mlp", "kan"] = "kan"
    num_targets: int = 2
    kan_basis: int = 8


class SMARTNIRRegressor(nn.Module):
    def __init__(self, cfg: SmartNIRRegressionConfig):
        super().__init__()
        self.cfg = cfg
        self.mk = MultiKernelBlock(in_ch=1, out_ch_per_branch=cfg.out_ch_per_branch)

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
            self.head = MLPRegressor(in_dim, cfg.num_targets)
        else:
            self.head = KANRegressor(in_dim, cfg.num_targets, n_basis=cfg.kan_basis)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        if r.dim() == 2:
            r = r.unsqueeze(1)
        x0 = self.mk(r)                 
        x1 = self.proj(x0)          
        z  = self.encoder(x1)           
        cls = z[:, 0, :]                   
        outs = self.head(cls)            
        return outs