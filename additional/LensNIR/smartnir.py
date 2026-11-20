import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from kan import KAN


class MultiKernelBlock(nn.Module):
    """
    4 parallel Conv2d branches with kernels 4×1, 8×1, 16×1, 32×1.
    Input: (B, 1, L, 1)  Output: (B, 4*C_out, H_out, 1)
    All branches yield H_out=128 when L=512, stride=(4,1).
    """
    KERNEL_CONFIGS = [
        (4,  (0, 0)),
        (8,  (3, 0)),
        (16, (7, 0)),
        (32, (15, 0)),
    ]

    def __init__(self, c_out: int = 64):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv2d(1, c_out, kernel_size=(k, 1), stride=(4, 1), padding=pad, bias=True)
            for k, pad in self.KERNEL_CONFIGS
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, L, 1)
        outs = [branch(x) for branch in self.branches]   # each (B, C_out, H_out, 1)
        return torch.cat(outs, dim=1)                     # (B, 4*C_out, H_out, 1)


class DualMLP(nn.Module):
    """
    Splits input along last dim into two halves m, n.
    Each half goes through Linear → GELU → Linear independently,
    then concatenated back.
    """
    def __init__(self, d_model: int, h_hidden: int = 128):
        super().__init__()
        half = d_model // 2
        h_half = h_hidden // 2
        self.fc1 = nn.Linear(half, h_half)
        self.fc2 = nn.Linear(half, h_half)
        self.fc3 = nn.Linear(h_half, half)
        self.fc4 = nn.Linear(h_half, half)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, seq, d_model)
        m, n = z.chunk(2, dim=-1)                       # each (B, seq, half)
        out_m = self.fc4(F.gelu(self.fc2(m)))
        out_n = self.fc3(F.gelu(self.fc1(n)))
        return torch.cat([out_m, out_n], dim=-1)        # (B, seq, d_model)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, h_hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = DualMLP(d_model, h_hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class SMARTNIRBackbone(nn.Module):
    """
    Shared feature extractor: Multi-kernel block + Position Embedding + Transformer Encoder.
    Returns (cls_token, seq_tokens, mean_pooled) for downstream task heads.
    """

    def __init__(
        self,
        c_out: int = 64,
        n_layers: int = 6,
        n_heads: int = 6,
        h_hidden: int = 128,
        seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        concat_dim = 4 * c_out

        d_model = math.ceil(concat_dim / n_heads) * n_heads
        self.d_model = d_model
        self.proj = nn.Linear(concat_dim, d_model) if d_model != concat_dim else nn.Identity()

        self.multi_kernel = MultiKernelBlock(c_out)
        h_out = (seq_len - 4) // 4 + 1

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, h_out + 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Sequential(*[
            TransformerEncoderLayer(d_model, n_heads, h_hidden, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        B = x.size(0)
        x = x.unsqueeze(1).unsqueeze(-1)          # (B, 1, L, 1)
        x = self.multi_kernel(x)                   # (B, 4*C_out, H_out, 1)
        x = x.squeeze(-1).permute(0, 2, 1)         # (B, H_out, 4*C_out)
        x = self.proj(x)                           # (B, H_out, d_model)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)             # (B, H_out+1, d_model)
        x = self.drop(x + self.pos_embed)
        x = self.encoder(x)
        x = self.norm(x)
        cls_out = x[:, 0]                          # (B, d_model)
        seq_out = x[:, 1:]                         # (B, H_out, d_model)
        mean_out = seq_out.mean(dim=1)             # (B, d_model)
        return cls_out, seq_out, mean_out


class SMART_NIR(nn.Module):
    """
    SMART-NIR: Multi-Kernel ViT with KAN classifier for NIR spectral classification.

    Args:
        n_classes:  number of output classes (default 4)
        c_out:      per-branch output channels (default 64); concat dim = 4*c_out
        n_layers:   number of Transformer encoder layers (default 6)
        n_heads:    number of attention heads (default 6)
        h_hidden:   Dual-MLP hidden size (default 128)
        seq_len:    padded input length (default 512)
        use_kan:    use KAN classifier head (True) or MLP (False)
        dropout:    dropout probability
    """

    def __init__(
        self,
        n_classes: int = 4,
        c_out: int = 64,
        n_layers: int = 6,
        n_heads: int = 6,
        h_hidden: int = 128,
        seq_len: int = 512,
        use_kan: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = SMARTNIRBackbone(c_out, n_layers, n_heads, h_hidden, seq_len, dropout)
        d_model = self.backbone.d_model

        if use_kan:
            self.classifier = KAN([d_model, 32, 16, n_classes])
        else:
            self.classifier = nn.Sequential(
                nn.Linear(d_model, 32), nn.GELU(),
                nn.Linear(32, 16),     nn.GELU(),
                nn.Linear(16, n_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_out, _, _ = self.backbone(x)
        return self.classifier(cls_out)
