"""
LensNIR with Cross-Scale Spectral Encoder (MFSE) and task-aware scale
selection.

Architecture overview
---------------------
  NIR spectrum (L-dim; Rapeseed L=1152, Grainit L=348, Mango L=172)
      ↓
  S convolutional scale stems  [odd kernel k_s, stride s=4, padding (k_s-1)/2
                                → T = L/s tokens per scale, all centred on
                                band s·i, i.e. aligned across scales]
      ↓
  Shared positional + per-scale embeddings
      ↓
  N_L × MFSELayer (SHARED weights across all scales):
    ├── Within-scale MHSA   (T tokens per stream; cost ∝ number of streams)
    ├── Scale Interaction   (S×S learned mixing of stream summaries + broadcast)
    └── LocalGatedFFN       (depthwise-conv + SwiGLU gate, default)
          or DualMLP        (original, kept for ablation)
      ↓
  Per-stream summaries h_s (final LN per token → mean pool)
      ↓
  Task-specific attentive scale fusion, one per task t
  (classification, each regression target, quality head if any):
      ω_{t,s} ∝ z_{t,s}·exp(e_{t,s}),  e_{t,s} = <w_t, h_s> + b_{t,s},
      plus a learned null slot, c_t = Σ_s ω_{t,s} h_s + ω_{t,0} n_t
      ↓
   ├── MLP([D,32,16,n_classes]) (or LN + KAN) on c_cls        → class logits
   ├── Linear(D, 1) on c_{reg:j}, one per target     → z-scored targets
   └── MLP([D,32,16,n_aux]) on c_aux                 → quality logits
                                                       (Rapeseed only)

Scale selection (scale_select = "task" | "shared"; "none" = all gates 1)
-----------------------------------------------------------------------
  z_{t,s} ∈ [0,1] are hard-concrete gates (Louizos et al., 2018). A stream is
  computed iff at least one task reads it, so the encoder cost is
  proportional to the size of the UNION of the task scale sets. The expected
  union size is
        C = Σ_s [1 − Π_t (1 − π_{t,s})],   π_{t,s} = P(z_{t,s} ≠ 0),
  and is held at a budget k with a Lagrangian (see losses.BudgetConstraint).
  "shared" ties one gate per scale across tasks (plain group-L0 baseline);
  cost="sum" uses the per-task count (1/T)Σ π_{t,s} instead of the union
  (task-independent baseline that ignores shared computation).

FFN comparison (D=258, H=128)
------------------------------
  DualMLP       : 33 410 params/layer  — additive split-branch, no spatial mixing
  LocalGatedFFN : 49 728 params/layer  — SwiGLU gate + depthwise spectral mixing
  LocalGatedFFN (iso-param, expand_ratio=1/3) : 32 634 — matches DualMLP
"""

import math
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from kan import KAN
from smartnir import SMARTNIRBackbone


# ─── Constants ────────────────────────────────────────────────────────────────

STRIDE = 4                                     # token stride s; T = L / s
DEFAULT_KERNELS = (5, 9, 17, 33)               # fixed S = 4 configuration
CANDIDATE_KERNELS = (5, 7, 9, 13, 17, 25, 33, 49)   # S = 8 search space (~√2 ladder)


def check_kernels(kernels: Sequence[int], seq_len: int, stride: int = STRIDE):
    kernels = tuple(int(k) for k in kernels)
    if len(set(kernels)) != len(kernels):
        raise ValueError(f"duplicate kernel sizes: {kernels}")
    for k in kernels:
        if k % 2 == 0 or k < stride:
            raise ValueError(f"kernel {k}: must be odd and >= stride {stride} "
                             "(symmetric padding, no skipped bands)")
        if k > seq_len:
            raise ValueError(f"kernel {k} longer than the spectrum (L={seq_len})")
    if seq_len % stride:
        raise ValueError(f"seq_len {seq_len} not divisible by stride {stride}")
    return kernels


# ─── Feed-forward networks ────────────────────────────────────────────────────

class DualMLP(nn.Module):
    """
    Original dual-branch FFN from SMART-NIR (kept for ablation).

    Splits input into two halves, applies independent Linear→GELU→Linear
    to each, then concatenates. Cross-wired so m uses fc2/fc4, n uses fc1/fc3.
    """
    def __init__(self, d_model: int, h_hidden: int):
        super().__init__()
        h, d = h_hidden // 2, d_model // 2
        self.fc1 = nn.Linear(d, h)
        self.fc2 = nn.Linear(d, h)
        self.fc3 = nn.Linear(h, d)
        self.fc4 = nn.Linear(h, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m, n = x.chunk(2, dim=-1)
        return torch.cat([self.fc4(F.gelu(self.fc2(m))),
                          self.fc3(F.gelu(self.fc1(n)))], dim=-1)


class LocalGatedFFN(nn.Module):
    """
    Local Gated Feed-Forward Network — replaces DualMLP.

    Two improvements over DualMLP
    ──────────────────────────────
    1. SwiGLU-style multiplicative gate: the gate branch learns which
       spectral features to amplify rather than unconditionally activating
       everything with GELU. This is the FFN used in LLaMA / PaLM, and
       gives stronger non-linearity with the same parameter budget.

    2. Depthwise Conv1d on the value branch: mixes adjacent tokens
       (spectral positions) within each scale stream before gating.
       NIR absorption peaks span several adjacent wavelengths, so a
       kernel-3 conv adds physically motivated local smoothing/sharpening
       without extra parameters (depthwise = groups=E, only E×k weights).

    Architecture
    ────────────
       x: (B, T, D)
       [v, g] = Linear(D, 2E)(x)         fused up-projection
       v      = DWConv1d(v, k)           local spectral mixing  (E×k params)
       y      = v ⊙ SiLU(g)             selective gating
       out    = Linear(E, D)(y)          down-projection

    Parameter budget  (D = d_model, H = h_hidden)
    ───────────────────────────────────────────────
       DualMLP       : D × H                       (baseline = 1×)
       LocalGatedFFN : E(3D + k)  where E = H × expand_ratio
         expand_ratio=0.50  →  E=64  →  1.49× DualMLP  (default — best performance)
         expand_ratio=0.33  →  E=42  →  0.98× DualMLP  (iso-param ablation)

    Args
    ────
        d_model      : token feature dimension D
        h_hidden     : reference hidden dim H (same as DualMLP's h_hidden)
        kernel_size  : depthwise conv kernel (3 = ±1 neighbour, 5 = ±2)
        expand_ratio : E = max(H × ratio, 8); 0.5 for default, 1/3 for iso-param
    """

    def __init__(
        self,
        d_model:      int,
        h_hidden:     int,
        kernel_size:  int   = 3,
        expand_ratio: float = 0.5,
    ):
        super().__init__()
        E = max(int(h_hidden * expand_ratio), 8)
        self.gate_up = nn.Linear(d_model, 2 * E, bias=False)
        self.dw_conv = nn.Conv1d(
            E, E, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=E, bias=False,
        )
        self.down = nn.Linear(E, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        v, g = self.gate_up(x).chunk(2, dim=-1)              # each (B, T, E)
        v = self.dw_conv(v.transpose(1, 2)).transpose(1, 2)  # local spectral mix
        return self.down(v * F.silu(g))                       # (B, T, D)


def build_ffn(ffn_type: str, d_model: int, h_hidden: int) -> nn.Module:
    """Factory: instantiate FFN by name."""
    if ffn_type == "local_gated":
        return LocalGatedFFN(d_model, h_hidden, kernel_size=3, expand_ratio=0.5)
    if ffn_type == "local_gated_iso":
        return LocalGatedFFN(d_model, h_hidden, kernel_size=3, expand_ratio=1 / 3)
    if ffn_type == "dual_mlp":
        return DualMLP(d_model, h_hidden)
    raise ValueError(f"Unknown ffn_type '{ffn_type}'. "
                     "Choose: 'local_gated' | 'local_gated_iso' | 'dual_mlp'")



# ─── Hard-concrete gates (Louizos, Welling & Kingma, 2018) ───────────────────

class HardConcreteGates(nn.Module):
    """
    A [n_rows, n_scales] matrix of hard-concrete gates.

    Training : z = clamp(sigmoid((log u − log(1−u) + log α)/β)·(ζ−γ) + γ, 0, 1),
               u ~ U(0,1), one draw per mini-batch.
    Test     : ẑ = clamp(sigmoid(log α)·(ζ−γ) + γ, 0, 1).
    P(z ≠ 0) = sigmoid(log α − β·log(−γ/ζ))        (Louizos et al., Eq. 12)

    After `freeze(mask)` the given test-time mask is stored in the `z_fixed`
    buffer and used in every mode, so the frozen network is exactly the
    deployable pruned network. The mask is chosen by the model (see
    LensNIR.test_mask): under an expected-size constraint the gates
    need not polarise to 0/1, so the plain rule ẑ > 0 can disagree with the
    budget (e.g. all π ≈ 0.15 → every ẑ = 0 although E|A| = 3).
    """

    def __init__(self, n_rows: int, n_scales: int, init_prob: float = 0.832,
                 beta: float = 2 / 3, gamma: float = -0.1, zeta: float = 1.1):
        super().__init__()
        self.beta, self.gamma, self.zeta = beta, gamma, zeta
        self._shift = beta * math.log(-gamma / zeta)          # < 0
        la0 = math.log(init_prob / (1 - init_prob)) + self._shift
        self.log_alpha = nn.Parameter(torch.full((n_rows, n_scales), la0))
        self.register_buffer("z_fixed", torch.ones(n_rows, n_scales))
        self.register_buffer("frozen", torch.tensor(False))

    def prob_open(self) -> torch.Tensor:
        return torch.sigmoid(self.log_alpha - self._shift)

    def deterministic(self) -> torch.Tensor:
        z = torch.sigmoid(self.log_alpha) * (self.zeta - self.gamma) + self.gamma
        return z.clamp(0.0, 1.0)

    def forward(self) -> torch.Tensor:
        if bool(self.frozen):
            return self.z_fixed
        if not self.training:
            return self.deterministic()
        la = self.log_alpha.float()
        u = torch.rand_like(la).clamp(1e-6, 1 - 1e-6)
        s = torch.sigmoid((torch.log(u) - torch.log1p(-u) + la) / self.beta)
        return (s * (self.zeta - self.gamma) + self.gamma).clamp(0.0, 1.0)

    @torch.no_grad()
    def freeze(self, mask: torch.Tensor):
        self.z_fixed.copy_(mask.to(self.z_fixed))
        self.frozen.fill_(True)
        self.log_alpha.requires_grad_(False)


class ScaleTaskGates(nn.Module):
    """
    Gates of the scale selection, z_{t,k} = s_k · u_{t,k}:

      s_k     one hard-concrete gate per scale  — is stream k computed?
      u_{t,k} one hard-concrete gate per (task, scale) — does task t read it?

    structure = "hier"   : both levels (default for task-aware selection)
                "flat"   : u only, s ≡ 1 (the plain per-(task, scale) gates;
                           ablation — its union cost has vanishing gradients)
                "shared" : s only, u ≡ 1 (one gate per scale for all tasks)

    With independent gates the probability that stream k is computed is

        P_k = π(s_k) · [1 − Π_t (1 − π(u_{t,k}))],

    so E|A| = Σ_k P_k is exact for every structure. With "flat",
    ∂P_k/∂π(u_{t,k}) = Π_{t'≠t} (1 − π(u_{t',k})), which vanishes as soon as
    several tasks keep k open: a scale read by every task can then no longer
    be pruned whatever the multiplier. With "hier", ∂P_k/∂π(s_k) =
    1 − Π_t (1 − π(u_{t,k})) ≈ 1 for such a scale, so the budget acts on it
    directly.
    """

    def __init__(self, n_tasks: int, n_scales: int, structure: str = "hier",
                 init_prob: float = 0.832):
        super().__init__()
        if structure not in ("hier", "flat", "shared"):
            raise ValueError(f"unknown gate structure '{structure}'")
        self.structure, self.n_tasks, self.n_scales = structure, n_tasks, n_scales
        self.scale = (HardConcreteGates(1, n_scales, init_prob)
                      if structure in ("hier", "shared") else None)
        self.task = (HardConcreteGates(n_tasks, n_scales, init_prob)
                     if structure in ("hier", "flat") else None)

    def _parts(self):
        return [g for g in (self.scale, self.task) if g is not None]

    @property
    def frozen(self) -> torch.Tensor:
        return self._parts()[0].frozen

    @property
    def z_fixed(self) -> torch.Tensor:
        z = torch.ones(self.n_tasks, self.n_scales, device=self.frozen.device)
        for g in self._parts():
            z = z * g.z_fixed
        return z

    def scale_prob(self) -> torch.Tensor:
        """[S] π(s_k) (ones without a scale level)."""
        if self.scale is None:
            return torch.ones(self.n_scales, device=self.frozen.device)
        return self.scale.prob_open()[0]

    def task_prob(self) -> torch.Tensor:
        """[n_tasks, S] π(u_{t,k}) (ones without a task level)."""
        if self.task is None:
            return torch.ones(self.n_tasks, self.n_scales, device=self.frozen.device)
        return self.task.prob_open()

    def prob_open(self) -> torch.Tensor:
        """[n_tasks, S] P(z_{t,k} ≠ 0) = π(s_k) π(u_{t,k})."""
        return self.scale_prob().unsqueeze(0) * self.task_prob()

    def union_prob(self) -> torch.Tensor:
        """[S] P(stream k is computed) = π(s_k) [1 − Π_t (1 − π(u_{t,k}))]."""
        pu = self.task_prob().float()
        return self.scale_prob().float() * (1.0 - torch.prod(1.0 - pu, dim=0))

    def deterministic(self) -> torch.Tensor:
        z = torch.ones(self.n_tasks, self.n_scales, device=self.frozen.device)
        for g in self._parts():
            z = z * g.deterministic()
        return z

    def forward(self) -> torch.Tensor:
        """[n_tasks, S] gate values (samples in training, z_fixed once frozen)."""
        z = torch.ones(self.n_tasks, self.n_scales, device=self.frozen.device)
        for g in self._parts():
            z = z * g()
        return z

    @torch.no_grad()
    def freeze(self, mask: torch.Tensor):
        """mask: binary [n_tasks, S]; the scale level keeps the union of its columns."""
        mask = mask.float()
        if self.scale is not None:
            self.scale.freeze(mask.amax(dim=0, keepdim=True))
        if self.task is not None:
            self.task.freeze(mask)


# ─── Scale Interaction Module ─────────────────────────────────────────────────

class ScaleInteractionModule(nn.Module):
    """
    Cross-scale communication between the active streams.

    1. Mean-pool each stream → summary g_s ∈ R^D, LayerNorm
    2. ḡ_s = Σ_{j∈A} softmax_{j∈A}(M_{s,·})_j · g_j   (A = computed streams;
       restricting the softmax to A is identical to physically removing the
       other streams)
    3. Project and broadcast as a residual to every token of stream s

    M (S×S) is initialised to the identity.
    """

    def __init__(self, d_model: int, n_scales: int, mode: str = "summary"):
        super().__init__()
        if mode not in ("summary", "none"):
            raise ValueError(f"unknown SIM mode '{mode}'")
        self.mode = mode
        self.mix  = nn.Parameter(torch.eye(n_scales))
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.ln   = nn.LayerNorm(d_model)

    def forward(self, z: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        # z: [B, A, T, D]; active: LongTensor [A] of scale indices
        if self.mode == "none":                                     # ablation: no
            return z                                                # cross-scale exchange
        m = torch.softmax(self.mix[active][:, active], dim=-1)      # [A, A]
        g = self.ln(z.mean(dim=2))                                  # [B, A, D]
        mixed = self.proj(torch.einsum('sk,bkd->bsd', m.to(g.dtype), g))
        return z + mixed.unsqueeze(2)


# ─── Encoder layer ────────────────────────────────────────────────────────────

class MFSELayer(nn.Module):
    """
    One encoder layer — the same weights are applied to every active stream
    (streams are folded into the batch dimension).

      1. Within-scale MHSA (pre-LN, residual)
      2. Scale Interaction Module
      3. FFN (pre-LN, residual): LocalGatedFFN (default) or DualMLP
    """

    def __init__(self, d_model: int, n_heads: int, h_hidden: int, n_scales: int,
                 dropout: float = 0.1, ffn_type: str = "local_gated",
                 sim_mode: str = "summary"):
        super().__init__()
        self.attn    = nn.MultiheadAttention(d_model, n_heads,
                                             dropout=dropout, batch_first=True)
        self.ln_attn = nn.LayerNorm(d_model)
        self.si      = ScaleInteractionModule(d_model, n_scales, sim_mode)
        self.ffn     = build_ffn(ffn_type, d_model, h_hidden)
        self.ln_ffn  = nn.LayerNorm(d_model)
        self.drop    = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        B, A, T, D = z.shape
        f = z.reshape(B * A, T, D)
        f_n = self.ln_attn(f)
        f = f + self.drop(self.attn(f_n, f_n, f_n, need_weights=False)[0])
        z = self.si(f.reshape(B, A, T, D), active)
        f = z.reshape(B * A, T, D)
        f = f + self.drop(self.ffn(self.ln_ffn(f)))
        return f.reshape(B, A, T, D)


# ─── Cross-Scale Spectral Encoder ────────────────────────────────────────────

class MultiFocalSpectralEncoder(nn.Module):
    """
    Cross-Scale Spectral Encoder (MFSE).

    forward(x, active) computes ONLY the streams listed in `active` and
    returns their summaries [B, |A|, D]. Streams outside `active` are never
    evaluated, so the cost is proportional to |A|.

    Args:
        kernels  : odd kernel sizes, one stream each (S = len(kernels))
        c_out    : channels per convolutional stem
        d_model  : token width D (independent of S)
        n_layers, n_heads, h_hidden, dropout, ffn_type : as usual
        seq_len  : input length L (divisible by STRIDE)
    """

    def __init__(self, kernels: Sequence[int] = DEFAULT_KERNELS, c_out: int = 64,
                 d_model: int = 258, n_layers: int = 6, n_heads: int = 6,
                 h_hidden: int = 128, seq_len: int = 512, dropout: float = 0.1,
                 ffn_type: str = "local_gated", stem: str = "linear",
                 sim_mode: str = "summary", token_pool: int = 1, readout: str = "mean",
                 readout_rank: int = 8):
        super().__init__()
        self.kernels = check_kernels(kernels, seq_len)
        self.n_scales = S = len(self.kernels)
        # token_pool p: average p neighbouring tokens after the stem (same for
        # every scale, so the streams stay aligned); T = ceil(L / (4 p)).
        self.token_pool = int(token_pool)
        self.seq_len = seq_len
        self.n_tokens = -(-(seq_len // STRIDE) // self.token_pool)
        self.c_out, self.d_model = c_out, d_model
        self.n_layers, self.h_hidden, self.ffn_type = n_layers, h_hidden, ffn_type

        if stem not in ("linear", "deep"):
            raise ValueError(f"unknown stem '{stem}'")
        self.stem = stem
        # linear: one strided conv (a linear patch embedding).
        # deep  : conv(k, stride 1) → BN → GELU → conv(3, stride 4); token i is
        #         still centred on band 4i, so the scales stay aligned.
        self.conv_branches = nn.ModuleList([
            nn.Conv1d(1, c_out, kernel_size=k, stride=STRIDE, padding=(k - 1) // 2)
            if stem == "linear" else nn.Sequential(
                nn.Conv1d(1, c_out, kernel_size=k, stride=1, padding=(k - 1) // 2),
                nn.BatchNorm1d(c_out), nn.GELU(),
                nn.Conv1d(c_out, c_out, kernel_size=3, stride=STRIDE, padding=1))
            for k in self.kernels
        ])
        self.scale_proj  = nn.Linear(c_out, d_model)
        self.pos_embed   = nn.Parameter(torch.zeros(self.n_tokens, d_model))
        self.scale_embed = nn.Parameter(torch.zeros(S, d_model))
        nn.init.trunc_normal_(self.pos_embed,   std=0.02)
        nn.init.trunc_normal_(self.scale_embed, std=0.02)

        self.layers = nn.ModuleList([
            MFSELayer(d_model, n_heads, h_hidden, S, dropout, ffn_type, sim_mode)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Stream readout (tokens → one vector per stream), shared by all streams:
        #   mean : average of the tokens (position-blind)
        #   flat : position-specific linear readout, as a flatten/FC head or
        #          a PLS coefficient vector: token i → r dims, flatten T·r → D
        if readout not in ("mean", "flat"):
            raise ValueError(f"unknown readout '{readout}'")
        self.readout = readout
        if readout == "flat":
            self.ro_down = nn.Linear(d_model, readout_rank)
            self.ro_proj = nn.Sequential(nn.Linear(self.n_tokens * readout_rank, d_model),
                                         nn.GELU())

    def _readout(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, A, T, D] normalised tokens → [B, A, D]
        if self.readout == "mean":
            return z.mean(dim=2)
        B, A = z.shape[:2]
        flat = self.ro_down(z).reshape(B, A, -1)                   # [B, A, T·r]
        self.last_flat = flat                                       # for task-linear heads
        return z.mean(dim=2) + self.ro_proj(flat)

    def forward(self, x: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        x1 = x.unsqueeze(1)                                        # [B, 1, L]
        feats = [self.conv_branches[int(k)](x1) for k in active]  # each [B, C, T]
        f = torch.stack(feats, dim=1)                              # [B, A, C, T]
        if self.token_pool > 1:
            B_, A_, C_, T_ = f.shape
            f = F.avg_pool1d(f.reshape(B_ * A_, C_, T_), self.token_pool,
                             ceil_mode=True).reshape(B_, A_, C_, -1)
        f = f.transpose(2, 3)                                      # [B, A, T, C]
        z = (self.scale_proj(f) + self.pos_embed
             + self.scale_embed[active].unsqueeze(1))              # [B, A, T, D]
        for layer in self.layers:
            z = layer(z, active)
        return self._readout(self.norm(z))                         # [B, A, D]

    def macs(self, n_active: int) -> int:
        """Approximate multiply-accumulates of the encoder for n_active streams."""
        T, D, C, H = self.n_tokens, self.d_model, self.c_out, self.h_hidden
        k_mean = sum(self.kernels) / len(self.kernels)
        stem = (self.seq_len // STRIDE) * C * k_mean + T * C * D   # conv before pooling
        attn = 4 * T * D * D + 2 * T * T * D
        if self.ffn_type == "dual_mlp":
            ffn = T * D * H
        else:
            E = max(int(H * (0.5 if self.ffn_type == "local_gated" else 1 / 3)), 8)
            ffn = T * (3 * D * E + 3 * E)
        return int(n_active * (stem + self.n_layers * (attn + ffn)))


# ─── Task-specific attentive scale fusion ─────────────────────────────────────

class TaskScaleFusion(nn.Module):
    """
    One attentive fusion per task over the computed streams (selective-kernel
    style, Li et al. 2019), with the gate scaling each stream's share:

        a_{t,k}  = <w_t, h_k> + b_{t,k}
        ω̃_{t,k}  = softmax over {k : z_{t,k} > 0} of a_{t,·}
        c_t      = Σ_k z_{t,k} ω̃_{t,k} h_k

    z enters outside the normalisation, so a gate changes how much of its
    stream reaches the task (gradient to every open gate) and the mixture
    cannot route around the gates. With every gate of a task shut, c_t = 0
    (no NaN: the softmax is computed only over open streams). The fixed model
    (all z = 1) reduces to plain selective-kernel fusion.

    An earlier variant normalised z inside the softmax and added a learned
    "null" slot to keep gradients for a lone gate; in training the null slot
    absorbed the weight (the model preferred a constant input to a noisy
    gated one) and learning collapsed, so it was removed.
    """

    def __init__(self, n_tasks: int, n_scales: int, d_model: int, shared: bool = False):
        super().__init__()
        # shared (ablation): one set of fusion weights for every task
        self.n_tasks = n_tasks
        n_p = 1 if shared else n_tasks
        self.w = nn.Parameter(torch.zeros(n_p, d_model))
        self.b = nn.Parameter(torch.zeros(n_p, n_scales))
        nn.init.trunc_normal_(self.w, std=0.02)

    def forward(self, h: torch.Tensor, z: torch.Tensor, active: torch.Tensor):
        # h: [B, A, D]; z: [n_tasks, A] gate values of the active streams
        h = h.float()
        w_t = self.w.float().expand(self.n_tasks, -1)
        b_t = self.b[:, active].float().expand(self.n_tasks, -1)
        a = torch.einsum('bad,td->bta', h, w_t) + b_t
        open_ = (z > 0).unsqueeze(0).expand_as(a)                    # [B, n_t, A]
        amax = a.masked_fill(~open_, float('-inf')).amax(dim=-1, keepdim=True)
        amax = torch.where(torch.isfinite(amax), amax, torch.zeros_like(amax))
        ex = torch.exp((a - amax).masked_fill(~open_, float('-inf')))  # 0 if shut
        w_norm = ex / ex.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        w = w_norm * z.float().unsqueeze(0)                          # [B, n_t, A]
        c = torch.einsum('bta,bad->btd', w, h)
        return c, w                                                    # [B, n_t, D]


# ─── Shared multi-task interface ──────────────────────────────────────────────

class _MultiTaskModel(nn.Module):
    """
    Common forward()/predict() for every model variant in this file.

    Subclasses implement `_encode(x) → (B, D)` and define `food_head`,
    `reg_head` and `aux_head` (a second classification head, or None when the
    dataset has no quality label). The regression head predicts z-scored
    targets; the train-fold mean/std are stored as buffers (via
    set_target_stats) so they travel with the checkpoint and predict()
    returns targets in their original units.
    """

    def _init_target_stats(self, n_targets: int, seq_len: int):
        self.n_targets = n_targets
        self.register_buffer("reg_mean", torch.zeros(n_targets))
        self.register_buffer("reg_std",  torch.ones(n_targets))
        # Per-wavelength input standardisation (train-fold statistics, set by
        # set_input_stats). Identity by default; stored in the checkpoint so
        # every forward() — training, evaluation, export — applies the same
        # transform.
        self.register_buffer("x_mean", torch.zeros(seq_len))
        self.register_buffer("x_std",  torch.ones(seq_len))

    def set_input_stats(self, mean, std):
        self.x_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.x_std.copy_(torch.as_tensor(std, dtype=torch.float32))

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.x_mean) / self.x_std

    def set_target_stats(self, mean, std):
        self.reg_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.reg_std.copy_(torch.as_tensor(std, dtype=torch.float32))

    def denormalize(self, reg_z: torch.Tensor) -> torch.Tensor:
        return reg_z.float() * self.reg_std + self.reg_mean

    def forward(self, x: torch.Tensor):
        """
        Returns:
            food_logits : (B, n_classes)  CrossEntropy target
            reg_z       : (B, n_targets)  standardised regression output
            aux_logits  : (B, n_aux) or None  quality-head logits
        """
        h = self._encode(self._prep(x))
        aux_logits = self.aux_head(h) if self.aux_head is not None else None
        return self.food_head(h), self.reg_head(h), aux_logits

    def predict(self, x: torch.Tensor):
        """
        Returns:
            food_pred : (B,)           class index
            reg_pred  : (B, n_targets) regression targets in original units
            aux_pred  : (B,) or None   quality-head class index
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            food_logits, reg_z, aux_logits = self(x)
        if was_training:
            self.train()
        aux_pred = aux_logits.argmax(-1) if aux_logits is not None else None
        return food_logits.argmax(-1), self.denormalize(reg_z), aux_pred


# ─── Multi-Task Model ─────────────────────────────────────────────────────────


# ─── Multi-Task Model ─────────────────────────────────────────────────────────

class LensNIR(_MultiTaskModel):
    """
    Multi-task NIR model: MFSE + task-specific scale fusion + task heads.

    Tasks (one fusion each): "cls", "reg:<j>" for every regression target,
    and "aux" when the dataset has a quality head.

    Args:
        kernels      : odd stem kernels (S = len(kernels)); DEFAULT_KERNELS
                       for the fixed model, CANDIDATE_KERNELS for selection
        scale_select : "none"   — no gates, every task reads every stream
                       "task"   — one hard-concrete gate per (task, scale)
                       "shared" — one gate per scale, shared by all tasks
        ffn_type     : 'local_gated' (default) | 'local_gated_iso' | 'dual_mlp'
    """

    def __init__(
        self,
        n_food_classes: int   = 5,
        n_targets:      int   = 2,
        n_aux_classes:  int   = 0,
        c_out:          int   = 64,
        n_layers:       int   = 6,
        n_heads:        int   = 6,
        h_hidden:       int   = 128,
        seq_len:        int   = 512,
        use_kan:        bool  = True,
        dropout:        float = 0.1,
        ffn_type:       str   = "local_gated",
        kernels:        Sequence[int] = DEFAULT_KERNELS,
        d_model:        Optional[int] = None,
        scale_select:   str   = "none",
        gate_init_prob: float = 0.832,
        gate_struct:    str   = "hier",
        stem:           str   = "linear",
        sim_mode:       str   = "summary",
        token_pool:     int   = 1,
        readout:        str   = "mean",
        readout_rank:   int   = 8,
        task_linear:    str   = "none",
        cond_reg:       str   = "none",
        fusion:         str   = "task",
    ):
        super().__init__()
        self._init_target_stats(n_targets, seq_len)
        if d_model is None:
            # Kept from SMART-NIR (D = 4·c_out rounded to a multiple of n_heads
            # = 258); a width hyper-parameter, NOT tied to the number of scales.
            d_model = math.ceil(4 * c_out / n_heads) * n_heads
        self._init_kwargs = dict(
            n_food_classes=n_food_classes, n_targets=n_targets,
            n_aux_classes=n_aux_classes, c_out=c_out, n_layers=n_layers,
            n_heads=n_heads, h_hidden=h_hidden, seq_len=seq_len,
            use_kan=use_kan, dropout=dropout, ffn_type=ffn_type,
            kernels=tuple(kernels), d_model=d_model, scale_select=scale_select,
            gate_init_prob=gate_init_prob, gate_struct=gate_struct,
            stem=stem, sim_mode=sim_mode, token_pool=token_pool, readout=readout,
            readout_rank=readout_rank, task_linear=task_linear,
            cond_reg=cond_reg, fusion=fusion)

        self.encoder = MultiFocalSpectralEncoder(
            kernels, c_out, d_model, n_layers, n_heads, h_hidden, seq_len,
            dropout, ffn_type, stem=stem, sim_mode=sim_mode, token_pool=token_pool,
            readout=readout, readout_rank=readout_rank)
        # Task-specific linear readout (needs readout="flat"): each task fuses
        # the flattened low-rank token maps of its streams with its own fusion
        # weights and adds a linear term, one coefficient per (token, dim), to
        # its output — a per-task PLS-like coefficient vector over wavelength.
        # Label-conditioned property heads (a soft, end-to-end form of
        # class-specific / local calibration). For property j:
        #   y_j += Σ_m Σ_c p_m(c|x) · (w_{j,m,c}·c_j + b_{j,m,c})
        # over EVERY classification label m of the dataset (class, and the
        # learned quality label when there is one), using the model's own
        # predicted probabilities. Zero-init (starts as the plain head).
        #   "soft"   : gradients flow into the classifiers too
        #   "detach" : the probabilities are treated as constants
        if cond_reg not in ("none", "soft", "detach"):
            raise ValueError(f"unknown cond_reg '{cond_reg}'")
        self.cond_reg = cond_reg
        n_cond = n_food_classes + n_aux_classes
        if cond_reg != "none":
            self.cond_w = nn.Parameter(torch.zeros(n_targets, n_cond, d_model))
            self.cond_b = nn.Parameter(torch.zeros(n_targets, n_cond))

        # task_linear: "none" | "reg" (property outputs only) | "all"
        if task_linear is True:
            task_linear = "all"
        elif task_linear in (False, None):
            task_linear = "none"
        if task_linear not in ("none", "reg", "all"):
            raise ValueError(f"unknown task_linear '{task_linear}'")
        self._init_kwargs["task_linear"] = task_linear
        self.task_linear_mode = task_linear
        self.task_linear = None
        if task_linear != "none":
            if readout != "flat":
                raise ValueError("task_linear requires readout='flat'")
            F_dim = self.encoder.n_tokens * readout_rank
            outs = [n_food_classes] + [1] * n_targets + ([n_aux_classes] if n_aux_classes else [])
            # "reg": every output except the class logits (properties and the
            # learned quality label)
            self.task_linear = nn.ModuleDict({
                str(i): nn.Linear(F_dim, o) for i, o in enumerate(outs)
                if task_linear == "all" or i > 0})
            for m in self.task_linear.values():
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
        D, S = d_model, self.encoder.n_scales

        self.task_names = (["cls"] + [f"reg:{j}" for j in range(n_targets)]
                           + (["aux"] if n_aux_classes else []))
        n_tasks = len(self.task_names)

        if scale_select not in ("none", "task", "shared"):
            raise ValueError(f"unknown scale_select '{scale_select}'")
        self.scale_select = scale_select
        # task: z_{t,k} = s_k·u_{t,k} ("hier") or u_{t,k} alone ("flat",
        # ablation); shared: z_{t,k} = s_k for every task.
        struct = "shared" if scale_select == "shared" else gate_struct
        self.gates = (None if scale_select == "none" else
                      ScaleTaskGates(n_tasks, S, struct, init_prob=gate_init_prob))
        if fusion not in ("task", "shared"):
            raise ValueError(f"unknown fusion '{fusion}'")
        self.fusion = TaskScaleFusion(n_tasks, S, D, shared=(fusion == "shared"))
        # Dense warm-up: the trainer keeps gates off (all z = 1) until the
        # network has left its initial plateau, then switches them on.
        self.gates_on = True

        def cls_head(n_out):
            if use_kan:
                # LayerNorm keeps the fused vector inside the KAN grid range.
                return nn.Sequential(nn.LayerNorm(D), KAN([D, 32, 16, n_out]))
            return nn.Sequential(
                nn.Linear(D, 32), nn.GELU(),
                nn.Linear(32, 16), nn.GELU(),
                nn.Linear(16, n_out),
            )

        self.food_head = cls_head(n_food_classes)
        self.reg_heads = nn.ModuleList([nn.Linear(D, 1) for _ in range(n_targets)])
        self.aux_head  = cls_head(n_aux_classes) if n_aux_classes else None
        self.last_fusion_weights = None

    # ── gates ────────────────────────────────────────────────────────────────
    def gate_values(self) -> torch.Tensor:
        """
        [n_tasks, S] gate values for this forward pass: hard-concrete samples
        in training, the binary test mask in eval, the frozen mask after
        freeze_gates().
        """
        n_t, S = len(self.task_names), self.encoder.n_scales
        if self.gates is None or not self.gates_on:
            return torch.ones(n_t, S, device=self.fusion.b.device)
        if self.training and not bool(self.gates.frozen):
            return self.gates()
        if bool(self.gates.frozen):
            return self.gates.z_fixed
        return self.test_mask()

    @torch.no_grad()
    def test_mask(self) -> torch.Tensor:
        """
        Binary test-time mask that honours the budget (CoFi-style: keep the
        expected number of units, ranked by their keep probability):
          1. n = round(E|A|) = round(Σ_s P_s), P_s = P(stream s computed)
          2. keep the n scales with the largest P_s
          3. task t reads a kept scale s iff π(u_ts) ≥ 0.5; a task left with
             none reads its most probable kept scale
          4. a kept scale read by no task is given to the task with the
             largest π(u) on it, so exactly n streams are computed
        """
        p = self.gates.task_prob().float()                  # [n_tasks, S]
        u = self.gates.union_prob()                         # [S]
        n = int(min(max(round(float(u.sum())), 1), u.numel()))
        keep = torch.zeros_like(u, dtype=torch.bool)
        keep[torch.topk(u, n).indices] = True
        m = (p >= 0.5) & keep.unsqueeze(0)
        for t in range(p.shape[0]):
            if not m[t].any():
                m[t, torch.where(keep, p[t], torch.full_like(p[t], -1.0)).argmax()] = True
        for k in torch.where(keep & ~m.any(dim=0))[0]:
            m[p[:, k].argmax(), k] = True
        return m.float()

    def expected_cost(self, cost: str = "union") -> torch.Tensor:
        """
        Expected number of computed streams (∝ encoder FLOPs).
          union : Σ_s π(s_s) [1 − Π_t (1 − π(u_{t,s}))]   (exact E|A|)
          sum   : (1/n_tasks) Σ_{t,s} π(s_s) π(u_{t,s})  (ignores shared computation)
        """
        if self.gates is None:
            return torch.tensor(float(self.encoder.n_scales))
        if cost == "union":
            return self.gates.union_prob().sum()
        if cost == "sum":
            p = self.gates.prob_open().float()
            return p.sum() / p.shape[0]
        raise ValueError(f"unknown cost '{cost}'")

    def freeze_gates(self):
        if self.gates is not None and not bool(self.gates.frozen):
            self.gates_on = True
            self.gates.freeze(self.test_mask())

    @torch.no_grad()
    def selection_summary(self) -> dict:
        """Test-time gates, open probabilities, active union and encoder MACs."""
        was = self.training
        self.eval()
        z = self.gate_values().cpu()
        self.train(was)
        active = (z > 0).any(dim=0).nonzero().flatten().tolist()
        out = {
            "kernels": list(self.encoder.kernels),
            "tasks": self.task_names,
            "gates": z.tolist(),
            "active": active,
            "active_kernels": [self.encoder.kernels[i] for i in active],
            "n_active": len(active),
            "macs_M": self.encoder.macs(len(active)) / 1e6,
            "macs_full_M": self.encoder.macs(self.encoder.n_scales) / 1e6,
        }
        if self.gates is not None:
            out["gate_struct"] = self.gates.structure
            out["prob_open"] = self.gates.prob_open().cpu().tolist()
            out["scale_prob"] = self.gates.scale_prob().cpu().tolist()
            out["union_prob"] = self.gates.union_prob().cpu().tolist()
        return out

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor):
        z = self.gate_values()                                    # [n_t, S]
        open_any = (z > 0).any(dim=0)
        if not open_any.any():
            # Every gate sampled shut (rare): compute the most probable stream
            # so the encoder still runs; every task then receives c_t = 0.
            score = (self.gates.union_prob()
                     if self.gates is not None else torch.ones_like(z[0]))
            open_any = torch.zeros_like(open_any)
            open_any[score.argmax()] = True
        active = open_any.nonzero().flatten()
        xp = self._prep(x)
        h = self.encoder(xp, active)                              # [B, A, D]
        c, w = self.fusion(h, z[:, active], active)               # [B, n_t, D]
        self.last_fusion_weights = (active.detach(), w.detach())

        food_logits = self.food_head(c[:, 0])
        if self.task_linear is not None:
            Ft = torch.einsum('bta,baf->btf', w, self.encoder.last_flat.float())
            tl = {int(i): head(Ft[:, int(i)]) for i, head in self.task_linear.items()}
            if 0 in tl:
                food_logits = food_logits + tl[0]
        n_reg = len(self.reg_heads)
        # n_reg=0 (classification-only dataset, e.g. DanangLeafy6): torch.cat
        # on an empty list raises, so short-circuit to an empty (B, 0) tensor.
        if n_reg:
            reg_z = torch.cat([head(c[:, 1 + j]) for j, head in enumerate(self.reg_heads)],
                              dim=-1)
        else:
            reg_z = c.new_zeros(c.shape[0], 0)
        aux_logits = self.aux_head(c[:, 1 + n_reg]) if self.aux_head is not None else None
        if self.task_linear is not None:
            if n_reg:
                reg_z = reg_z + torch.cat([tl[1 + j] for j in range(n_reg)], dim=-1)
            if aux_logits is not None:
                aux_logits = aux_logits + tl[1 + n_reg]
        if self.cond_reg != "none" and n_reg:
            # n_reg=0: nothing to condition (reg_z is already an empty (B,0)
            # tensor) -- skip entirely, since einsum on a 0-sized dimension
            # also breaks opt_einsum's internal path-cost estimate (a
            # decimal.InvalidOperation 0/0 inside contract_path).
            probs = [food_logits.float().softmax(-1)]
            if aux_logits is not None:
                probs.append(aux_logits.float().softmax(-1))
            pr = torch.cat(probs, dim=-1)                          # [B, K]
            if self.cond_reg == "detach":
                pr = pr.detach()
            cr = c[:, 1:1 + n_reg].float()                         # [B, n_reg, D]
            reg_z = reg_z + (torch.einsum('bk,jkd,bjd->bj', pr, self.cond_w.float(), cr)
                             + pr @ self.cond_b.float().t())
        return food_logits, reg_z, aux_logits

    # ── physical pruning ─────────────────────────────────────────────────────
    @torch.no_grad()
    def export_pruned(self) -> "LensNIR":
        """
        Copy of this model that contains only the streams in the test-time
        union. Its outputs equal this model's eval outputs exactly (up to
        float round-off); see tests in scripts/check_scale_selection.py.
        """
        summ = self.selection_summary()
        keep = summ["active"]
        was = self.training
        self.eval()
        z = self.gate_values() if self.gates is None else (
            self.gates.z_fixed if bool(self.gates.frozen) else self.test_mask())
        self.train(was)
        kw = dict(self._init_kwargs, kernels=tuple(summ["active_kernels"]))
        new = LensNIR(**kw).to(self.reg_mean.device)
        src = self.state_dict()
        idx = torch.tensor(keep, device=self.reg_mean.device)
        out = {}
        for name, t in src.items():
            if name.startswith("encoder.conv_branches."):
                i = int(name.split(".")[2])
                if i in keep:
                    out[name.replace(f"conv_branches.{i}.",
                                     f"conv_branches.{keep.index(i)}.")] = t
            elif name == "encoder.scale_embed":
                out[name] = t[idx]
            elif name.endswith(".si.mix"):
                out[name] = t[idx][:, idx]
            elif name == "fusion.b":
                out[name] = t[:, idx]
            elif name.startswith("gates.") and not name.endswith(".frozen"):
                out[name] = t[:, idx]
            else:
                out[name] = t
        new.load_state_dict(out)
        if new.gates is not None:
            new.gates.freeze(z[:, idx])
        return new.eval()


class FlatSMARTNIRMultiTask(_MultiTaskModel):
    """
    Flat-encoder multi-task baseline (SMART-NIR-MT).

    Uses the original SMARTNIRBackbone (4 scales concatenated → single flat
    Transformer) with the same task heads as LensNIR.
    This is a direct architectural ablation: replacing MFSE with the flat
    encoder isolates the contribution of scale-separated attention and the
    Scale Interaction Module.

    Key difference vs LensNIR:
      - Flat: T = L/4 tokens, each concatenating all 4 scales along the
              channel axis (early fusion; scale identity is lost).
      - MFSE: S streams of T scale-pure tokens each, coupled only through the
              SIM. Attention cost is ~S times the flat encoder's, the price of
              keeping scales separable (and therefore selectable).
    """

    def __init__(
        self,
        n_food_classes: int = 5,
        n_targets: int = 2,
        n_aux_classes: int = 0,
        c_out: int = 64,
        n_layers: int = 6,
        n_heads: int = 6,
        h_hidden: int = 128,
        seq_len: int = 512,
        use_kan: bool = True,
        dropout: float = 0.1,
        ffn_type: str = "local_gated",   # accepted but ignored (uses DualMLP inside backbone)
    ):
        super().__init__()
        self._init_target_stats(n_targets, seq_len)
        self.backbone = SMARTNIRBackbone(
            c_out=c_out, n_layers=n_layers, n_heads=n_heads,
            h_hidden=h_hidden, seq_len=seq_len, dropout=dropout,
        )
        D = self.backbone.d_model
        def cls_head(n_out):
            if use_kan:
                return nn.Sequential(nn.LayerNorm(D), KAN([D, 32, 16, n_out]))
            return nn.Sequential(nn.LayerNorm(D), nn.Linear(D, n_out))

        self.food_head = cls_head(n_food_classes)
        self.reg_head  = nn.Linear(D, n_targets)
        self.aux_head  = cls_head(n_aux_classes) if n_aux_classes else None

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)[0]


# ─── External transformer baseline (Spectraformer, N×ST usage) ───────────────

class SpectraformerMultiTask(_MultiTaskModel):
    """
    Adaptation of Spectraformer (Chen, Zhou & Ren, 2024, "Spectraformer:
    deep learning model for grain spectral qualitative analysis based on
    transformer structure", RSC Advances 14(12):8053-8066,
    DOI: 10.1039/d3ra07708j) for comparison against LensNIR.

    The published model is SINGLE-TASK (grain-variety classification only)
    and its authors' source code was not released. Per the paper's
    architecture description, it is a CNN front-end -- four kernel-3 Conv1d
    layers followed by four kernel-1 Conv1d layers -- feeding a standard
    ("Vanilla") Transformer encoder that treats the spectrum as a token
    sequence, followed by fully-connected output layers. This class is our
    reimplementation from that description (exact hyperparameters, e.g.
    encoder depth/width and any downsampling, are not specified in the
    paper and are therefore our own choice, documented below), given an
    extra regression head (and optional quality head) alongside the
    original classification head.

    Usage note: intended to be trained as INDEPENDENT SINGLE-TASK
    instances via train.py --model_type spectraformer
    --st_task {food,reg,aux}, each run optimising only its own task's loss
    (see the `_st_task` branch in train_fold()) and combined post-hoc.
    All heads exist on every instance so the same architecture and
    train_fold() loop serve every run; only the loss actually
    backpropagated differs per run.

    Sequence length differs across datasets (Rapeseed L=1152 vs. Mango
    L=172); since the CNN front-end here does not downsample (kernel=3/1,
    stride=1, 'same' padding, as described), an AdaptiveAvgPool1d
    compresses the token sequence to a fixed length before the Transformer
    encoder, bounding self-attention cost (O(T^2)) consistently across
    datasets -- the original paper only evaluates a single
    fixed-resolution instrument and does not address this.
    """

    def __init__(
        self,
        n_food_classes: int = 5,
        n_targets:      int = 2,
        n_aux_classes:  int = 0,
        seq_len:        int = 512,   # accepted for interface parity; unused (pooling is adaptive)
        d_model:        int = 128,
        n_heads:        int = 8,
        n_layers:       int = 4,
        n_tokens:       int = 128,   # fixed token count after adaptive pooling
        dropout:        float = 0.1,
        use_kan:        bool = True,   # ignored — Spectraformer uses a plain FC head
        ffn_type:       str = "local_gated",  # ignored — accepted for interface parity
    ):
        super().__init__()
        self._init_target_stats(n_targets, seq_len)
        self.n_tokens = n_tokens

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

        def cls_head(n_out):
            return nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, n_out))

        self.food_head = cls_head(n_food_classes)
        self.reg_head  = nn.Linear(d_model, n_targets)
        self.aux_head  = cls_head(n_aux_classes) if n_aux_classes else None

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        h = x.unsqueeze(1)                    # (B, 1, L)
        h = self.cnn_stem(h)                  # (B, D, L)
        h = self.pool(h)                      # (B, D, n_tokens)
        h = h.transpose(1, 2) + self.pos_embed  # (B, n_tokens, D)
        h = self.transformer(h)               # (B, n_tokens, D)
        return h.mean(dim=1)                  # (B, D)  mean-pooled representation


# ─── External multi-task baseline (ResNet-1D backbone) ───────────────────────

class _ResBlock1D(nn.Module):
    """Basic 1D residual block: two Conv1d(k=3)+BN+ReLU with a skip connection."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.act   = nn.ReLU(inplace=True)
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


class ResNet1DMultiTask(_MultiTaskModel):
    """
    Multi-task adaptation of a ResNet-18-style 1D backbone for NIRS, in the
    spirit of Pan et al. 2025's ResNet + multi-task-learning architecture
    for near-infrared spectroscopy (Spectrochimica Acta Part A 333:125866,
    DOI: 10.1016/j.saa.2025.125866). Their model predicts several chemical
    properties from one shared ResNet backbone; the published paper does
    not release source code or specify exact layer/channel counts, so this
    is our own ResNet-1D design (stem + four residual stages, doubling
    channels each stage) following the same general shared-backbone-plus-
    multiple-outputs principle, given the same task heads as the
    other baselines here (classification, regression, optional quality).

    Same usage pattern as SpectraformerMultiTask: intended to be trained as
    independent single-task instances (one model, all heads present,
    only one task's loss backpropagated per run) and combined post-hoc.
    """

    def __init__(
        self,
        n_food_classes: int = 5,
        n_targets:      int = 2,
        n_aux_classes:  int = 0,
        seq_len:        int = 512,   # accepted for interface parity; unused (pooling is adaptive)
        base_channels:  int = 64,
        blocks_per_stage: int = 2,
        dropout:        float = 0.1,
        use_kan:        bool = True,   # ignored — plain FC head, as in the source architecture
        ffn_type:       str = "local_gated",  # ignored — accepted for interface parity
    ):
        super().__init__()
        self._init_target_stats(n_targets, seq_len)

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

        def cls_head(n_out):
            return nn.Sequential(nn.Linear(in_ch, 64), nn.ReLU(), nn.Linear(64, n_out))

        self.food_head = cls_head(n_food_classes)
        self.reg_head  = nn.Linear(in_ch, n_targets)
        self.aux_head  = cls_head(n_aux_classes) if n_aux_classes else None

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        h = x.unsqueeze(1)          # (B, 1, L)
        h = self.stem(h)
        h = self.stages(h)
        h = self.pool(h).flatten(1)  # (B, C)
        return self.dropout(h)


# ─── Shallow 1D-CNN baselines (single-task literature models) ────────────────
#
# The three classes below are our reimplementations of published spectral
# CNNs, each given the same output heads as the other baselines
# (classification, z-scored regression, optional quality). The source papers
# do not release code for these exact configurations; layer sizes below are
# our choice within each paper's described design and are fixed across
# datasets. They run either as independent single-task instances
# (--st_task food|reg|aux, combined post-hoc by combine_st.py) or multi-task.

class _HeadsMixin:
    def _make_heads(self, d, n_food, n_targets, n_aux):
        self.food_head = nn.Linear(d, n_food)
        self.reg_head  = nn.Linear(d, n_targets)
        self.aux_head  = nn.Linear(d, n_aux) if n_aux else None


class CNN1DMultiTask(_HeadsMixin, _MultiTaskModel):
    """
    Shallow 1D CNN after Acquarelli et al. (2017, Anal. Chim. Acta 954:22-31)
    and Cui & Fearn (2018, Chemometr. Intell. Lab. Syst. 182:9-20): ONE
    convolutional layer over the raw spectrum followed by fully connected
    layers. Here: Conv1d(1→16, k=7, 'same') + ELU, flatten, FC 64 → 32 (ELU,
    dropout), heads.
    """

    def __init__(self, n_food_classes=5, n_targets=2, n_aux_classes=0, seq_len=512,
                 dropout=0.1, n_filters=16, kernel=7, **_):
        super().__init__()
        self._init_target_stats(n_targets, seq_len)
        self.conv = nn.Sequential(nn.Conv1d(1, n_filters, kernel, padding=kernel // 2), nn.ELU())
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(n_filters * seq_len, 64), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ELU(),
        )
        self._make_heads(32, n_food_classes, n_targets, n_aux_classes)

    def _encode(self, x):
        return self.fc(self.conv(x.unsqueeze(1)))


class _InceptionBlock1D(nn.Module):
    """Four parallel branches: 1×1 | 1×1→3 | 1×1→5 | maxpool 3→1×1, concatenated."""

    def __init__(self, in_ch, branch_ch):
        super().__init__()
        b = branch_ch
        self.b1 = nn.Conv1d(in_ch, b, 1)
        self.b3 = nn.Sequential(nn.Conv1d(in_ch, b, 1), nn.ReLU(), nn.Conv1d(b, b, 3, padding=1))
        self.b5 = nn.Sequential(nn.Conv1d(in_ch, b, 1), nn.ReLU(), nn.Conv1d(b, b, 5, padding=2))
        self.bp = nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1), nn.Conv1d(in_ch, b, 1))

    def forward(self, x):
        return F.relu(torch.cat([self.b1(x), self.b3(x), self.b5(x), self.bp(x)], dim=1))


class DeepSpectraMultiTask(_HeadsMixin, _MultiTaskModel):
    """
    DeepSpectra after Zhang et al. (2019, Anal. Chim. Acta 1058:48-57): a
    strided convolutional stem followed by Inception-style modules with
    parallel kernels of different size, then fully connected layers. Here:
    Conv1d(1→8, k=7, stride 3) + ReLU, two Inception blocks (4×8 and 4×16
    channels), global-to-32 average pooling, flatten, FC 64 (dropout), heads.
    """

    def __init__(self, n_food_classes=5, n_targets=2, n_aux_classes=0, seq_len=512,
                 dropout=0.1, **_):
        super().__init__()
        self._init_target_stats(n_targets, seq_len)
        self.features = nn.Sequential(
            nn.Conv1d(1, 8, 7, stride=3, padding=3), nn.ReLU(),
            _InceptionBlock1D(8, 8),
            _InceptionBlock1D(32, 16),
            nn.AdaptiveAvgPool1d(32),
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Dropout(dropout),
                                nn.Linear(64 * 32, 64), nn.ReLU(), nn.Dropout(dropout))
        self._make_heads(64, n_food_classes, n_targets, n_aux_classes)

    def _encode(self, x):
        return self.fc(self.features(x.unsqueeze(1)))


class MultiOutputCNNMultiTask(_HeadsMixin, _MultiTaskModel):
    """
    Multi-output 1D CNN after Mishra & Passos (2022, Postharvest Biol.
    Technol.): one convolutional layer with a wide kernel and a small dense
    stack shared by several outputs. Here: Conv1d(1→8, k=21, 'same') + ELU,
    flatten, FC 36 → 18 → 12 (ELU), shared by all heads; trained jointly
    (--st_task all), i.e. as a multi-output model.
    """

    def __init__(self, n_food_classes=5, n_targets=2, n_aux_classes=0, seq_len=512,
                 dropout=0.1, **_):
        super().__init__()
        self._init_target_stats(n_targets, seq_len)
        self.conv = nn.Sequential(nn.Conv1d(1, 8, 21, padding=10), nn.ELU())
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(8 * seq_len, 36), nn.ELU(),
            nn.Linear(36, 18), nn.ELU(),
            nn.Linear(18, 12), nn.ELU(),
        )
        self._make_heads(12, n_food_classes, n_targets, n_aux_classes)

    def _encode(self, x):
        return self.fc(self.conv(x.unsqueeze(1)))


class AssadzadehMTCNN(_MultiTaskModel):
    """
    Multi-task CNN of Assadzadeh et al. (2020, J. Near Infrared Spectrosc.
    28:275-286, Table 1), which predicts grain type, protein and moisture
    from one spectrum. The spectrum is linearly resampled to the original
    input size (599 points), then
      shared : conv1 8x10/5 + BN + ReLU, maxpool 3/2,
               conv2 32x4/2 + BN + ReLU, maxpool 3/2         (599 -> 13 x 32)
      branch : one per output (class, each property, quality label):
               conv 64x5/2 + BN + ReLU (-> 5 x 64), flatten, linear
    (hard parameter sharing). The paper writes the pooling size as "13" and
    the branch kernel as 10 in the text; only pooling 3 and kernel 5 give the
    activation sizes of its Table 1 (118, 58, 28, 13, 5). Trained with our
    protocol (loss weighting, optimiser, schedule), not the paper's fixed
    weights and MATLAB training loop.
    """

    IN_LEN = 599

    def __init__(self, n_food_classes=5, n_targets=2, n_aux_classes=0, seq_len=512,
                 dropout=0.0, **_):
        super().__init__()
        self._init_target_stats(n_targets, seq_len)

        def cbr(i, o, k, s):
            return [nn.Conv1d(i, o, k, stride=s), nn.BatchNorm1d(o), nn.ReLU()]
        self.shared = nn.Sequential(*cbr(1, 8, 10, 5), nn.MaxPool1d(3, 2),
                                    *cbr(8, 32, 4, 2), nn.MaxPool1d(3, 2))

        def branch(n_out):
            return nn.Sequential(*cbr(32, 64, 5, 2), nn.Flatten(), nn.Linear(64 * 5, n_out))
        self.food_head = branch(n_food_classes)
        self.reg_branches = nn.ModuleList([branch(1) for _ in range(n_targets)])
        self.aux_head = branch(n_aux_classes) if n_aux_classes else None

    def forward(self, x):
        x = self._prep(x).unsqueeze(1)
        if x.shape[-1] != self.IN_LEN:
            x = F.interpolate(x, size=self.IN_LEN, mode="linear", align_corners=True)
        h = self.shared(x)
        reg_z = torch.cat([b(h) for b in self.reg_branches], dim=1)
        aux = self.aux_head(h) if self.aux_head is not None else None
        return self.food_head(h), reg_z, aux
