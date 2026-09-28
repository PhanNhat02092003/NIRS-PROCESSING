"""SMART-NIR + food one-hot backbone used for both Buoc 1 (presence/absence)
and Buoc 2 (An toan/Vuot nguong) of the pesticide-safety pipeline.

Extracted for serving from stage1_detection.py (train branch), which trains
this same class for both steps -- kept as a standalone module here so the
deploy branch doesn't need the training-only code (XGBoost/LightGBM loops,
CLI dispatch) that file also contains.
"""

import torch
import torch.nn as nn

from model.classification_model import KANClassifier, SMARTNIRClassifier, SmartNIRClassificationConfig


class SmartNIRWithFood(nn.Module):
    """SMART-NIR backbone; the food one-hot is concatenated to the CLS
    embedding before the KAN head (the stock model only sees the spectrum)."""

    def __init__(self, signal_len: int, n_food: int, d_model=128, depth=3, n_heads=4):
        super().__init__()
        cfg = SmartNIRClassificationConfig(
            signal_len=signal_len, out_ch_per_branch=64, d_model=d_model,
            depth=depth, n_heads=n_heads, classifier="kan", num_classes=2,
        )
        base = SMARTNIRClassifier(cfg)
        self.mk, self.proj, self.encoder = base.mk, base.proj, base.encoder
        self.head = KANClassifier(d_model + n_food, 2, n_basis=cfg.kan_basis)

    def forward(self, spec, food):
        z = self.encoder(self.proj(self.mk(spec.unsqueeze(1))))
        return self.head(torch.cat([z[:, 0, :], food.to(z.dtype)], dim=1))


@torch.no_grad()
def predict_logit_diff(model, Xs, Fs, idx, bs):
    """Returns logit(class1) - logit(class0) for rows `idx`, float32 on CPU."""
    model.eval()
    out = []
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        with torch.autocast("cuda" if Xs.is_cuda else "cpu", dtype=torch.bfloat16):
            lg = model(Xs[b], Fs[b])
        lg = lg.float()
        out.append((lg[:, 1] - lg[:, 0]).cpu())
    return torch.cat(out).numpy()


def food_prior_shrinkage(y_prob, food_idx, prior_by_food):
    """Blend a calibrated model probability with an empirical per-food prior
    P(Vuot nguong | food) (Buoc 2 only). Same as stage2_safety.py: alpha = 1 -
    4*p*(1-p) trusts the food's own historical rate when it is near-certain,
    and trusts the spectral model where the food is genuinely mixed."""
    p_food = prior_by_food[food_idx]
    alpha = 1.0 - 4.0 * p_food * (1.0 - p_food)
    alpha = alpha.clip(0.0, 1.0)
    return alpha * p_food + (1.0 - alpha) * y_prob
