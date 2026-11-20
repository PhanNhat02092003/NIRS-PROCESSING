"""
Multi-task loss functions for SMART-NIR multi-task learning.

Loss composition:
  L_total = w1*L_class + w2*L_regression [+ w3*L_quality]

L_quality is a second CrossEntropy, present only for datasets with a real
quality label (Rapeseed N-/N+); rule-derived quality has no loss of its own.

Two weighting strategies:
  - 'fixed'       : user-specified weights (λ1, λ2, λ3)
  - 'uncertainty' : learnable weights via Kendall et al. 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Component losses ────────────────────────────────────────────────────────

class FoodClassificationLoss(nn.Module):
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, labels)


class MaskedRegressionLoss(nn.Module):
    """
    Huber loss over observed regression targets only.

    Targets are z-scored per column (train-fold statistics), so a single
    Huber delta is meaningful across targets with different units. Missing
    targets are NaN and contribute nothing to the loss or its gradient.

    Args:
        delta: Huber loss transition point, in z-score units
    """
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(
        self,
        pred: torch.Tensor,    # (B, T) — standardised predictions
        target: torch.Tensor,  # (B, T) — standardised targets, NaN where missing
    ) -> torch.Tensor:
        mask = ~torch.isnan(target)
        if not mask.any():
            return pred.sum() * 0.0  # differentiable zero
        return F.huber_loss(pred.float()[mask], target[mask],
                            delta=self.delta, reduction="mean")


# ─── Uncertainty weighting (Kendall et al. 2018) ─────────────────────────────

class UncertaintyWeightedLoss(nn.Module):
    """
    Automatically balances N tasks via learnable log-variance (Kendall et al. 2018).

    Kendall Eq. 9:  L = Σ_i  1/(2σᵢ²) · Lᵢ  +  log σᵢ

    Parameterised as log_sigma = log(σ), so:
        L = Σ_i  (1/2) · exp(-2 · log_σᵢ) · Lᵢ  +  log_σᵢ

    Optimal: ∂L/∂log_σ = 0  →  log_σ* = (1/2)·log(Lᵢ/2), i.e. σ² ∝ Lᵢ.
    High-loss tasks get higher σ (lower weight); converges automatically.
    """
    def __init__(self, n_tasks: int = 2):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses: list) -> torch.Tensor:
        total = torch.zeros(1, device=self.log_sigma.device)
        for i, loss in enumerate(losses):
            # precision = 1/σ² = exp(-2·log_σ)
            precision = torch.exp(-2.0 * self.log_sigma[i])
            total = total + 0.5 * precision * loss + self.log_sigma[i]
        return total.squeeze()

    @property
    def task_weights(self):
        """Returns 1/σ² for each task (higher = more weight)."""
        return torch.exp(-2.0 * self.log_sigma).detach()


# ─── Composite multi-task loss ────────────────────────────────────────────────

class MultiTaskLoss(nn.Module):
    """
    Combines classification, masked regression and (optional) quality losses.

    Args:
        weighting:          'uncertainty' | 'fixed'
        use_aux:            add the quality-head CrossEntropy as a third task
        lambda_food:        fixed weight for classification loss (weighting='fixed')
        lambda_regression:  fixed weight for regression loss
        lambda_aux:         fixed weight for quality loss
        regression_delta:   Huber loss delta (z-score units)
    """

    def __init__(
        self,
        weighting: str = "uncertainty",
        use_aux: bool = False,
        lambda_food: float = 1.0,
        lambda_regression: float = 1.0,
        lambda_aux: float = 1.0,
        regression_delta: float = 1.0,
    ):
        super().__init__()
        assert weighting in ("uncertainty", "fixed")
        self.weighting = weighting
        self.use_aux = use_aux

        self.food_loss = FoodClassificationLoss()
        self.reg_loss = MaskedRegressionLoss(regression_delta)

        lambdas = [lambda_food, lambda_regression] + ([lambda_aux] if use_aux else [])
        if weighting == "uncertainty":
            self.uncertainty = UncertaintyWeightedLoss(n_tasks=len(lambdas))
        else:
            self.register_buffer("lambdas", torch.tensor(lambdas))

    def forward(
        self,
        food_logits: torch.Tensor,  # (B, n_classes)
        reg_pred: torch.Tensor,     # (B, T) standardised
        food_labels: torch.Tensor,  # (B,) int
        reg_target: torch.Tensor,   # (B, T) standardised, NaN where missing
        aux_logits: torch.Tensor = None,  # (B, n_aux) — required when use_aux
        aux_labels: torch.Tensor = None,  # (B,) int
    ):
        L1 = self.food_loss(food_logits, food_labels)
        L2 = self.reg_loss(reg_pred, reg_target)
        losses = [L1, L2]
        if self.use_aux:
            L3 = F.cross_entropy(aux_logits, aux_labels)
            losses.append(L3)
        else:
            L3 = torch.zeros((), device=L1.device)

        if self.weighting == "uncertainty":
            total = self.uncertainty(losses)
        else:
            total = sum(w * L for w, L in zip(self.lambdas, losses))

        return total, L1, L2, L3

    @property
    def task_weights(self):
        if self.weighting == "uncertainty":
            return self.uncertainty.task_weights
        return self.lambdas.detach()


# ─── Compute-budget constraint for scale selection ───────────────────────────

class BudgetConstraint(nn.Module):
    """
    Lagrangian that holds the expected number of computed scale streams C at
    a target k (Wang et al., 2020; Xia et al., 2022):

        L_budget = λ1 (C − k) + λ2 (C − k)²

    The model parameters (including the gate log α) MINIMISE this term while
    λ1, λ2 are MAXIMISED, so they must sit in their own optimiser created
    with `maximize=True` (see train.py). The target is annealed
    linearly from the initial expected cost to k over the first part of
    training.
    """

    def __init__(self, n_scales: float, budget: float, warmup_steps: int):
        """n_scales: starting target (the expected cost at initialisation)."""
        super().__init__()
        self.n_scales, self.budget = float(n_scales), float(budget)
        self.warmup_steps = max(1, int(warmup_steps))
        self.lambdas = nn.Parameter(torch.zeros(2))

    def target(self, step: int) -> float:
        t = min(1.0, step / self.warmup_steps)
        return self.n_scales + t * (self.budget - self.n_scales)

    def forward(self, cost: torch.Tensor, step: int) -> torch.Tensor:
        gap = cost.float() - self.target(step)
        return self.lambdas[0] * gap + self.lambdas[1] * gap * gap
