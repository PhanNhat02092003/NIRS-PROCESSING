"""Runs MT-SMART-NIR's Cross-Scale Spectral Encoder (CSSE) on the Rapeseed
INRAE winter oilseed rape tissue NIR dataset (prepare_rapeseed_dataset.py:
roots/leaves/stems/flowers/pods, N_content, C_content), ported the same way
as mango_mt_smartnir_engine.py / grainit_mt_smartnir_engine.py. 4 tasks here
-- tissue-type classification + N_content regression + C_content regression
+ N-regime (N-/N+) classification -- since N_content has no single %DM
cutoff that is valid across tissue types (N_content averages ~4.6% in
Flowers vs ~1.2% in Pods in this very dataset) or growth stages (the
agronomic standard is a biomass-dependent critical-N-dilution curve, and
this dataset has no biomass column), so instead of guessing a threshold the
"quality" task uses the dataset's own N-/N+ fertilization-treatment ground
truth directly as a classification target (model/mt_smartnir_model.py:
RapeseedCSSEModel). C_content has no analogous regime or deficiency
threshold in the literature (confirmed against the dataset's own source
paper, PMC11663971) -- carbon fraction of dry matter is driven by tissue
biochemical composition (cellulose/lignin/lipid/protein), not nutrient
status, so it stays a plain regression target with no paired "quality" task.
Still no detection head: neither N_content nor C_content has a
presence/absence concept (see prepare_rapeseed_dataset.py's docstring).

A per-target outlier flag nulls some N_content/C_content rows to -1
independently (ported from the source INRAE file's own per-target
model-set flags); their loss contribution is masked out (see
RapeseedMultiTaskDataset) the same way as Grainit's missing Moisture/Protein
rows.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))  # repo root: model/, dataset/, shared engines
_ALL = os.path.join(_ROOT, "..", "all-dataset")
sys.path.insert(0, _ROOT)
os.chdir(_HERE)  # relative outputs (history/, checkpoint/, data/) land next to this script

import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, SubsetRandomSampler

from dataset.rapeseed_multitask_dataset import RapeseedMultiTaskDataset
from model.mt_smartnir_model import RapeseedCSSEModel


class UncertaintyWeightedLoss(nn.Module):
    """Learnable task weighting (Kendall et al. 2018), ported from
    ../MT-SMART-NIR/losses.py: L = sum_i 0.5*exp(-2*log_sigma_i)*L_i + log_sigma_i."""

    def __init__(self, n_tasks: int = 4):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses: list) -> torch.Tensor:
        total = torch.zeros(1, device=self.log_sigma.device)
        for i, loss in enumerate(losses):
            precision = torch.exp(-2.0 * self.log_sigma[i])
            total = total + 0.5 * precision * loss + self.log_sigma[i]
        return total.squeeze()

    @property
    def task_weights(self):
        return torch.exp(-2.0 * self.log_sigma).detach()


def masked_huber(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, delta: float = 0.5) -> torch.Tensor:
    per_elem = F.huber_loss(pred, target, delta=delta, reduction="none")
    denom = mask.sum().clamp_min(1.0)
    return (per_elem * mask).sum() / denom


def evaluate(model, loader, device):
    model.eval()
    food_true, food_pred = [], []
    n_true_n, n_pred_n, n_mask = [], [], []
    c_true_n, c_pred_n, c_mask = [], [], []
    regime_true, regime_pred = [], []
    with torch.no_grad():
        for X_b, food_b, n_b, n_m, c_b, c_m, regime_b in loader:
            X_b = X_b.to(device)
            fl, npred, cpred, rl = model(X_b)
            food_pred.extend(fl.argmax(-1).cpu().numpy())
            food_true.extend(food_b.numpy())
            n_pred_n.extend(npred.cpu().numpy())
            n_true_n.extend(n_b.numpy())
            n_mask.extend(n_m.numpy())
            c_pred_n.extend(cpred.cpu().numpy())
            c_true_n.extend(c_b.numpy())
            c_mask.extend(c_m.numpy())
            regime_pred.extend(rl.argmax(-1).cpu().numpy())
            regime_true.extend(regime_b.numpy())

    food_true = np.array(food_true)
    food_pred = np.array(food_pred)
    food_acc = float(accuracy_score(food_true, food_pred) * 100.0)
    food_f1 = float(f1_score(food_true, food_pred, average="macro", zero_division=0))

    n_mask = np.array(n_mask) == 1
    n_true_raw = loader.dataset.inverse_transform_n(np.array(n_true_n))
    n_pred_raw = loader.dataset.inverse_transform_n(np.array(n_pred_n))
    n_mae = float(mean_absolute_error(n_true_raw[n_mask], n_pred_raw[n_mask]))
    n_r2 = float(r2_score(n_true_raw[n_mask], n_pred_raw[n_mask]))

    c_mask = np.array(c_mask) == 1
    c_true_raw = loader.dataset.inverse_transform_c(np.array(c_true_n))
    c_pred_raw = loader.dataset.inverse_transform_c(np.array(c_pred_n))
    c_mae = float(mean_absolute_error(c_true_raw[c_mask], c_pred_raw[c_mask]))
    c_r2 = float(r2_score(c_true_raw[c_mask], c_pred_raw[c_mask]))

    regime_true = np.array(regime_true)
    regime_pred = np.array(regime_pred)
    quality_acc = float(accuracy_score(regime_true, regime_pred))

    return {
        "food_acc": food_acc, "food_f1": food_f1,
        "n_mae": n_mae, "n_r2": n_r2,
        "c_mae": c_mae, "c_r2": c_r2,
        "quality_acc": quality_acc,
    }


def train_fold(model, criterion, train_loader, val_loader, device,
               max_epochs, patience, save_history_path, save_best_model_path):
    opt_params = list(model.parameters()) + list(criterion.parameters())
    optimizer = optim.AdamW(opt_params, lr=1e-3, weight_decay=1e-4)
    warmup_epochs = max(1, int(0.05 * max_epochs))
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs - warmup_epochs)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
    )

    history = {
        "train_loss": [], "loss_food": [], "loss_n": [], "loss_c": [], "loss_regime": [],
        "food_acc": [], "food_f1": [], "n_mae": [], "n_r2": [], "c_mae": [], "c_r2": [],
        "quality_acc": [],
        "task_weight_food": [], "task_weight_n": [], "task_weight_c": [], "task_weight_regime": [],
    }
    best_composite, best_state, bad_epochs = -1.0, None, 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        running, l_food_sum, l_n_sum, l_c_sum, l_regime_sum = 0.0, 0.0, 0.0, 0.0, 0.0
        for X_b, food_b, n_b, n_m, c_b, c_m, regime_b in train_loader:
            X_b, food_b, regime_b = X_b.to(device), food_b.to(device), regime_b.to(device)
            n_b, n_m = n_b.to(device), n_m.to(device)
            c_b, c_m = c_b.to(device), c_m.to(device)
            optimizer.zero_grad()
            food_logits, n_pred, c_pred, regime_logits = model(X_b)
            l_food = F.cross_entropy(food_logits, food_b)
            l_n = masked_huber(n_pred, n_b, n_m)
            l_c = masked_huber(c_pred, c_b, c_m)
            l_regime = F.cross_entropy(regime_logits, regime_b)
            loss = criterion([l_food, l_n, l_c, l_regime])
            loss.backward()
            nn.utils.clip_grad_norm_(opt_params, max_norm=1.0)
            optimizer.step()
            with torch.no_grad():
                criterion.log_sigma.clamp_(-3.0, 3.0)
            running += loss.item() * X_b.size(0)
            l_food_sum += l_food.item() * X_b.size(0)
            l_n_sum += l_n.item() * X_b.size(0)
            l_c_sum += l_c.item() * X_b.size(0)
            l_regime_sum += l_regime.item() * X_b.size(0)
        scheduler.step()

        n = len(train_loader.dataset)
        m = evaluate(model, val_loader, device)
        w = criterion.task_weights.cpu().numpy()
        composite = m["food_acc"] / 100.0 + max(0.0, m["n_r2"]) + max(0.0, m["c_r2"]) + m["quality_acc"]

        history["train_loss"].append(running / n)
        history["loss_food"].append(l_food_sum / n)
        history["loss_n"].append(l_n_sum / n)
        history["loss_c"].append(l_c_sum / n)
        history["loss_regime"].append(l_regime_sum / n)
        history["food_acc"].append(m["food_acc"])
        history["food_f1"].append(m["food_f1"])
        history["n_mae"].append(m["n_mae"])
        history["n_r2"].append(m["n_r2"])
        history["c_mae"].append(m["c_mae"])
        history["c_r2"].append(m["c_r2"])
        history["quality_acc"].append(m["quality_acc"])
        history["task_weight_food"].append(float(w[0]))
        history["task_weight_n"].append(float(w[1]))
        history["task_weight_c"].append(float(w[2]))
        history["task_weight_regime"].append(float(w[3]))

        print(f"  epoch {epoch:3d}/{max_epochs} loss={running/n:.4f} "
              f"(food={l_food_sum/n:.3f} n={l_n_sum/n:.3f} c={l_c_sum/n:.3f} regime={l_regime_sum/n:.3f}) "
              f"w=[{w[0]:.2f},{w[1]:.2f},{w[2]:.2f},{w[3]:.2f}] food_acc={m['food_acc']:.2f}% "
              f"n_r2={m['n_r2']:.3f} c_r2={m['c_r2']:.3f} "
              f"quality_acc={m['quality_acc']:.3f} composite={composite:.4f}", flush=True)

        if composite > best_composite:
            best_composite, bad_epochs = composite, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    model.load_state_dict(best_state)
    os.makedirs(os.path.dirname(save_history_path), exist_ok=True)
    with open(save_history_path, "w") as f:
        json.dump(history, f, indent=4)
    os.makedirs(os.path.dirname(save_best_model_path), exist_ok=True)
    torch.save(model.state_dict(), save_best_model_path)
    return history


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_epochs = 150
    patience = 15
    k_folds = 5
    machine = "Rapeseed"
    task = "category_classification_mt_smartnir"

    dataset_root = os.environ.get("RAPESEED_DATASET_ROOT", _ALL + "/RechercheDataGouv-Rapeseed")
    full_path = f"{dataset_root}/{machine}/ALL.csv"
    full_ds = RapeseedMultiTaskDataset(full_path)

    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_histories = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(full_ds.tissue_raw)), full_ds.tissue_raw)):
        save_history_path = f"history/{task}/{machine}/mt_smartnir_fold{fold + 1}.json"
        if os.path.exists(save_history_path):
            print(f"Skipping fold {fold + 1}/{k_folds}: already completed")
            with open(save_history_path) as f:
                fold_histories.append(json.load(f))
            continue

        print(f"Fold {fold + 1}/{k_folds}")
        save_fold_dir = f"data/{task}/{machine}/fold_{fold + 1}"
        full_ds.fit_normalization_and_labels(train_idx, save_dir=save_fold_dir)

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        # seq_len=1153 is ~3x Mango/Grainit's, so per-sample attention cost is
        # much higher (O(seq^2)); batch_size=128 OOM'd a 16GB GPU at ~15.3GB,
        # measured peak at batch=32 is ~6.1GB -- safe headroom for AdamW state.
        train_loader = DataLoader(full_ds, batch_size=32, sampler=train_sampler, num_workers=4, drop_last=True)
        val_loader = DataLoader(full_ds, batch_size=32, sampler=val_sampler, num_workers=4)

        model = RapeseedCSSEModel(
            n_food_classes=full_ds.n_classes, n_regime_classes=len(full_ds.regime_encoder.classes_),
            c_out=64, n_layers=6, n_heads=6, h_hidden=128, seq_len=full_ds.signal_length,
            use_kan=True, dropout=0.1,
        ).to(device)
        criterion = UncertaintyWeightedLoss(n_tasks=4).to(device)

        save_best_model_path = f"checkpoint/{task}/{machine}/checkpoint_fold{fold + 1}.pth"
        history = train_fold(
            model, criterion, train_loader, val_loader, device,
            max_epochs, patience, save_history_path, save_best_model_path,
        )
        fold_histories.append(history)

    if fold_histories:
        avg_food_acc = np.mean([max(h["food_acc"]) for h in fold_histories])
        avg_n_r2 = np.mean([max(h["n_r2"]) for h in fold_histories])
        avg_c_r2 = np.mean([max(h["c_r2"]) for h in fold_histories])
        avg_quality_acc = np.mean([max(h["quality_acc"]) for h in fold_histories])
        print(f"Average best food_acc={avg_food_acc:.2f}% n_r2={avg_n_r2:.4f} "
              f"c_r2={avg_c_r2:.4f} quality_acc={avg_quality_acc:.4f} "
              f"across {len(fold_histories)} fold(s)")
