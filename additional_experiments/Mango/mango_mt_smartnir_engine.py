"""Runs MT-SMART-NIR's Cross-Scale Spectral Encoder (CSSE) on the Mango
DMC/NIR dataset (prepare_mango_dataset.py: 10 cultivars, dry_matter), ported
from ../MT-SMART-NIR (model_multitask.py / losses.py / train_multitask.py
there). Only 2 tasks -- cultivar classification + dry_matter regression --
since Mango has no presence/absence concept (dry_matter is always measured),
so the pesticide-detection head and its loss/mask are dropped entirely
(model/mt_smartnir_model.py: MangoCSSEModel).

A dry-matter "grade" decision replaces the pesticide safety evaluator:
PASS if dry_matter >= a per-cultivar maturity threshold, FAIL otherwise.
Thresholds (industry convention): cultivar 'r2e2' uses 13%, every other
cultivar uses 15%. True grade uses the ground-truth cultivar + dry_matter;
predicted grade uses the model's OWN predicted cultivar (to pick which
threshold applies) + predicted dry_matter -- mirroring how MT-SMART-NIR's
SafetyEvaluator uses predicted detection (not ground truth) to decide which
concentration counts.
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

from dataset.mango_multitask_dataset import MangoMultiTaskDataset
from model.mt_smartnir_model import MangoCSSEModel

R2E2_NAME = "r2e2"
THRESHOLD_R2E2 = 13.0
THRESHOLD_DEFAULT = 15.0


def dm_threshold(cultivar_names: np.ndarray) -> np.ndarray:
    return np.where(cultivar_names == R2E2_NAME, THRESHOLD_R2E2, THRESHOLD_DEFAULT)


class UncertaintyWeightedLoss(nn.Module):
    """Learnable task weighting (Kendall et al. 2018), ported from
    ../MT-SMART-NIR/losses.py: L = sum_i 0.5*exp(-2*log_sigma_i)*L_i + log_sigma_i."""

    def __init__(self, n_tasks: int = 2):
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


def evaluate(model, loader, label_encoder, device):
    model.eval()
    food_true, food_pred = [], []
    reg_true_norm, reg_pred_norm = [], []
    with torch.no_grad():
        for X_b, food_b, reg_b in loader:
            X_b = X_b.to(device)
            fl, rp = model(X_b)
            food_pred.extend(fl.argmax(-1).cpu().numpy())
            food_true.extend(food_b.numpy())
            reg_pred_norm.extend(rp.cpu().numpy())
            reg_true_norm.extend(reg_b.numpy())

    food_true = np.array(food_true)
    food_pred = np.array(food_pred)
    food_acc = float(accuracy_score(food_true, food_pred) * 100.0)
    food_f1 = float(f1_score(food_true, food_pred, average="macro", zero_division=0))

    reg_true_raw = loader.dataset.inverse_transform_y(np.array(reg_true_norm))
    reg_pred_raw = loader.dataset.inverse_transform_y(np.array(reg_pred_norm))
    reg_mae = float(mean_absolute_error(reg_true_raw, reg_pred_raw))
    reg_r2 = float(r2_score(reg_true_raw, reg_pred_raw))

    cultivar_true = label_encoder.inverse_transform(food_true)
    cultivar_pred = label_encoder.inverse_transform(food_pred)
    true_grade = reg_true_raw >= dm_threshold(cultivar_true)
    pred_grade = reg_pred_raw >= dm_threshold(cultivar_pred)
    quality_acc = float(accuracy_score(true_grade, pred_grade))

    return {
        "food_acc": food_acc, "food_f1": food_f1,
        "reg_mae": reg_mae, "reg_r2": reg_r2,
        "quality_acc": quality_acc,
    }


def train_fold(model, criterion, train_loader, val_loader, label_encoder, device,
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
        "train_loss": [], "loss_food": [], "loss_reg": [],
        "food_acc": [], "food_f1": [], "reg_mae": [], "reg_r2": [], "quality_acc": [],
        "task_weight_food": [], "task_weight_reg": [],
    }
    best_composite, best_state, bad_epochs = -1.0, None, 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        running, l_food_sum, l_reg_sum = 0.0, 0.0, 0.0
        for X_b, food_b, reg_b in train_loader:
            X_b, food_b, reg_b = X_b.to(device), food_b.to(device), reg_b.to(device)
            optimizer.zero_grad()
            food_logits, reg_pred = model(X_b)
            l_food = F.cross_entropy(food_logits, food_b)
            l_reg = F.huber_loss(reg_pred, reg_b, delta=0.5)
            loss = criterion([l_food, l_reg])
            loss.backward()
            nn.utils.clip_grad_norm_(opt_params, max_norm=1.0)
            optimizer.step()
            with torch.no_grad():
                criterion.log_sigma.clamp_(-3.0, 3.0)
            running += loss.item() * X_b.size(0)
            l_food_sum += l_food.item() * X_b.size(0)
            l_reg_sum += l_reg.item() * X_b.size(0)
        scheduler.step()

        n = len(train_loader.dataset)
        m = evaluate(model, val_loader, label_encoder, device)
        w = criterion.task_weights.cpu().numpy()
        composite = m["food_acc"] / 100.0 + max(0.0, m["reg_r2"]) + m["quality_acc"]

        history["train_loss"].append(running / n)
        history["loss_food"].append(l_food_sum / n)
        history["loss_reg"].append(l_reg_sum / n)
        history["food_acc"].append(m["food_acc"])
        history["food_f1"].append(m["food_f1"])
        history["reg_mae"].append(m["reg_mae"])
        history["reg_r2"].append(m["reg_r2"])
        history["quality_acc"].append(m["quality_acc"])
        history["task_weight_food"].append(float(w[0]))
        history["task_weight_reg"].append(float(w[1]))

        print(f"  epoch {epoch:3d}/{max_epochs} loss={running/n:.4f} "
              f"(food={l_food_sum/n:.3f} reg={l_reg_sum/n:.3f}) w=[{w[0]:.2f},{w[1]:.2f}] "
              f"food_acc={m['food_acc']:.2f}% reg_r2={m['reg_r2']:.3f} "
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
    machine = "Mango"
    task = "category_classification_mt_smartnir"
    reg_task = "substance_regression_mt_smartnir"

    dataset_root = os.environ.get("MANGO_DATASET_ROOT", _ALL + "/Mendeley-Mango")
    full_path = f"{dataset_root}/{machine}/ALL.csv"
    full_ds = MangoMultiTaskDataset(full_path, target_column="dry_matter")

    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_histories = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(full_ds.y_raw)), full_ds.cultivar_raw)):
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
        train_loader = DataLoader(full_ds, batch_size=512, sampler=train_sampler, num_workers=4, drop_last=True)
        val_loader = DataLoader(full_ds, batch_size=512, sampler=val_sampler, num_workers=4)

        model = MangoCSSEModel(
            n_food_classes=full_ds.n_classes, c_out=64, n_layers=6, n_heads=6,
            h_hidden=128, seq_len=full_ds.signal_length, use_kan=True, dropout=0.1,
        ).to(device)
        criterion = UncertaintyWeightedLoss(n_tasks=2).to(device)

        save_best_model_path = f"checkpoint/{task}/{machine}/checkpoint_fold{fold + 1}.pth"
        history = train_fold(
            model, criterion, train_loader, val_loader, full_ds.label_encoder, device,
            max_epochs, patience, save_history_path, save_best_model_path,
        )
        fold_histories.append(history)

    if fold_histories:
        avg_food_acc = np.mean([max(h["food_acc"]) for h in fold_histories])
        avg_reg_r2 = np.mean([max(h["reg_r2"]) for h in fold_histories])
        avg_quality_acc = np.mean([max(h["quality_acc"]) for h in fold_histories])
        print(f"Average best food_acc={avg_food_acc:.2f}% reg_r2={avg_reg_r2:.4f} "
              f"quality_acc={avg_quality_acc:.4f} across {len(fold_histories)} fold(s)")
