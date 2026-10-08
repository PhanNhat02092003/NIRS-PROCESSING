"""MT-SMART-NIR's original 3-head Cross-Scale Spectral Encoder design (food
classification + per-substance detection + per-substance concentration
regression, model/mt_smartnir_model.py: MultiTaskSMARTNIRModel) on the
Danang pesticide dataset, restricted to an explicit substance subset
instead of all 19 (per user request, 2026-10-07):
    Thiamethoxam, Permethrin, Azoxystrobin, Difenoconazole, Cypermethrin,
    Chlothianidin

Unlike the Mango/Grainit/Rapeseed CSSE ports, the detection head is kept
here -- pesticide presence/absence is a real concept on this dataset
(Stage 1), not something to strip out. Loss composition mirrors
../MT-SMART-NIR/losses.py's MultiTaskLoss: food CrossEntropy + multi-label
BCE (per-compound pos_weight computed from the training fold, to handle
the presence-rate imbalance) + masked Huber regression (only backpropped
for rows where that substance is actually present -- see
dataset/pesticide_multitask_dataset.py's docstring for why "require all
targets present on one row" doesn't work for sparse, non-overlapping
pesticide presence), combined via the same learnable uncertainty weighting
(Kendall et al. 2018) used by every other CSSE port in this repo.

Usage:
    MACHINE=FLAMENIR python3 multitask_mt_smartnir_engine.py
    MACHINE=OCEANFX BIN=8 python3 multitask_mt_smartnir_engine.py
    SUBSTANCES=A,B,C MACHINE=FLAMENIR python3 multitask_mt_smartnir_engine.py
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, SubsetRandomSampler

from dataset.pesticide_multitask_dataset import PesticideMultiTaskDataset
from model.mt_smartnir_model import MultiTaskSMARTNIRModel
from xspecmamba_joint_pesticide_engine import DEFAULT_SUBSTANCES, masked_huber

load_dotenv()


class UncertaintyWeightedLoss(nn.Module):
    """Learnable task weighting (Kendall et al. 2018), same as every other
    CSSE engine in this repo: L = sum_i 0.5*exp(-2*log_sigma_i)*L_i + log_sigma_i."""

    def __init__(self, n_tasks: int = 3):
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


def compute_pos_weight(pres_train: np.ndarray, max_weight: float = 20.0) -> torch.Tensor:
    """Per-compound BCE pos_weight = n_neg/n_pos on the training fold,
    capped to avoid instability if a substance is especially rare within a
    given fold (ported convention from ../MT-SMART-NIR/losses.py's
    PesticideDetectionLoss docstring)."""
    n = len(pres_train)
    n_pos = pres_train.sum(axis=0)
    pw = (n - n_pos) / np.maximum(n_pos, 1)
    pw = np.clip(pw, 1.0, max_weight)
    return torch.tensor(pw, dtype=torch.float32)


def evaluate(model, loader, device, n_pesticides):
    model.eval()
    food_true, food_pred = [], []
    pest_true, pest_pred_prob = [], []
    conc_true, conc_pred = [], []
    with torch.no_grad():
        for X_b, food_b, pres_b, conc_b in loader:
            X_b = X_b.to(device)
            food_logits, pest_logits, pest_conc = model(X_b)
            food_pred.extend(food_logits.argmax(-1).cpu().numpy())
            food_true.extend(food_b.numpy())
            pest_pred_prob.append(torch.sigmoid(pest_logits).cpu().numpy())
            pest_true.append(pres_b.numpy())
            conc_pred.append(pest_conc.cpu().numpy())
            conc_true.append(conc_b.numpy())

    food_true, food_pred = np.array(food_true), np.array(food_pred)
    food_acc = float(accuracy_score(food_true, food_pred) * 100.0)
    food_f1 = float(f1_score(food_true, food_pred, average="macro", zero_division=0))

    pest_true = np.concatenate(pest_true, axis=0)
    pest_pred_prob = np.concatenate(pest_pred_prob, axis=0)
    pest_pred = (pest_pred_prob >= 0.5).astype(int)
    det_accs, det_f1s = [], []
    for j in range(n_pesticides):
        det_accs.append(accuracy_score(pest_true[:, j], pest_pred[:, j]))
        det_f1s.append(f1_score(pest_true[:, j], pest_pred[:, j], zero_division=0))
    det_acc = float(np.mean(det_accs) * 100.0)
    det_f1 = float(np.mean(det_f1s))

    conc_true = np.concatenate(conc_true, axis=0)
    conc_pred = np.concatenate(conc_pred, axis=0)
    conc_true_mgkg = np.expm1(conc_true)
    conc_pred_mgkg = np.expm1(conc_pred)
    mask = pest_true == 1
    maes, r2s = [], []
    for j in range(n_pesticides):
        m = mask[:, j]
        if m.sum() >= 2:
            maes.append(float(mean_absolute_error(conc_true_mgkg[m, j], conc_pred_mgkg[m, j])))
            r2s.append(float(r2_score(conc_true_mgkg[m, j], conc_pred_mgkg[m, j])))
        else:
            maes.append(float("nan"))
            r2s.append(float("nan"))

    return {
        "food_acc": food_acc, "food_f1": food_f1,
        "det_acc": det_acc, "det_f1": det_f1,
        "reg_mae": maes, "reg_r2": r2s,
    }


def train_fold(model, criterion, pos_weight, train_loader, val_loader, device,
               max_epochs, patience, substances, save_history_path, save_best_model_path):
    n_pest = len(substances)
    opt_params = list(model.parameters()) + list(criterion.parameters())
    optimizer = optim.AdamW(opt_params, lr=1e-3, weight_decay=1e-4)
    warmup_epochs = max(1, int(0.05 * max_epochs))
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs - warmup_epochs)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
    )
    pos_weight = pos_weight.to(device)

    history = {
        "train_loss": [], "loss_food": [], "loss_det": [], "loss_reg": [],
        "food_acc": [], "food_f1": [], "det_acc": [], "det_f1": [],
        "task_weight_food": [], "task_weight_det": [], "task_weight_reg": [],
    }
    for name in substances:
        history[f"reg_mae_{name}"] = []
        history[f"reg_r2_{name}"] = []
    best_composite, best_state, bad_epochs = -1e9, None, 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        running, l_food_sum, l_det_sum, l_reg_sum = 0.0, 0.0, 0.0, 0.0
        for X_b, food_b, pres_b, conc_b in train_loader:
            X_b, food_b = X_b.to(device), food_b.to(device)
            pres_b, conc_b = pres_b.to(device), conc_b.to(device)
            optimizer.zero_grad()
            food_logits, pest_logits, pest_conc = model(X_b)
            l_food = F.cross_entropy(food_logits, food_b)
            l_det = F.binary_cross_entropy_with_logits(pest_logits, pres_b, pos_weight=pos_weight)
            l_reg = masked_huber(pest_conc, conc_b, pres_b)
            loss = criterion([l_food, l_det, l_reg])
            loss.backward()
            nn.utils.clip_grad_norm_(list(model.parameters()) + list(criterion.parameters()), max_norm=1.0)
            optimizer.step()
            with torch.no_grad():
                criterion.log_sigma.clamp_(-3.0, 3.0)
            running += loss.item() * X_b.size(0)
            l_food_sum += l_food.item() * X_b.size(0)
            l_det_sum += l_det.item() * X_b.size(0)
            l_reg_sum += l_reg.item() * X_b.size(0)
        scheduler.step()

        n = len(train_loader.dataset)
        m = evaluate(model, val_loader, device, n_pest)
        w = criterion.task_weights.cpu().numpy()
        mean_r2 = float(np.nanmean(m["reg_r2"]))
        composite = m["food_acc"] / 100.0 + max(0.0, m["det_f1"]) + max(0.0, mean_r2)

        history["train_loss"].append(running / n)
        history["loss_food"].append(l_food_sum / n)
        history["loss_det"].append(l_det_sum / n)
        history["loss_reg"].append(l_reg_sum / n)
        history["food_acc"].append(m["food_acc"])
        history["food_f1"].append(m["food_f1"])
        history["det_acc"].append(m["det_acc"])
        history["det_f1"].append(m["det_f1"])
        history["task_weight_food"].append(float(w[0]))
        history["task_weight_det"].append(float(w[1]))
        history["task_weight_reg"].append(float(w[2]))
        for j, name in enumerate(substances):
            history[f"reg_mae_{name}"].append(m["reg_mae"][j])
            history[f"reg_r2_{name}"].append(m["reg_r2"][j])

        print(f"  epoch {epoch:3d}/{max_epochs} loss={running/n:.4f} "
              f"(food={l_food_sum/n:.3f} det={l_det_sum/n:.3f} reg={l_reg_sum/n:.3f}) "
              f"w=[{w[0]:.2f},{w[1]:.2f},{w[2]:.2f}] food_acc={m['food_acc']:.2f}% "
              f"det_f1={m['det_f1']:.3f} mean_r2={mean_r2:.3f} composite={composite:.4f}", flush=True)

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
    machine = os.environ["MACHINE"]
    task = "multitask_mt_smartnir"

    substances = DEFAULT_SUBSTANCES
    if os.environ.get("SUBSTANCES"):
        substances = [s.strip() for s in os.environ["SUBSTANCES"].split(",")]

    full_path = f"{os.environ['DATASET_ROOT']}/{machine}/ALL.csv"
    bin_factor = int(os.environ.get("BIN", 1))
    sample_n = int(os.environ["SAMPLE_N"]) if os.environ.get("SAMPLE_N") else None
    full_ds = PesticideMultiTaskDataset(full_path, substances, bin_factor=bin_factor, sample_n=sample_n)
    print(f"n samples: {len(full_ds.food_raw)}, signal_len: {full_ds.signal_length}, substances: {substances}")

    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_histories = []

    batch_size = int(os.environ.get("BATCH_SIZE", 64))
    # FOLDS=1,2,3 restricts this process to only those (1-indexed) folds --
    # e.g. to split the 5 folds across 2 GPUs by running two processes, one
    # per CUDA_VISIBLE_DEVICES, each with a disjoint FOLDS set.
    only_folds = None
    if os.environ.get("FOLDS"):
        only_folds = {int(f.strip()) for f in os.environ["FOLDS"].split(",")}

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(full_ds.food_raw)), full_ds.food_raw)):
        if only_folds is not None and (fold + 1) not in only_folds:
            continue
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
        train_loader = DataLoader(full_ds, batch_size=batch_size, sampler=train_sampler, num_workers=4, drop_last=True)
        val_loader = DataLoader(full_ds, batch_size=batch_size, sampler=val_sampler, num_workers=4)

        model = MultiTaskSMARTNIRModel(
            n_food_classes=full_ds.n_classes, n_pesticides=len(substances),
            c_out=64, n_layers=6, n_heads=6, h_hidden=128, seq_len=full_ds.signal_length,
            use_kan=True, dropout=0.1,
        ).to(device)
        criterion = UncertaintyWeightedLoss(n_tasks=3).to(device)
        pos_weight = compute_pos_weight(full_ds.pres_raw[train_idx])

        save_best_model_path = f"checkpoint/{task}/{machine}/checkpoint_fold{fold + 1}.pth"
        history = train_fold(
            model, criterion, pos_weight, train_loader, val_loader, device,
            max_epochs, patience, substances, save_history_path, save_best_model_path,
        )
        fold_histories.append(history)

    if fold_histories:
        avg_food_acc = np.mean([max(h["food_acc"]) for h in fold_histories])
        avg_det_f1 = np.mean([max(h["det_f1"]) for h in fold_histories])
        print(f"Average best food_acc={avg_food_acc:.2f}% det_f1={avg_det_f1:.4f}")
        for name in substances:
            best_r2s = [np.nanmax(h[f"reg_r2_{name}"]) for h in fold_histories]
            print(f"{name}: average best reg_r2 across {len(fold_histories)} fold(s): {np.nanmean(best_r2s):.4f}")
