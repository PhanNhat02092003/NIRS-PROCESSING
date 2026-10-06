"""Runs MT-SMART-NIR's Cross-Scale Spectral Encoder (CSSE) on the Grainit
cereal NIR dataset (prepare_grainit_dataset.py: barley/corn/wheat, Moisture,
Protein), ported from ../MT-SMART-NIR the same way as mango_mt_smartnir_engine.py.
3 tasks here instead of 2 -- cereal-type classification + Moisture regression
+ Protein regression -- since Grainit tracks two always-measured grain
properties instead of Mango's one. Still no detection head: neither property
has a presence/absence concept (see prepare_grainit_dataset.py's docstring),
so the pesticide-detection head and its loss/mask are dropped entirely
(model/mt_smartnir_model.py: GrainitCSSEModel).

A per-category "quality grade" decision replaces the pesticide safety
evaluator (grading criteria as given by the user, 2026-10-06):
  - wheat:  Protein >= 11.5%  AND  Moisture <= 14.5%
  - barley: Moisture <= 14.5%  (no Protein criterion)
  - corn:   Moisture <= 13.5%  (no Protein criterion)
True grade uses the ground-truth cereal type + ground-truth Moisture/Protein;
predicted grade uses the model's OWN predicted cereal type (to pick which
thresholds apply) + predicted Moisture/Protein -- mirroring how MT-SMART-NIR's
SafetyEvaluator uses predicted detection (not ground truth) to decide which
concentration counts, and matching mango_mt_smartnir_engine.py's dm_threshold.

A handful of rows are missing Moisture or Protein (raw-file 0%-placeholder
rows nulled to -1 by prepare_grainit_dataset.py): their loss contribution for
that target is masked out (see GrainitMultiTaskDataset), and quality-grade
accuracy is only evaluated over rows where every criterion required for that
row's true cereal type has a valid ground-truth value.
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

from dataset.grainit_multitask_dataset import GrainitMultiTaskDataset
from model.mt_smartnir_model import GrainitCSSEModel

MOISTURE_THRESHOLD = {"wheat": 14.5, "barley": 14.5, "corn": 13.5}
PROTEIN_THRESHOLD = {"wheat": 11.5}  # only wheat has a protein criterion


def moisture_threshold(cultivar_names: np.ndarray) -> np.ndarray:
    return np.array([MOISTURE_THRESHOLD[c] for c in cultivar_names])


def needs_protein(cultivar_names: np.ndarray) -> np.ndarray:
    return np.array([c in PROTEIN_THRESHOLD for c in cultivar_names])


def protein_threshold(cultivar_names: np.ndarray) -> np.ndarray:
    return np.array([PROTEIN_THRESHOLD.get(c, np.inf) for c in cultivar_names])


def quality_grade(cultivar_names: np.ndarray, moisture: np.ndarray, protein: np.ndarray) -> np.ndarray:
    moist_ok = moisture <= moisture_threshold(cultivar_names)
    wheat_mask = needs_protein(cultivar_names)
    protein_ok = np.where(wheat_mask, protein >= protein_threshold(cultivar_names), True)
    return moist_ok & protein_ok


class UncertaintyWeightedLoss(nn.Module):
    """Learnable task weighting (Kendall et al. 2018), ported from
    ../MT-SMART-NIR/losses.py: L = sum_i 0.5*exp(-2*log_sigma_i)*L_i + log_sigma_i."""

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


def masked_huber(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, delta: float = 0.5) -> torch.Tensor:
    per_elem = F.huber_loss(pred, target, delta=delta, reduction="none")
    denom = mask.sum().clamp_min(1.0)
    return (per_elem * mask).sum() / denom


def evaluate(model, loader, label_encoder, device):
    model.eval()
    food_true, food_pred = [], []
    moist_true_n, moist_pred_n, moist_mask = [], [], []
    prot_true_n, prot_pred_n, prot_mask = [], [], []
    with torch.no_grad():
        for X_b, food_b, moist_b, moist_m, prot_b, prot_m in loader:
            X_b = X_b.to(device)
            fl, mp, pp = model(X_b)
            food_pred.extend(fl.argmax(-1).cpu().numpy())
            food_true.extend(food_b.numpy())
            moist_pred_n.extend(mp.cpu().numpy())
            moist_true_n.extend(moist_b.numpy())
            moist_mask.extend(moist_m.numpy())
            prot_pred_n.extend(pp.cpu().numpy())
            prot_true_n.extend(prot_b.numpy())
            prot_mask.extend(prot_m.numpy())

    food_true = np.array(food_true)
    food_pred = np.array(food_pred)
    food_acc = float(accuracy_score(food_true, food_pred) * 100.0)
    food_f1 = float(f1_score(food_true, food_pred, average="macro", zero_division=0))

    moist_mask = np.array(moist_mask) == 1
    moist_true_raw_all = loader.dataset.inverse_transform_moisture(np.array(moist_true_n))
    moist_pred_raw_all = loader.dataset.inverse_transform_moisture(np.array(moist_pred_n))
    moisture_mae = float(mean_absolute_error(moist_true_raw_all[moist_mask], moist_pred_raw_all[moist_mask]))
    moisture_r2 = float(r2_score(moist_true_raw_all[moist_mask], moist_pred_raw_all[moist_mask]))

    prot_mask = np.array(prot_mask) == 1
    prot_true_raw_all = loader.dataset.inverse_transform_protein(np.array(prot_true_n))
    prot_pred_raw_all = loader.dataset.inverse_transform_protein(np.array(prot_pred_n))
    protein_mae = float(mean_absolute_error(prot_true_raw_all[prot_mask], prot_pred_raw_all[prot_mask]))
    protein_r2 = float(r2_score(prot_true_raw_all[prot_mask], prot_pred_raw_all[prot_mask]))

    cultivar_true = label_encoder.inverse_transform(food_true)
    cultivar_pred = label_encoder.inverse_transform(food_pred)
    required_valid = moist_mask & (~needs_protein(cultivar_true) | prot_mask)

    true_grade = quality_grade(cultivar_true[required_valid], moist_true_raw_all[required_valid],
                                prot_true_raw_all[required_valid])
    pred_grade = quality_grade(cultivar_pred[required_valid], moist_pred_raw_all[required_valid],
                                prot_pred_raw_all[required_valid])
    quality_acc = float(accuracy_score(true_grade, pred_grade)) if required_valid.sum() else float("nan")

    return {
        "food_acc": food_acc, "food_f1": food_f1,
        "moisture_mae": moisture_mae, "moisture_r2": moisture_r2,
        "protein_mae": protein_mae, "protein_r2": protein_r2,
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
        "train_loss": [], "loss_food": [], "loss_moisture": [], "loss_protein": [],
        "food_acc": [], "food_f1": [], "moisture_mae": [], "moisture_r2": [],
        "protein_mae": [], "protein_r2": [], "quality_acc": [],
        "task_weight_food": [], "task_weight_moisture": [], "task_weight_protein": [],
    }
    best_composite, best_state, bad_epochs = -1.0, None, 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        running, l_food_sum, l_moist_sum, l_prot_sum = 0.0, 0.0, 0.0, 0.0
        for X_b, food_b, moist_b, moist_m, prot_b, prot_m in train_loader:
            X_b, food_b = X_b.to(device), food_b.to(device)
            moist_b, moist_m = moist_b.to(device), moist_m.to(device)
            prot_b, prot_m = prot_b.to(device), prot_m.to(device)
            optimizer.zero_grad()
            food_logits, moist_pred, prot_pred = model(X_b)
            l_food = F.cross_entropy(food_logits, food_b)
            l_moist = masked_huber(moist_pred, moist_b, moist_m)
            l_prot = masked_huber(prot_pred, prot_b, prot_m)
            loss = criterion([l_food, l_moist, l_prot])
            loss.backward()
            nn.utils.clip_grad_norm_(opt_params, max_norm=1.0)
            optimizer.step()
            with torch.no_grad():
                criterion.log_sigma.clamp_(-3.0, 3.0)
            running += loss.item() * X_b.size(0)
            l_food_sum += l_food.item() * X_b.size(0)
            l_moist_sum += l_moist.item() * X_b.size(0)
            l_prot_sum += l_prot.item() * X_b.size(0)
        scheduler.step()

        n = len(train_loader.dataset)
        m = evaluate(model, val_loader, label_encoder, device)
        w = criterion.task_weights.cpu().numpy()
        composite = m["food_acc"] / 100.0 + max(0.0, m["moisture_r2"]) + max(0.0, m["protein_r2"]) + m["quality_acc"]

        history["train_loss"].append(running / n)
        history["loss_food"].append(l_food_sum / n)
        history["loss_moisture"].append(l_moist_sum / n)
        history["loss_protein"].append(l_prot_sum / n)
        history["food_acc"].append(m["food_acc"])
        history["food_f1"].append(m["food_f1"])
        history["moisture_mae"].append(m["moisture_mae"])
        history["moisture_r2"].append(m["moisture_r2"])
        history["protein_mae"].append(m["protein_mae"])
        history["protein_r2"].append(m["protein_r2"])
        history["quality_acc"].append(m["quality_acc"])
        history["task_weight_food"].append(float(w[0]))
        history["task_weight_moisture"].append(float(w[1]))
        history["task_weight_protein"].append(float(w[2]))

        print(f"  epoch {epoch:3d}/{max_epochs} loss={running/n:.4f} "
              f"(food={l_food_sum/n:.3f} moist={l_moist_sum/n:.3f} prot={l_prot_sum/n:.3f}) "
              f"w=[{w[0]:.2f},{w[1]:.2f},{w[2]:.2f}] food_acc={m['food_acc']:.2f}% "
              f"moist_r2={m['moisture_r2']:.3f} prot_r2={m['protein_r2']:.3f} "
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
    machine = "Grainit"
    task = "category_classification_mt_smartnir"

    dataset_root = os.environ.get("GRAINIT_DATASET_ROOT", _ALL + "/Zenodo-Grainit")
    full_path = f"{dataset_root}/{machine}/ALL.csv"
    full_ds = GrainitMultiTaskDataset(full_path)

    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_histories = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(full_ds.cultivar_raw)), full_ds.cultivar_raw)):
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
        train_loader = DataLoader(full_ds, batch_size=64, sampler=train_sampler, num_workers=4, drop_last=True)
        val_loader = DataLoader(full_ds, batch_size=64, sampler=val_sampler, num_workers=4)

        model = GrainitCSSEModel(
            n_food_classes=full_ds.n_classes, c_out=64, n_layers=6, n_heads=6,
            h_hidden=128, seq_len=full_ds.signal_length, use_kan=True, dropout=0.1,
        ).to(device)
        criterion = UncertaintyWeightedLoss(n_tasks=3).to(device)

        save_best_model_path = f"checkpoint/{task}/{machine}/checkpoint_fold{fold + 1}.pth"
        history = train_fold(
            model, criterion, train_loader, val_loader, full_ds.label_encoder, device,
            max_epochs, patience, save_history_path, save_best_model_path,
        )
        fold_histories.append(history)

    if fold_histories:
        avg_food_acc = np.mean([max(h["food_acc"]) for h in fold_histories])
        avg_moist_r2 = np.mean([max(h["moisture_r2"]) for h in fold_histories])
        avg_prot_r2 = np.mean([max(h["protein_r2"]) for h in fold_histories])
        avg_quality_acc = np.mean([max(h["quality_acc"]) for h in fold_histories])
        print(f"Average best food_acc={avg_food_acc:.2f}% moisture_r2={avg_moist_r2:.4f} "
              f"protein_r2={avg_prot_r2:.4f} quality_acc={avg_quality_acc:.4f} "
              f"across {len(fold_histories)} fold(s)")
