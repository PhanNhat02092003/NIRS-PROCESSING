"""SMART-NIR classification benchmark on DanangLeafy6 (prepare_
danangleafy6_dataset.py: Danang pesticide dataset, CAP=25 downsampled, the
4 leafy greens collapsed into one Cai Thia-only "Rau La" class).
StratifiedGroupKFold on `_group_id` (same leakage-safe protocol as
main/food_classification_hierarchical.py's run_subtask, reused here
directly since main/food_classification.py's own run_guideddcnet() is
GuidedDCNet-specific and has no SMART-NIR equivalent).

Env: MACHINE (FLAMENIR | OCEANFX), BIN (OCEANFX: average wavelengths in
groups of 8, matching every other deep model on this machine).

Usage: MACHINE=FLAMENIR python3 danangleafy6_classification_engine.py
       MACHINE=OCEANFX BIN=8 python3 danangleafy6_classification_engine.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))  # repo root: model/, dataset/, shared engines
sys.path.insert(0, os.path.join(_ROOT, "main"))
sys.path.insert(0, os.path.dirname(_HERE))  # additional/: lets Python merge additional/dataset/ into the dataset namespace package
os.chdir(_HERE)  # relative outputs (history/, checkpoint/, data/) land next to this script

import joblib
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

from food_classification import FocalLoss, train
from model.classification_model import SMARTNIRClassifier, SmartNIRClassificationConfig
from dataset.preprocessing import preprocess_spectra

MAX_EPOCHS = 200
PATIENCE = 10
K_FOLDS = 5
TASK = "category_classification"


def bin_wavelengths(X: np.ndarray, bin_factor: int) -> np.ndarray:
    if bin_factor <= 1:
        return X
    n_out = (X.shape[1] // bin_factor) // 8 * 8
    return X[:, :n_out * bin_factor].reshape(len(X), n_out, bin_factor).mean(axis=2).astype(np.float32)


class ArrayDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    machine = os.environ["MACHINE"]
    bin_factor = int(os.environ.get("BIN", 1))

    full_path = f"{_HERE}/data/{machine}/ALL.csv"
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"{full_path} not found -- run prepare_danangleafy6_dataset.py first")
    df = pd.read_csv(full_path)
    w_cols = [c for c in df.columns if c.startswith("w_")]
    X_raw_all = df[w_cols].values.astype(np.float32)
    category = df["category"].values
    group_id = df["_group_id"].values

    X, keep_mask = preprocess_spectra(X_raw_all)
    X = bin_wavelengths(X, bin_factor)
    labels = category[keep_mask]
    groups = group_id[keep_mask]
    print(f"n={len(X)} signal_len={X.shape[1]} n_classes={len(set(labels))} (bin_factor={bin_factor})", flush=True)

    kf = StratifiedGroupKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    splits = list(kf.split(np.arange(len(labels)), labels, groups))

    save_history_dir = f"history/{TASK}/{machine}"
    save_checkpoint_dir = f"checkpoint/{TASK}/{machine}"
    os.makedirs(save_history_dir, exist_ok=True)
    os.makedirs(save_checkpoint_dir, exist_ok=True)

    fold_histories = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"Fold {fold + 1}/{K_FOLDS}", flush=True)
        overlap = set(groups[train_idx]) & set(groups[val_idx])
        assert not overlap, f"fold {fold + 1}: {len(overlap)} groups leak across train/val"

        save_fold_dir = f"data/{TASK}/{machine}/fold_{fold + 1}"
        os.makedirs(save_fold_dir, exist_ok=True)

        mean = X[train_idx].mean(axis=0, keepdims=True)
        std = X[train_idx].std(axis=0, keepdims=True) + 1e-8
        Xn = (X - mean) / std
        np.savez(f"{save_fold_dir}/stats.npz", mean=mean, std=std)

        le = LabelEncoder().fit(labels[train_idx])
        y = le.transform(labels)
        joblib.dump(le, f"{save_fold_dir}/label_encoder.pkl")

        ds = ArrayDataset(Xn, y)
        train_loader = DataLoader(ds, batch_size=256, sampler=SubsetRandomSampler(train_idx), num_workers=4)
        val_loader = DataLoader(ds, batch_size=256, sampler=SubsetRandomSampler(val_idx), num_workers=4)

        cfg = SmartNIRClassificationConfig(
            signal_len=X.shape[1], out_ch_per_branch=64, d_model=128,
            depth=3, n_heads=4, classifier="kan", num_classes=len(le.classes_),
        )
        model = SMARTNIRClassifier(cfg).to(device)

        class_counts = np.bincount(y[train_idx], minlength=len(le.classes_))
        alpha = (class_counts.sum() / (len(le.classes_) * np.maximum(class_counts, 1)))
        alpha = torch.tensor(alpha, dtype=torch.float32, device=device)
        criterion = FocalLoss(gamma=2.0, alpha=alpha)

        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        warmup_epochs = max(1, int(0.05 * MAX_EPOCHS))
        warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS - warmup_epochs)
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

        save_history_path = f"{save_history_dir}/fold{fold + 1}.json"
        save_fig_path = f"{save_history_dir}/plot_fold{fold + 1}.png"
        save_best_model_path = f"{save_checkpoint_dir}/checkpoint_fold{fold + 1}.pth"

        model, history = train(
            model, train_loader, val_loader, device, MAX_EPOCHS, criterion, optimizer,
            scheduler=scheduler, patience=PATIENCE,
            save_history_path=save_history_path, save_fig_path=save_fig_path,
            save_best_model_path=save_best_model_path,
        )
        fold_histories.append(history)

    avg_val_acc = np.mean([max(h["val_acc"]) for h in fold_histories])
    print(f"Average best validation accuracy across folds: {avg_val_acc:.4f}", flush=True)


if __name__ == "__main__":
    main()
