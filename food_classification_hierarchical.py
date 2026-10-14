"""Two-stage ("hierarchical") food classification.

The confusion-matrix analysis of the deployed pipeline on test-data found
food-classification errors concentrated almost entirely among 4 spectrally
similar leafy-green categories (Xa Lach / Cai Be Xanh / Cai Thia / Mong
Toi); the other 5 (fruit/root) categories already score ~95-99%. This
splits the 9-way problem into:

  1. gate: binary SMART-NIR classifier, leafy vs non-leafy, plain SG+SNV
     features (same preprocessing as the existing 9-way classifier).
  2. leafy: 4-way SMART-NIR classifier among the leafy set only, trained on
     SG+SNV-then-first-derivative features (dataset.preprocessing.
     preprocess_spectra_derivative) instead of plain SNV absorbance, since
     the derivative emphasizes local curve *shape* rather than overall
     *level* -- the chemometrics-standard way to resolve same-family
     overlapping pigment bands that plain SNV collapses together.
  3. nonleafy: 5-way SMART-NIR classifier among the rest, plain SG+SNV
     (this half is already easy; kept as a specialized model mainly so the
     overall pipeline is a clean 2-level tree, not because it needs help).

All three train on {machine}/ALL_downsampled.csv with StratifiedGroupKFold
on `_group_id` (same leakage-safe protocol as
food_classification_downsampled.py). Checkpoints/history land under
task="category_classification_hier_{gate,leafy,nonleafy}".

Usage: MACHINE=FLAMENIR python3 food_classification_hierarchical.py
       MACHINE=OCEANFX BIN=8 python3 food_classification_hierarchical.py
"""

import os

import joblib
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv

from food_classification import FocalLoss, train
from model.classification_model import SMARTNIRClassifier, SmartNIRClassificationConfig
from dataset.preprocessing import preprocess_spectra, preprocess_spectra_derivative

load_dotenv()

LEAFY = {"Xà Lách", "Cải Bẹ Xanh", "Cải Thìa", "Mồng Tơi"}
MAX_EPOCHS = 200
PATIENCE = 10
K_FOLDS = 5


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


def run_subtask(name: str, machine: str, X_raw: np.ndarray, labels_raw: np.ndarray,
                 groups_raw: np.ndarray, derivative: bool, bin_factor: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    task = f"category_classification_hier_{name}"

    pp_fn = preprocess_spectra_derivative if derivative else preprocess_spectra
    X, keep_mask = pp_fn(X_raw)
    X = bin_wavelengths(X, bin_factor)
    labels = labels_raw[keep_mask]
    groups = groups_raw[keep_mask]

    print(f"[{name}] n={len(X)} signal_len={X.shape[1]} n_classes={len(set(labels))} "
          f"(derivative={derivative}, bin_factor={bin_factor})", flush=True)

    kf = StratifiedGroupKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    splits = list(kf.split(np.arange(len(labels)), labels, groups))

    os.makedirs(f"history/{task}/{machine}", exist_ok=True)
    os.makedirs(f"checkpoint/{task}/{machine}", exist_ok=True)

    fold_histories = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"[{name}] Fold {fold + 1}/{K_FOLDS}", flush=True)
        overlap = set(groups[train_idx]) & set(groups[val_idx])
        assert not overlap, f"{name} fold {fold + 1}: {len(overlap)} groups leak across train/val"

        save_fold_dir = f"data/{task}/{machine}/fold_{fold + 1}"
        os.makedirs(save_fold_dir, exist_ok=True)

        mean = X[train_idx].mean(axis=0, keepdims=True)
        std = X[train_idx].std(axis=0, keepdims=True) + 1e-8
        Xn = (X - mean) / std
        np.savez(f"{save_fold_dir}/stats.npz", mean=mean, std=std)

        le = LabelEncoder().fit(labels[train_idx])
        y = le.transform(labels)
        joblib.dump(le, f"{save_fold_dir}/label_encoder.pkl")

        ds = ArrayDataset(Xn, y)
        train_loader = DataLoader(ds, batch_size=512, sampler=SubsetRandomSampler(train_idx), num_workers=4)
        val_loader = DataLoader(ds, batch_size=512, sampler=SubsetRandomSampler(val_idx), num_workers=4)

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

        save_history_path = f"history/{task}/{machine}/fold{fold + 1}.json"
        save_fig_path = f"history/{task}/{machine}/plot_fold{fold + 1}.png"
        save_best_model_path = f"checkpoint/{task}/{machine}/checkpoint_fold{fold + 1}.pth"

        model, history = train(
            model, train_loader, val_loader, device, MAX_EPOCHS, criterion, optimizer,
            scheduler=scheduler, patience=PATIENCE,
            save_history_path=save_history_path, save_fig_path=save_fig_path,
            save_best_model_path=save_best_model_path,
        )
        fold_histories.append(history)

    avg_val_acc = np.mean([max(h["val_acc"]) for h in fold_histories])
    print(f"[{name}] Average best validation accuracy across folds: {avg_val_acc:.4f}", flush=True)
    return avg_val_acc


def main():
    machine = os.environ["MACHINE"]
    bin_factor = int(os.environ.get("BIN", 1))

    path = f"{os.environ['DATASET_ROOT']}/{machine}/ALL_downsampled.csv"
    df = pd.read_csv(path)
    w_cols = [c for c in df.columns if c.startswith("w_")]
    X_raw_all = df[w_cols].values.astype(np.float32)
    category = df["category"].values
    group_id = df["_group_id"].values
    is_leafy = np.isin(category, list(LEAFY))

    # 1. gate: leafy vs non-leafy, all rows, plain SG+SNV
    run_subtask("gate", machine, X_raw_all, is_leafy.astype(str), group_id,
                derivative=False, bin_factor=bin_factor)

    # 2. leafy fine: 4-way among the leafy set, SG+SNV + 1st derivative
    run_subtask("leafy", machine, X_raw_all[is_leafy], category[is_leafy], group_id[is_leafy],
                derivative=True, bin_factor=bin_factor)

    # 3. nonleafy fine: 5-way among the rest, plain SG+SNV
    run_subtask("nonleafy", machine, X_raw_all[~is_leafy], category[~is_leafy], group_id[~is_leafy],
                derivative=False, bin_factor=bin_factor)


if __name__ == "__main__":
    main()
