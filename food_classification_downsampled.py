"""Retrains the SMART-NIR food classifier (food_classification.py METHOD=smartnir)
on the downsampled, less-redundant training set produced by
downsample_training_data.py ({machine}/ALL_downsampled.csv), using
StratifiedGroupKFold on the `_group_id` column instead of plain
StratifiedKFold.

Why: food_classification.py's ALL.csv groups rows by (category, 19-substance
concentration vector) into ~320-340 label groups averaging ~1000-1100 rows
each (up to ~5800) -- almost certainly one prepared batch scanned hundreds
of times, not hundreds of independent physical specimens, with no
session/date id to confirm it directly. Plain row-level StratifiedKFold can
split one batch's rows across train and validation, letting a model exploit
batch-level acquisition quirks as a shortcut for the label and inflating the
reported CV accuracy (93.98% for FLAMENIR/SMART-NIR) versus a genuinely
independent test. This script trains on the de-duplicated, capped subset
(see downsample_training_data.py) with StratifiedGroupKFold so no single
batch (`_group_id`) ever straddles train/validation in any fold, giving an
honest benchmark comparable against external test-data (see
evaluate_downsampled_food_classifier.py).

Checkpoints/history land under task="category_classification_downsampled"
-- separate from category_classification (the original SMART-NIR run) and
category_classification_guideddcnet (the deployed model), so neither is
overwritten.

Usage: MACHINE=FLAMENIR python3 food_classification_downsampled.py
       MACHINE=OCEANFX BIN=8 python3 food_classification_downsampled.py
"""

import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import StratifiedGroupKFold
from dotenv import load_dotenv

from food_classification import FocalLoss, train
from model.classification_model import SMARTNIRClassifier, SmartNIRClassificationConfig
from dataset.classification_dataset import ClassificationNIRSDataset

load_dotenv()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_epochs = 200
    patience = 10
    k_folds = 5
    machine = os.environ["MACHINE"]
    task = "category_classification_downsampled"

    full_path = f"{os.environ['DATASET_ROOT']}/{machine}/ALL_downsampled.csv"
    full_ds = ClassificationNIRSDataset(
        full_path, bin_factor=int(os.environ.get("BIN", 1)), group_col="_group_id"
    )
    assert full_ds.group_raw is not None, "ALL_downsampled.csv must have a _group_id column"

    kf = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_histories = []

    splits = list(kf.split(np.arange(len(full_ds.y_raw)), full_ds.y_raw, full_ds.group_raw))
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"Fold {fold + 1}/{k_folds}")
        # sanity: no group straddles train/val
        overlap = set(full_ds.group_raw[train_idx]) & set(full_ds.group_raw[val_idx])
        assert not overlap, f"fold {fold + 1}: {len(overlap)} groups leak across train/val"

        save_fold_dir = f"data/{task}/{machine}/fold_{fold + 1}"
        full_ds.fit_normalization_and_labels(train_idx, save_dir=save_fold_dir)

        cfg = SmartNIRClassificationConfig(
            signal_len=full_ds.signal_length,
            out_ch_per_branch=64,
            d_model=128,
            depth=3,
            n_heads=4,
            classifier="kan",
            num_classes=full_ds.n_classes,
        )

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(full_ds, batch_size=512, sampler=train_sampler, num_workers=4)
        val_loader = DataLoader(full_ds, batch_size=512, sampler=val_sampler, num_workers=4)

        model = SMARTNIRClassifier(cfg).to(device)

        class_counts = np.bincount(full_ds.y[train_idx], minlength=full_ds.n_classes)
        alpha = (class_counts.sum() / (full_ds.n_classes * np.maximum(class_counts, 1)))
        alpha = torch.tensor(alpha, dtype=torch.float32, device=device)
        criterion = FocalLoss(gamma=2.0, alpha=alpha)

        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        warmup_epochs = max(1, int(0.05 * max_epochs))
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs
        )
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs - warmup_epochs
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
        )

        os.makedirs(f"history/{task}/{machine}", exist_ok=True)
        os.makedirs(f"checkpoint/{task}/{machine}", exist_ok=True)
        save_history_path = f"history/{task}/{machine}/smart_nir_classification_fold{fold + 1}.json"
        save_fig_path = f"history/{task}/{machine}/plot_fold{fold + 1}.png"
        save_best_model_path = f"checkpoint/{task}/{machine}/checkpoint_fold{fold + 1}.pth"

        model, history = train(
            model, train_loader, val_loader, device, max_epochs, criterion, optimizer,
            scheduler=scheduler, patience=patience,
            save_history_path=save_history_path, save_fig_path=save_fig_path,
            save_best_model_path=save_best_model_path,
        )

        fold_histories.append(history)

    avg_val_acc = np.mean([max(h["val_acc"]) for h in fold_histories])
    print(f"Average best validation accuracy across folds (group-safe CV): {avg_val_acc:.4f}")


if __name__ == "__main__":
    main()
