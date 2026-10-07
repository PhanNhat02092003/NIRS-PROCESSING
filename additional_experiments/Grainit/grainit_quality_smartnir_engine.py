"""Quality-grade classification benchmark on the sensAIfood Grainit dataset:
predicts PASS/FAIL directly from spectra, where PASS is wheat (protein >=
11.5% AND moisture <= 14.5%), barley (moisture <= 14.5%), or corn (moisture
<= 13.5%) -- see grainit_mt_smartnir_engine.py's quality_grade -- the same
ground-truth label CSSE-MT's quality head is scored against
(grainit_mt_smartnir_benchmark.py), so SMART-NIR has a directly comparable
baseline in benchmark/quality/Grainit.csv instead of only cereal-type
classification and Moisture/Protein regression.

Rows missing a reading required for their cereal type's grade (e.g. a wheat
row with no Protein value) have an undefined grade and are dropped by
QualityGradeDataset, not guessed.

Reuses the SMART-NIR classifier and training loop from food_classification.py
unchanged -- only the dataset class and target label differ from
grainit_classification_engine.py.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))  # repo root: model/, dataset/, shared engines
_ALL = os.path.join(_ROOT, "..", "all-dataset")
sys.path.insert(0, _ROOT)
os.chdir(_HERE)  # relative outputs (history/, checkpoint/, data/) land next to this script
import json
import os

import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, SubsetRandomSampler

from food_classification import FocalLoss, train
from dataset.quality_grade_dataset import QualityGradeDataset
from grainit_mt_smartnir_engine import needs_protein, quality_grade
from model.classification_model import SMARTNIRClassifier, SmartNIRClassificationConfig


def grainit_quality_label(df):
    cultivar = df["category"].values
    moisture = df["Moisture"].values
    protein = df["Protein"].values
    required_valid = (moisture != -1) & (~needs_protein(cultivar) | (protein != -1))
    grade = quality_grade(cultivar, moisture, protein)
    label = np.where(grade, "PASS", "FAIL")
    return np.where(required_valid, label, None)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_epochs = 200
    patience = 10
    k_folds = 5
    machine = "Grainit"
    task = "category_classification_quality_smartnir"

    dataset_root = os.environ.get("GRAINIT_DATASET_ROOT", _ALL + "/Zenodo-Grainit")
    full_path = f"{dataset_root}/{machine}/ALL.csv"
    full_ds = QualityGradeDataset(full_path, label_fn=grainit_quality_label)

    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_histories = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(full_ds.y_raw)), full_ds.y_raw)):
        print(f"Fold {fold + 1}/{k_folds}")

        save_history_path = f"history/{task}/{machine}/smart_nir_quality_fold{fold + 1}.json"
        if os.path.exists(save_history_path):
            print(f"Skipping fold {fold + 1}/{k_folds}: already completed")
            with open(save_history_path) as f:
                fold_histories.append(json.load(f))
            continue

        save_fold_dir = f"data/{task}/{machine}/fold_{fold + 1}"
        full_ds.fit_normalization_and_labels(train_idx, save_dir=save_fold_dir)

        cfg = SmartNIRClassificationConfig(
            signal_len=full_ds.signal_length,
            out_ch_per_branch=64,
            d_model=128,
            depth=3,
            n_heads=4,
            classifier="kan",
            num_classes=full_ds.n_classes
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

        save_fig_path = f"history/{task}/{machine}/plot_fold{fold + 1}.png"
        os.makedirs(f"history/{task}/{machine}", exist_ok=True)
        save_best_model_path = f"checkpoint/{task}/{machine}/checkpoint_fold{fold + 1}.pth"
        os.makedirs(f"checkpoint/{task}/{machine}", exist_ok=True)

        model, history = train(
            model, train_loader, val_loader, device, max_epochs, criterion, optimizer,
            scheduler=scheduler, patience=patience,
            save_history_path=save_history_path, save_fig_path=save_fig_path,
            save_best_model_path=save_best_model_path
        )

        fold_histories.append(history)

    if fold_histories:
        avg_best_acc = np.mean([max(h["val_acc"]) for h in fold_histories])
        print(f"Average best val_acc across {len(fold_histories)} fold(s): {avg_best_acc:.4f}")
