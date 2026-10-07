"""Quality-grade classification benchmark on the INRAE rapeseed NIR dataset
using GuidedDCNet (model/guideddcnet_model.py) -- see
rapeseed_quality_smartnir_engine.py's docstring for the N-regime label.
Reuses the exact same QualityGradeDataset, StratifiedKFold split, and
pretrain()/train_diffusion() from food_classification.py as
rapeseed_guideddcnet_classification_engine.py; only the dataset class and
target label differ.
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
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, SubsetRandomSampler

from dataset.quality_grade_dataset import QualityGradeDataset
from food_classification import pretrain, train_diffusion
from model.guideddcnet_model import GuidedDCNet, GuidedDCNetConfig
from rapeseed_quality_smartnir_engine import rapeseed_quality_label

load_dotenv()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pretrain_epochs = 25
    max_epochs = 200
    patience = None  # fixed epoch budget for every fold, see train_diffusion docstring
    k_folds = 5
    lam = 0.5
    machine = "Rapeseed"
    task = "category_classification_quality_guideddcnet"

    dataset_root = os.environ.get("RAPESEED_DATASET_ROOT", _ALL + "/RechercheDataGouv-Rapeseed")
    full_path = f"{dataset_root}/{machine}/ALL.csv"
    full_ds = QualityGradeDataset(full_path, label_fn=rapeseed_quality_label)

    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_histories = []

    save_history_dir = f"history/{task}/{machine}"
    save_checkpoint_dir = f"checkpoint/{task}/{machine}"
    os.makedirs(save_history_dir, exist_ok=True)
    os.makedirs(save_checkpoint_dir, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(full_ds.y_raw)), full_ds.y_raw)):
        save_history_path = f"{save_history_dir}/guideddcnet_quality_fold{fold + 1}.json"

        if os.path.exists(save_history_path):
            print(f"Skipping fold {fold + 1}/{k_folds}: already completed")
            with open(save_history_path) as f:
                fold_histories.append(json.load(f))
            continue

        print(f"Fold {fold + 1}/{k_folds}")

        save_fold_dir = f"data/{task}/{machine}/fold_{fold + 1}"
        full_ds.fit_normalization_and_labels(train_idx, save_dir=save_fold_dir)

        cfg = GuidedDCNetConfig(num_classes=full_ds.n_classes)
        model = GuidedDCNet(cfg).to(device)
        data_head = nn.Linear(model.data_encoder.embed_dim, full_ds.n_classes).to(device)

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(full_ds, batch_size=512, sampler=train_sampler, num_workers=4, drop_last=True)
        val_loader = DataLoader(full_ds, batch_size=512, sampler=val_sampler, num_workers=4)

        pretrain_params = (
            list(model.mcgm.parameters())
            + list(model.data_encoder.parameters())
            + list(data_head.parameters())
        )
        pretrain_optimizer = optim.Adam(pretrain_params, lr=5e-4)
        pretrain(model, data_head, train_loader, val_loader, device, pretrain_epochs, pretrain_optimizer)

        optimizer = optim.Adam([
            {"params": list(model.mcgm.parameters()) + list(model.data_encoder.parameters()), "lr": 5e-4},
            {"params": model.unet.parameters(), "lr": 1e-3},
        ])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

        save_fig_path = f"{save_history_dir}/plot_fold{fold + 1}.png"
        save_best_model_path = f"{save_checkpoint_dir}/checkpoint_fold{fold + 1}.pth"

        model, history = train_diffusion(
            model, train_loader, val_loader, device, max_epochs, full_ds.n_classes, optimizer,
            scheduler=scheduler, lam=lam, patience=patience,
            save_history_path=save_history_path, save_fig_path=save_fig_path,
            save_best_model_path=save_best_model_path,
        )

        fold_histories.append(history)

    avg_val_acc = np.mean([max(h["val_acc"]) for h in fold_histories])
    print(f"Average best validation accuracy across folds: {avg_val_acc:.4f}")
