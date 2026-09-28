"""Stage-2-style (continuous) regression benchmark on the Mango DMC/NIR
dataset (prepare_mango_dataset.py): dry matter content from NIR spectra of
10 mango cultivars. Reuses SMART-NIR's regressor and training loop from
regression.py unchanged; single target (dry_matter) instead of
the 19 pesticide substances. No Stage 1 -- see prepare_mango_dataset.py's
docstring for why.
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
from torch.utils.data import DataLoader, SubsetRandomSampler

from dataset.regression_dataset import RegressionNIRSDataset
from model.regression_model import SMARTNIRRegressor, SmartNIRRegressionConfig
from regression import make_folds, train

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_epochs = 500
    patience = 50
    k_folds = 5
    machine = "Mango"
    task = "substance_regression"

    dataset_root = os.environ.get("MANGO_DATASET_ROOT", _ALL + "/Mendeley-Mango")
    full_path = f"{dataset_root}/{machine}/ALL.csv"

    properties = ["dry_matter"]

    for prop in properties:
        print(f"Training for {prop}")

        save_history_dir = f"history/{task}/stage2/{machine}/{prop}"
        save_fig_dir = f"history/{task}/stage2/{machine}/{prop}"
        save_best_model_dir = f"checkpoint/{task}/stage2/{machine}/{prop}"

        try:
            full_ds = RegressionNIRSDataset(full_path, prop)
        except ValueError as e:
            print(f"Skipping {prop}: {e}")
            os.makedirs(save_history_dir, exist_ok=True)
            os.makedirs(save_fig_dir, exist_ok=True)
            os.makedirs(save_best_model_dir, exist_ok=True)
            continue

        folds = make_folds(full_ds.y_raw, k_folds)

        fold_histories = []

        os.makedirs(save_history_dir, exist_ok=True)
        os.makedirs(save_fig_dir, exist_ok=True)
        os.makedirs(save_best_model_dir, exist_ok=True)

        for fold, (train_idx, val_idx) in enumerate(folds):
            save_history_path = f"{save_history_dir}/smart_nir_regression_fold{fold + 1}.json"

            if os.path.exists(save_history_path):
                print(f"Skipping fold {fold + 1}/{k_folds} for {prop}: already completed")
                with open(save_history_path) as f:
                    fold_histories.append(json.load(f))
                continue

            print(f"Fold {fold + 1}/{k_folds} for {prop}")

            save_fold_dir = f"data/{task}/stage2/{machine}/{prop}/fold_{fold + 1}"
            full_ds.fit_normalization(train_idx, save_dir=save_fold_dir)

            cfg = SmartNIRRegressionConfig(
                signal_len=full_ds.X_raw.shape[1],
                out_ch_per_branch=64,
                d_model=128,
                depth=3,
                n_heads=4,
                classifier="kan",
                num_targets=1,
                kan_basis=8
            )

            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)

            train_loader = DataLoader(full_ds, batch_size=512, sampler=train_sampler, num_workers=4)
            val_loader = DataLoader(full_ds, batch_size=512, sampler=val_sampler, num_workers=4)

            model = SMARTNIRRegressor(cfg).to(device)

            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5
            )

            save_fig_path = f"{save_fig_dir}/plot_fold{fold + 1}.png"
            save_best_model_path = f"{save_best_model_dir}/checkpoint_fold{fold + 1}.pth"

            model, history = train(
                model, train_loader, val_loader, device, max_epochs, criterion, optimizer,
                scheduler=scheduler, patience=patience, inverse_transform_y=full_ds.inverse_transform_y,
                save_history_path=save_history_path, save_fig_path=save_fig_path,
                save_best_model_path=save_best_model_path
            )

            fold_histories.append(history)

        if fold_histories:
            avg_best_r2 = np.mean([max(h["val_r2"]) for h in fold_histories])
            print(f"{prop}: average best val_r2 across {len(fold_histories)} fold(s): {avg_best_r2:.4f}")
