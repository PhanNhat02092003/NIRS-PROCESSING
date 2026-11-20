"""Stage-2-style (continuous) regression benchmark on DanangLeafy6
(prepare_danangleafy6_dataset.py): 6 pesticide concentrations from NIR
spectra. Reuses SMART-NIR's regressor and training loop from regression.py
unchanged -- mirrors Grainit/grainit_regression_stage2_engine.py, with
group-safe folds (StratifiedGroupKFold on `_group_id` via
regression.make_group_folds) instead of plain quantile-stratified KFold.

Env: MACHINE (FLAMENIR | OCEANFX), BIN (OCEANFX: average wavelengths in
groups of 8, matching every other model on this machine).

Usage: MACHINE=FLAMENIR python3 danangleafy6_regression_stage2_engine.py
       MACHINE=OCEANFX BIN=8 python3 danangleafy6_regression_stage2_engine.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))  # repo root: model/, dataset/, shared engines
sys.path.insert(0, os.path.join(_ROOT, "main"))
sys.path.insert(0, os.path.dirname(_HERE))  # additional/: lets Python merge additional/dataset/ into the dataset namespace package
os.chdir(_HERE)  # relative outputs (history/, checkpoint/, data/) land next to this script
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from torch.utils.data import DataLoader, SubsetRandomSampler

from dataset.regression_dataset import RegressionNIRSDataset
from model.regression_model import SMARTNIRRegressor, SmartNIRRegressionConfig
from regression import make_group_folds, train

load_dotenv()

SUBSTANCES = ['Thiamethoxam', 'Permethrin', 'Azoxystrobin',
              'Difenoconazole', 'Cypermethrin', 'Chlothianidin']


def bin_wavelengths(X: np.ndarray, bin_factor: int) -> np.ndarray:
    if bin_factor <= 1:
        return X
    n_out = (X.shape[1] // bin_factor) // 8 * 8
    return X[:, :n_out * bin_factor].reshape(len(X), n_out, bin_factor).mean(axis=2).astype(np.float32)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    machine = os.environ["MACHINE"]
    bin_factor = int(os.environ.get("BIN", 1))
    max_epochs = 500
    patience = 50
    k_folds = 5
    task = "substance_regression"

    full_path = f"{_HERE}/data/{machine}/ALL_synthetic.csv"
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"{full_path} not found -- run prepare_danangleafy6_dataset.py first")

    for substance in SUBSTANCES:
        print(f"Training for {substance}")

        save_history_dir = f"history/{task}/stage2/{machine}/{substance}"
        save_fig_dir = save_history_dir
        save_best_model_dir = f"checkpoint/{task}/stage2/{machine}/{substance}"

        try:
            full_ds = RegressionNIRSDataset(full_path, substance, group_col="_group_id")
        except ValueError as e:
            print(f"Skipping {substance}: {e}")
            os.makedirs(save_history_dir, exist_ok=True)
            os.makedirs(save_best_model_dir, exist_ok=True)
            continue

        full_ds.X_raw = bin_wavelengths(full_ds.X_raw, bin_factor)
        folds = make_group_folds(full_ds.y_raw, full_ds.group_raw, k_folds)

        fold_histories = []
        os.makedirs(save_history_dir, exist_ok=True)
        os.makedirs(save_fig_dir, exist_ok=True)
        os.makedirs(save_best_model_dir, exist_ok=True)

        for fold, (train_idx, val_idx) in enumerate(folds):
            save_history_path = f"{save_history_dir}/smart_nir_regression_fold{fold + 1}.json"

            if os.path.exists(save_history_path):
                print(f"Skipping fold {fold + 1}/{k_folds} for {substance}: already completed")
                with open(save_history_path) as f:
                    fold_histories.append(json.load(f))
                continue

            overlap = set(full_ds.group_raw[train_idx]) & set(full_ds.group_raw[val_idx])
            assert not overlap, f"fold {fold + 1}: {len(overlap)} groups leak across train/val"

            print(f"Fold {fold + 1}/{k_folds} for {substance}")

            save_fold_dir = f"data/{task}/stage2/{machine}/{substance}/fold_{fold + 1}"
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

            train_loader = DataLoader(full_ds, batch_size=256, sampler=train_sampler, num_workers=4)
            val_loader = DataLoader(full_ds, batch_size=256, sampler=val_sampler, num_workers=4)

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
                save_best_model_path=save_best_model_path, log_space_metrics=True
            )

            fold_histories.append(history)

        if fold_histories:
            avg_best_r2 = np.mean([max(h["val_r2"]) for h in fold_histories])
            print(f"{substance}: average best val_r2 across {len(fold_histories)} fold(s): {avg_best_r2:.4f}")
