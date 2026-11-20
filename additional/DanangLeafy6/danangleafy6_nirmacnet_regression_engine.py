"""Trains NirMACNet (model/nirmacnet_model.py) as a Stage-2-style regression
baseline against SMART-NIR on DanangLeafy6 (prepare_danangleafy6_dataset.py)
-- mirrors Grainit/grainit_nirmacnet_regression_engine.py, with group-safe
folds (StratifiedGroupKFold on `_group_id` via regression.make_group_folds)
instead of plain quantile-stratified KFold.

Env: MACHINE (FLAMENIR | OCEANFX), BIN (OCEANFX: average wavelengths in
groups of 8, matching every other model on this machine).

Usage: MACHINE=FLAMENIR python3 danangleafy6_nirmacnet_regression_engine.py
       MACHINE=OCEANFX BIN=8 python3 danangleafy6_nirmacnet_regression_engine.py
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
from model.nirmacnet_model import NirMACNet, NirMACNetConfig
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
    max_epochs = 200
    patience = 20
    k_folds = 5
    task = "substance_regression"

    full_path = f"{_HERE}/data/{machine}/ALL_synthetic.csv"
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"{full_path} not found -- run prepare_danangleafy6_dataset.py first")

    for substance in SUBSTANCES:
        print(f"Training for {substance}")

        save_history_dir = f"history/{task}/stage2_nirmacnet/{machine}/{substance}"
        save_fig_dir = save_history_dir
        save_best_model_dir = f"checkpoint/{task}/stage2_nirmacnet/{machine}/{substance}"

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
        os.makedirs(save_best_model_dir, exist_ok=True)

        for fold, (train_idx, val_idx) in enumerate(folds):
            save_history_path = f"{save_history_dir}/nirmacnet_regression_fold{fold + 1}.json"

            if os.path.exists(save_history_path):
                print(f"Skipping fold {fold + 1}/{k_folds} for {substance}: already completed")
                with open(save_history_path) as f:
                    fold_histories.append(json.load(f))
                continue

            overlap = set(full_ds.group_raw[train_idx]) & set(full_ds.group_raw[val_idx])
            assert not overlap, f"fold {fold + 1}: {len(overlap)} groups leak across train/val"

            print(f"Fold {fold + 1}/{k_folds} for {substance}")

            save_fold_dir = f"data/{task}/stage2_nirmacnet/{machine}/{substance}/fold_{fold + 1}"
            full_ds.fit_normalization(train_idx, save_dir=save_fold_dir)

            cfg = NirMACNetConfig(num_targets=1)
            model = NirMACNet(cfg).to(device)

            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)
            micro_batch = 128
            accum_steps = 2
            train_loader = DataLoader(full_ds, batch_size=micro_batch, sampler=train_sampler, num_workers=4,
                                       drop_last=True)
            val_loader = DataLoader(full_ds, batch_size=micro_batch, sampler=val_sampler, num_workers=4)

            criterion = nn.MSELoss()
            optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
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

            save_fig_path = f"{save_fig_dir}/plot_fold{fold + 1}.png"
            save_best_model_path = f"{save_best_model_dir}/checkpoint_fold{fold + 1}.pth"

            model, history = train(
                model, train_loader, val_loader, device, max_epochs, criterion, optimizer,
                scheduler=scheduler, patience=patience, inverse_transform_y=full_ds.inverse_transform_y,
                accum_steps=accum_steps,
                save_history_path=save_history_path, save_fig_path=save_fig_path,
                save_best_model_path=save_best_model_path, log_space_metrics=True,
            )

            fold_histories.append(history)

        if fold_histories:
            avg_best_r2 = np.mean([max(h["val_r2"]) for h in fold_histories])
            print(f"Average best validation R2 across folds for {substance}: {avg_best_r2:.4f}")
