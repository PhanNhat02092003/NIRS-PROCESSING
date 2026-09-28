"""Trains XSpecMamba/SpectralMamba (model/xspecmamba_model.py) as a
Stage-2-style regression baseline against SMART-NIR on the Mango DMC/NIR
dataset (prepare_mango_dataset.py) -- reuses regression.py's
`train()`/`make_folds()` verbatim and the same GAF/RP/Corr image conversion
+ optimizer/scheduler as regression.py, so results are
directly comparable to that file's pesticide-dataset runs. No Stage 1, see
prepare_mango_dataset.py.
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
from torch.utils.data import DataLoader, SubsetRandomSampler

from dataset.regression_dataset import RegressionNIRSDataset
from model.xspecmamba_model import XSpecMamba, XSpecMambaConfig
from regression import make_folds, train
from regression import SpectralImageDataset

load_dotenv()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_epochs = 200
    patience = 20
    k_folds = 5
    img_size = 64
    machine = "Mango"
    task = "substance_regression"

    dataset_root = os.environ.get("MANGO_DATASET_ROOT", _ALL + "/Mendeley-Mango")
    full_path = f"{dataset_root}/{machine}/ALL.csv"

    properties = ["dry_matter"]

    for prop in properties:
        print(f"Training for {prop}")

        save_history_dir = f"history/{task}/stage2_xspecmamba/{machine}/{prop}"
        save_best_model_dir = f"checkpoint/{task}/stage2_xspecmamba/{machine}/{prop}"

        try:
            # apply_savgol_snv=False: the reference paper explicitly does
            # NOT apply Savitzky-Golay or scatter correction externally --
            # its NIR Gradient Enhancement module (model/xspecmamba_model.py)
            # is designed to learn that transform end-to-end from
            # relatively raw spectra instead.
            full_ds = RegressionNIRSDataset(full_path, prop, apply_savgol_snv=False)
        except ValueError as e:
            print(f"Skipping {prop}: {e}")
            os.makedirs(save_history_dir, exist_ok=True)
            os.makedirs(save_best_model_dir, exist_ok=True)
            continue

        folds = make_folds(full_ds.y_raw, k_folds)

        fold_histories = []
        os.makedirs(save_history_dir, exist_ok=True)
        os.makedirs(save_best_model_dir, exist_ok=True)

        for fold, (train_idx, val_idx) in enumerate(folds):
            save_history_path = f"{save_history_dir}/xspecmamba_regression_fold{fold + 1}.json"

            if os.path.exists(save_history_path):
                print(f"Skipping fold {fold + 1}/{k_folds} for {prop}: already completed")
                with open(save_history_path) as f:
                    fold_histories.append(json.load(f))
                continue

            print(f"Fold {fold + 1}/{k_folds} for {prop}")

            save_fold_dir = f"data/{task}/stage2_xspecmamba/{machine}/{prop}/fold_{fold + 1}"
            full_ds.fit_normalization(train_idx, save_dir=save_fold_dir, apply_zscore_X=False)
            img_ds = SpectralImageDataset(full_ds, img_size=img_size)

            cfg = XSpecMambaConfig(img_size=img_size, n_outputs=1, emb_dim=128, depth=1, n_dirs=4,
                                    backbone="mamba")
            model = XSpecMamba(cfg).to(device)

            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)
            train_loader = DataLoader(img_ds, batch_size=64, sampler=train_sampler, num_workers=4, drop_last=True)
            val_loader = DataLoader(img_ds, batch_size=64, sampler=val_sampler, num_workers=4)

            criterion = nn.MSELoss()
            # AdamW lr=5e-4 peak + 5% linear warmup + cosine annealing, grad
            # clip max-norm 1.0 -- matches the reference paper's training
            # recipe (beta1=0.9, beta2=0.999, weight_decay=0.01 are AdamW's
            # own defaults already).
            optimizer = optim.AdamW(model.parameters(), lr=5e-4)
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

            save_fig_path = f"{save_history_dir}/plot_fold{fold + 1}.png"
            save_best_model_path = f"{save_best_model_dir}/checkpoint_fold{fold + 1}.pth"

            model, history = train(
                model, train_loader, val_loader, device, max_epochs, criterion, optimizer,
                scheduler=scheduler, patience=patience, inverse_transform_y=full_ds.inverse_transform_y,
                grad_clip_norm=1.0, use_amp=True,
                save_history_path=save_history_path, save_fig_path=save_fig_path,
                save_best_model_path=save_best_model_path,
            )

            fold_histories.append(history)

        if fold_histories:
            avg_best_r2 = np.mean([max(h["val_r2"]) for h in fold_histories])
            print(f"Average best validation R2 across folds for {prop}: {avg_best_r2:.4f}")
