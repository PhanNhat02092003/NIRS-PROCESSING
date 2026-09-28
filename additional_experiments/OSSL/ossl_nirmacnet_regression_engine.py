"""Trains NirMACNet (model/nirmacnet_model.py) as a Stage-2-style regression
baseline against SMART-NIR on the OSSL soil VisNIR dataset
(prepare_ossl_dataset.py) -- reuses regression.py's
`train()`/`make_folds()` verbatim and the same optimizer/scheduler choices
as regression.py, so results are directly comparable to
that file's pesticide-dataset runs. No Stage 1, see prepare_ossl_dataset.py.
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
from model.nirmacnet_model import NirMACNet, NirMACNetConfig
from regression import make_folds, train

load_dotenv()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_epochs = 200
    patience = 20
    k_folds = 5
    machine = "OSSL"
    task = "substance_regression"

    dataset_root = os.environ.get("OSSL_DATASET_ROOT", _ALL + "/OSSL-Soil")
    full_path = f"{dataset_root}/{machine}/ALL.csv"

    properties = [
        "Organic_Carbon", "pH_H2O", "Clay_Content", "Sand_Content", "Silt_Content"
    ]

    for prop in properties:
        print(f"Training for {prop}")

        save_history_dir = f"history/{task}/stage2_nirmacnet/{machine}/{prop}"
        save_fig_dir = save_history_dir
        save_best_model_dir = f"checkpoint/{task}/stage2_nirmacnet/{machine}/{prop}"

        try:
            full_ds = RegressionNIRSDataset(full_path, prop)
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
            save_history_path = f"{save_history_dir}/nirmacnet_regression_fold{fold + 1}.json"

            if os.path.exists(save_history_path):
                print(f"Skipping fold {fold + 1}/{k_folds} for {prop}: already completed")
                with open(save_history_path) as f:
                    fold_histories.append(json.load(f))
                continue

            print(f"Fold {fold + 1}/{k_folds} for {prop}")

            save_fold_dir = f"data/{task}/stage2_nirmacnet/{machine}/{prop}/fold_{fold + 1}"
            full_ds.fit_normalization(train_idx, save_dir=save_fold_dir)

            cfg = NirMACNetConfig(num_targets=1)
            model = NirMACNet(cfg).to(device)

            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)
            # micro_batch + accum_steps, not a plain batch_size=512: NirMACNet
            # keeps the full sequence length through several channels-256
            # Conv1d layers (unlike SMART-NIR's stride-4 downsampling), so
            # per-forward-pass activation memory scales with signal length;
            # OSSL's 1049 bands are ~8x FLAMENIR's 128, mirroring the
            # OCEANFX OOM risk this pattern was originally introduced for.
            micro_batch = 128
            accum_steps = 4
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
                save_best_model_path=save_best_model_path,
            )

            fold_histories.append(history)

        if fold_histories:
            avg_best_r2 = np.mean([max(h["val_r2"]) for h in fold_histories])
            print(f"Average best validation R2 across folds for {prop}: {avg_best_r2:.4f}")
