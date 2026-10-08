"""Measures the regression metrics the pesticide-6 benchmark CSV tracks
(mse/mae/rmse/r2) for the joint multi-output XSpecMamba baseline
(xspecmamba_joint_pesticide_engine.py) and appends an "XSpecMamba-joint" row
per substance, for comparison against MT-SMARTNIR's concentration head in
the same table (benchmark/multitask_pesticide6/regression/{Machine}.csv).

Only evaluates whichever folds/machines actually have a saved checkpoint --
this run was stopped early by request (FLAMENIR folds 1-2 only, OCEANFX
never started), so this reports a 2-fold average for FLAMENIR and skips
OCEANFX entirely rather than pretending a 5-fold result exists.
"""

import glob
import os
import re

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold

from additional_experiments.benchmark_utils import append_regression_row
from dataset.pesticide_multitask_dataset import PesticideMultiTaskDataset
from model.xspecmamba_model import XSpecMamba, XSpecMambaConfig
from xspecmamba_joint_pesticide_engine import DEFAULT_SUBSTANCES, PesticideImageDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
task = "pesticide_regression_xspecmamba_joint"
substances = DEFAULT_SUBSTANCES
img_size = 64

MACHINE_CFG = {
    "FLAMENIR": {"bin_factor": 1},
    "OCEANFX": {"bin_factor": 8},
}

for machine, cfg in MACHINE_CFG.items():
    ckpt_dir = f"checkpoint/{task}/{machine}"
    hist_dir = f"history/{task}/{machine}"
    # Only folds with a completed history JSON are real results (a stray
    # checkpoint from a fold killed mid-training exists on disk too, but
    # its history was never written -- see this script's own docstring).
    done_folds = sorted(
        int(re.search(r"fold(\d+)", f).group(1))
        for f in glob.glob(f"{hist_dir}/xspecmamba_joint_fold*.json")
    )
    if not done_folds:
        print(f"{machine}: no completed folds, skipping.")
        continue
    print(f"{machine}: evaluating completed folds {done_folds}")

    full_path = f"{os.environ['DATASET_ROOT']}/{machine}/ALL.csv"
    full_ds = PesticideMultiTaskDataset(full_path, substances, bin_factor=cfg["bin_factor"], sample_n=100000)
    splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                  .split(np.arange(len(full_ds.food_raw)), full_ds.food_raw))

    reg_mse = {s: [] for s in substances}
    reg_mae = {s: [] for s in substances}
    reg_rmse = {s: [] for s in substances}
    reg_r2 = {s: [] for s in substances}

    for fold in done_folds:
        train_idx, val_idx = splits[fold - 1]
        full_ds.fit_normalization_and_labels(train_idx, save_dir=None)
        img_ds = PesticideImageDataset(full_ds, img_size=img_size)

        cfg_model = XSpecMambaConfig(img_size=img_size, n_outputs=len(substances), emb_dim=128, depth=1, n_dirs=4,
                                      backbone="mamba")
        model = XSpecMamba(cfg_model).to(device)
        model.load_state_dict(torch.load(f"{ckpt_dir}/checkpoint_fold{fold}.pth", map_location=device))
        model.eval()

        pred_all, true_all, mask_all = [], [], []
        with torch.no_grad():
            for start in range(0, len(val_idx), 64):
                chunk = val_idx[start:start + 64]
                imgs = torch.stack([img_ds[i][0] for i in chunk]).to(device)
                out = model(imgs)
                pred_all.append(out.cpu().numpy())
                true_all.append(np.stack([img_ds[i][2].numpy() for i in chunk]))
                mask_all.append(np.stack([img_ds[i][1].numpy() for i in chunk]))
        pred_all = np.concatenate(pred_all, axis=0)
        true_all = np.concatenate(true_all, axis=0)
        mask_all = np.concatenate(mask_all, axis=0) == 1

        pred_mgkg, true_mgkg = np.expm1(pred_all), np.expm1(true_all)
        for j, name in enumerate(substances):
            m = mask_all[:, j]
            if m.sum() >= 2:
                mse = mean_squared_error(true_mgkg[m, j], pred_mgkg[m, j])
                reg_mse[name].append(mse)
                reg_mae[name].append(mean_absolute_error(true_mgkg[m, j], pred_mgkg[m, j]))
                reg_rmse[name].append(np.sqrt(mse))
                reg_r2[name].append(r2_score(true_mgkg[m, j], pred_mgkg[m, j]) * 100.0)
        print(f"{machine} fold {fold}: done")

    for name in substances:
        if reg_mse[name]:
            append_regression_row(
                f"benchmark/multitask_pesticide6/regression/{machine}.csv", name, "XSpecMamba-joint",
                np.mean(reg_mse[name]), np.std(reg_mse[name]), np.mean(reg_mae[name]), np.std(reg_mae[name]),
                np.mean(reg_rmse[name]), np.std(reg_rmse[name]), np.mean(reg_r2[name]), np.std(reg_r2[name]),
            )
    print(f"{machine}: benchmark CSV updated (n={len(done_folds)} fold(s)).")
