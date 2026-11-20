"""Trains EBAR (model/ebar_model.py) as a Stage-2-style regression baseline
against SMART-NIR (danangleafy6_regression_stage2_engine.py) on
DanangLeafy6 (prepare_danangleafy6_dataset.py) -- mirrors
Grainit/grainit_ebar_regression_engine.py, with ONE difference: folds are
split group-safe (StratifiedGroupKFold on `_group_id`, regression.
make_group_folds) instead of plain quantile-stratified KFold, since one
scanned batch must never straddle train/val here (Grainit has no such
grouping).

Env: MACHINE (FLAMENIR | OCEANFX), BIN (OCEANFX: average wavelengths in
groups of 8, matching every other model on this machine).

Usage: MACHINE=FLAMENIR python3 danangleafy6_ebar_regression_engine.py
       MACHINE=OCEANFX BIN=8 python3 danangleafy6_ebar_regression_engine.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))  # repo root: model/, dataset/, shared engines
sys.path.insert(0, os.path.join(_ROOT, "main"))
sys.path.insert(0, os.path.dirname(_HERE))  # additional/: lets Python merge additional/dataset/ into the dataset namespace package
os.chdir(_HERE)  # relative outputs (history/, checkpoint/, data/) land next to this script

import numpy as np
from dotenv import load_dotenv

from dataset.regression_dataset import RegressionNIRSDataset
from regression import make_group_folds, train_for_substance

load_dotenv()

SUBSTANCES = ['Thiamethoxam', 'Permethrin', 'Azoxystrobin',
              'Difenoconazole', 'Cypermethrin', 'Chlothianidin']


def bin_wavelengths(X: np.ndarray, bin_factor: int) -> np.ndarray:
    """Same average-pool-by-bin_factor convention as every other DanangLeafy6/
    main engine (e.g. danangleafy6_classification_engine.py) -- applied here
    post-construction since RegressionNIRSDataset has no bin_factor param."""
    if bin_factor <= 1:
        return X
    n_out = (X.shape[1] // bin_factor) // 8 * 8
    return X[:, :n_out * bin_factor].reshape(len(X), n_out, bin_factor).mean(axis=2).astype(np.float32)


if __name__ == "__main__":
    machine = os.environ["MACHINE"]
    bin_factor = int(os.environ.get("BIN", 1))
    task = "substance_regression"

    full_path = f"{_HERE}/data/{machine}/ALL_synthetic.csv"
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"{full_path} not found -- run prepare_danangleafy6_dataset.py first")

    for substance in SUBSTANCES:
        print(f"Training for {substance}")

        save_history_dir = f"history/{task}/stage2_ebar/{machine}/{substance}"
        save_checkpoint_dir = f"checkpoint/{task}/stage2_ebar/{machine}/{substance}"

        try:
            full_ds = RegressionNIRSDataset(full_path, substance, group_col="_group_id")
        except ValueError as e:
            print(f"Skipping {substance}: {e}")
            os.makedirs(save_history_dir, exist_ok=True)
            os.makedirs(save_checkpoint_dir, exist_ok=True)
            continue

        full_ds.X_raw = bin_wavelengths(full_ds.X_raw, bin_factor)

        folds = make_group_folds(full_ds.y_raw, full_ds.group_raw, k_folds=5)
        for fold, (train_idx, val_idx) in enumerate(folds):
            overlap = set(full_ds.group_raw[train_idx]) & set(full_ds.group_raw[val_idx])
            assert not overlap, f"fold {fold + 1}: {len(overlap)} groups leak across train/val"

        train_for_substance(full_ds.X_raw, full_ds.y_raw, substance, folds,
                             save_history_dir, save_checkpoint_dir, log_space_metrics=True)
