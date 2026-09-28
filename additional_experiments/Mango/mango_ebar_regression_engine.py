"""Trains EBAR (model/ebar_model.py) as a Stage-2-style regression baseline
against SMART-NIR on the Mango DMC/NIR dataset (prepare_mango_dataset.py) --
reuses the exact same RegressionNIRSDataset preprocessing and make_folds()
fold split, and the same train_for_substance()/fit_xy_scaler() from
regression.py, so results are directly comparable to that
file's pesticide-dataset runs. No Stage 1, see prepare_mango_dataset.py.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))  # repo root: model/, dataset/, shared engines
_ALL = os.path.join(_ROOT, "..", "all-dataset")
sys.path.insert(0, _ROOT)
os.chdir(_HERE)  # relative outputs (history/, checkpoint/, data/) land next to this script
import os

from dotenv import load_dotenv

from dataset.regression_dataset import RegressionNIRSDataset
from regression import train_for_substance
from regression import make_folds

load_dotenv()

if __name__ == "__main__":
    machine = "Mango"
    task = "substance_regression"
    k_folds = 5

    dataset_root = os.environ.get("MANGO_DATASET_ROOT", _ALL + "/Mendeley-Mango")
    full_path = f"{dataset_root}/{machine}/ALL.csv"

    properties = ["dry_matter"]

    for prop in properties:
        print(f"Training for {prop}")

        save_history_dir = f"history/{task}/stage2_ebar/{machine}/{prop}"
        save_checkpoint_dir = f"checkpoint/{task}/stage2_ebar/{machine}/{prop}"

        try:
            full_ds = RegressionNIRSDataset(full_path, prop)
        except ValueError as e:
            print(f"Skipping {prop}: {e}")
            os.makedirs(save_history_dir, exist_ok=True)
            os.makedirs(save_checkpoint_dir, exist_ok=True)
            continue

        folds = make_folds(full_ds.y_raw, k_folds)

        train_for_substance(full_ds.X_raw, full_ds.y_raw, prop, folds,
                             save_history_dir, save_checkpoint_dir)
