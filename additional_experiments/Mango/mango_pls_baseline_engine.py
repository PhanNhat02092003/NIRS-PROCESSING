"""PLS-DA (cultivar classification) + PLS-R (dry_matter regression)
chemometric baselines on the Mango DMC/NIR dataset
(prepare_mango_dataset.py), ported from ../MT-SMART-NIR
(scripts/baselines.py there -> model/pls_baseline.py here). Same 5-fold
splits as the deep-model engines (StratifiedKFold(42) via
ClassificationNIRSDataset / make_folds(42) via RegressionNIRSDataset), so
results are directly comparable. No GPU, no epochs -- runs in seconds.
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

import joblib
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                              mean_squared_error, precision_score, r2_score,
                              recall_score)
from sklearn.model_selection import StratifiedKFold

from dataset.classification_dataset import ClassificationNIRSDataset
from dataset.regression_dataset import RegressionNIRSDataset
from model.pls_baseline import PLSDAClassifier
from regression import make_folds

N_COMPONENTS = 20
K_FOLDS = 5
MACHINE = "Mango"


def run_classification():
    task = "category_classification_pls"
    dataset_root = os.environ.get("MANGO_DATASET_ROOT", _ALL + "/Mendeley-Mango")
    full_ds = ClassificationNIRSDataset(f"{dataset_root}/{MACHINE}/ALL.csv")

    save_history_dir = f"history/{task}/{MACHINE}"
    save_model_dir = f"checkpoint/{task}/{MACHINE}"
    os.makedirs(save_history_dir, exist_ok=True)
    os.makedirs(save_model_dir, exist_ok=True)

    kf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_accs = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(full_ds.y_raw)), full_ds.y_raw)):
        save_history_path = f"{save_history_dir}/pls_classification_fold{fold + 1}.json"
        if os.path.exists(save_history_path):
            print(f"Skipping fold {fold + 1}/{K_FOLDS}: already completed")
            with open(save_history_path) as f:
                fold_accs.append(json.load(f)["val_acc"][0])
            continue

        print(f"Fold {fold + 1}/{K_FOLDS}")
        full_ds.fit_normalization_and_labels(train_idx)

        X_tr, y_tr = full_ds.X[train_idx], full_ds.y[train_idx]
        X_va, y_va = full_ds.X[val_idx], full_ds.y[val_idx]

        clf = PLSDAClassifier(n_components=N_COMPONENTS).fit(X_tr, y_tr)
        y_pred = clf.predict(X_va)

        history = {
            "val_acc": [float(accuracy_score(y_va, y_pred))],
            "val_precision": [float(precision_score(y_va, y_pred, average="macro", zero_division=0))],
            "val_recall": [float(recall_score(y_va, y_pred, average="macro", zero_division=0))],
            "val_f1": [float(f1_score(y_va, y_pred, average="macro", zero_division=0))],
        }
        with open(save_history_path, "w") as f:
            json.dump(history, f, indent=4)
        joblib.dump(clf, f"{save_model_dir}/model_fold{fold + 1}.pkl")
        fold_accs.append(history["val_acc"][0])
        print(f"  val_acc={history['val_acc'][0]:.4f}")

    print(f"PLS-DA average val_acc across {len(fold_accs)} fold(s): {np.mean(fold_accs):.4f}")


def run_regression():
    task = "substance_regression_pls"
    prop = "dry_matter"
    dataset_root = os.environ.get("MANGO_DATASET_ROOT", _ALL + "/Mendeley-Mango")
    full_ds = RegressionNIRSDataset(f"{dataset_root}/{MACHINE}/ALL.csv", prop)

    save_history_dir = f"history/{task}/stage2/{MACHINE}/{prop}"
    save_model_dir = f"checkpoint/{task}/stage2/{MACHINE}/{prop}"
    os.makedirs(save_history_dir, exist_ok=True)
    os.makedirs(save_model_dir, exist_ok=True)

    folds = make_folds(full_ds.y_raw, K_FOLDS)
    fold_r2s = []
    for fold, (train_idx, val_idx) in enumerate(folds):
        save_history_path = f"{save_history_dir}/pls_regression_fold{fold + 1}.json"
        if os.path.exists(save_history_path):
            print(f"Skipping fold {fold + 1}/{K_FOLDS} for {prop}: already completed")
            with open(save_history_path) as f:
                fold_r2s.append(json.load(f)["val_r2"][0])
            continue

        print(f"Fold {fold + 1}/{K_FOLDS} for {prop}")
        full_ds.fit_normalization(train_idx)

        X_tr, y_tr = full_ds.X[train_idx], full_ds.y[train_idx]
        X_va, y_va = full_ds.X[val_idx], full_ds.y[val_idx]

        nc = max(1, min(N_COMPONENTS, X_tr.shape[1], X_tr.shape[0] - 1))
        reg = PLSRegression(n_components=nc, max_iter=1000, scale=True)
        reg.fit(X_tr, y_tr.reshape(-1, 1))
        y_pred = reg.predict(X_va).flatten()

        # metrics in original mg-DM/100g units (undo z-score + log1p), matching
        # regression.py's train() convention for the deep-model engines
        y_va_raw = full_ds.inverse_transform_y(y_va)
        y_pred_raw = full_ds.inverse_transform_y(y_pred)

        history = {
            "val_mae": [float(mean_absolute_error(y_va_raw, y_pred_raw))],
            "val_rmse": [float(mean_squared_error(y_va_raw, y_pred_raw) ** 0.5)],
            "val_r2": [float(r2_score(y_va_raw, y_pred_raw))],
        }
        with open(save_history_path, "w") as f:
            json.dump(history, f, indent=4)
        joblib.dump(reg, f"{save_model_dir}/model_fold{fold + 1}.pkl")
        fold_r2s.append(history["val_r2"][0])
        print(f"  val_r2={history['val_r2'][0]:.4f}")

    print(f"PLS-R ({prop}) average val_r2 across {len(fold_r2s)} fold(s): {np.mean(fold_r2s):.4f}")


if __name__ == "__main__":
    print("=== PLS-DA: cultivar classification ===")
    run_classification()
    print("\n=== PLS-R: dry_matter regression ===")
    run_regression()
