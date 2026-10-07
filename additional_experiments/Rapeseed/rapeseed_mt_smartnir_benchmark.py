"""Measures the classification/regression metrics the Rapeseed benchmark
CSVs track (accuracy/precision/recall/f1 and mse/mae/rmse/r2) for the CSSE
multi-task model (rapeseed_mt_smartnir_engine.py) and appends a "CSSE-MT"
row to each -- see mango_mt_smartnir_benchmark.py's docstring for why this
is a separate pass instead of reading it off the training history.

N-regime (the "quality" classification head) has no baseline-method
equivalent, so it gets its own benchmark/quality/Rapeseed.csv instead of
being forced into the classification or regression table.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_ALL = os.path.join(_ROOT, "..", "all-dataset")
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(_HERE))
os.chdir(_HERE)

import numpy as np
import torch
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              precision_score, r2_score, recall_score)
from sklearn.model_selection import StratifiedKFold

from benchmark_utils import append_classification_row, append_regression_row
from dataset.rapeseed_multitask_dataset import RapeseedMultiTaskDataset
from model.mt_smartnir_model import RapeseedCSSEModel

device = "cuda" if torch.cuda.is_available() else "cpu"
k_folds = 5
machine = "Rapeseed"
task = "category_classification_mt_smartnir"

dataset_root = os.environ.get("RAPESEED_DATASET_ROOT", _ALL + "/RechercheDataGouv-Rapeseed")
full_path = f"{dataset_root}/{machine}/ALL.csv"
full_ds = RapeseedMultiTaskDataset(full_path)

kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
splits = list(kf.split(np.arange(len(full_ds.tissue_raw)), full_ds.tissue_raw))

accs, precs, recs, f1s = [], [], [], []
n_mse, n_mae, n_rmse, n_r2 = [], [], [], []
c_mse, c_mae, c_rmse, c_r2 = [], [], [], []
qaccs, qprecs, qrecs, qf1s = [], [], [], []

for fold, (train_idx, val_idx) in enumerate(splits):
    full_ds.fit_normalization_and_labels(train_idx, save_dir=None)

    model = RapeseedCSSEModel(
        n_food_classes=full_ds.n_classes, n_regime_classes=len(full_ds.regime_encoder.classes_),
        c_out=64, n_layers=6, n_heads=6, h_hidden=128, seq_len=full_ds.signal_length,
        use_kan=True, dropout=0.1,
    ).to(device)
    ckpt_path = f"checkpoint/{task}/{machine}/checkpoint_fold{fold + 1}.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    food_true, food_pred = [], []
    n_true_n, n_pred_n, n_mask = [], [], []
    c_true_n, c_pred_n, c_mask = [], [], []
    regime_true, regime_pred = [], []
    with torch.no_grad():
        for start in range(0, len(val_idx), 32):
            chunk = val_idx[start:start + 32]
            X_b = torch.tensor(full_ds.X[chunk], dtype=torch.float32, device=device)
            fl, npred, cpred, rl = model(X_b)
            food_pred.extend(fl.argmax(-1).cpu().numpy())
            food_true.extend(full_ds.food_y[chunk])
            n_pred_n.extend(npred.cpu().numpy())
            n_true_n.extend(full_ds.n_content_y[chunk])
            n_mask.extend(full_ds.n_content_valid[chunk])
            c_pred_n.extend(cpred.cpu().numpy())
            c_true_n.extend(full_ds.c_content_y[chunk])
            c_mask.extend(full_ds.c_content_valid[chunk])
            regime_pred.extend(rl.argmax(-1).cpu().numpy())
            regime_true.extend(full_ds.regime_y[chunk])

    food_true, food_pred = np.array(food_true), np.array(food_pred)
    accs.append((food_true == food_pred).mean() * 100.0)
    precs.append(precision_score(food_true, food_pred, average="macro", zero_division=0) * 100.0)
    recs.append(recall_score(food_true, food_pred, average="macro", zero_division=0) * 100.0)
    f1s.append(2 * precs[-1] * recs[-1] / (precs[-1] + recs[-1]) if precs[-1] + recs[-1] > 0 else 0.0)

    n_mask = np.array(n_mask) == 1
    n_true_raw = full_ds.inverse_transform_n(np.array(n_true_n))[n_mask]
    n_pred_raw = full_ds.inverse_transform_n(np.array(n_pred_n))[n_mask]
    mse = mean_squared_error(n_true_raw, n_pred_raw)
    n_mse.append(mse)
    n_mae.append(mean_absolute_error(n_true_raw, n_pred_raw))
    n_rmse.append(np.sqrt(mse))
    n_r2.append(r2_score(n_true_raw, n_pred_raw) * 100.0)

    c_mask = np.array(c_mask) == 1
    c_true_raw = full_ds.inverse_transform_c(np.array(c_true_n))[c_mask]
    c_pred_raw = full_ds.inverse_transform_c(np.array(c_pred_n))[c_mask]
    mse = mean_squared_error(c_true_raw, c_pred_raw)
    c_mse.append(mse)
    c_mae.append(mean_absolute_error(c_true_raw, c_pred_raw))
    c_rmse.append(np.sqrt(mse))
    c_r2.append(r2_score(c_true_raw, c_pred_raw) * 100.0)

    regime_true, regime_pred = np.array(regime_true), np.array(regime_pred)
    qaccs.append((regime_true == regime_pred).mean() * 100.0)
    qprecs.append(precision_score(regime_true, regime_pred, average="macro", zero_division=0) * 100.0)
    qrecs.append(recall_score(regime_true, regime_pred, average="macro", zero_division=0) * 100.0)
    qf1s.append(2 * qprecs[-1] * qrecs[-1] / (qprecs[-1] + qrecs[-1]) if qprecs[-1] + qrecs[-1] > 0 else 0.0)

    print(f"Fold {fold + 1}/{k_folds}: acc={accs[-1]:.2f} prec={precs[-1]:.2f} "
          f"rec={recs[-1]:.2f} f1={f1s[-1]:.2f} | n_r2={n_r2[-1]:.2f} c_r2={c_r2[-1]:.2f} "
          f"quality_acc={qaccs[-1]:.2f}")

append_classification_row(
    "benchmark/classification/Rapeseed.csv", "CSSE-MT",
    np.mean(accs), np.std(accs), np.mean(precs), np.std(precs),
    np.mean(recs), np.std(recs), np.mean(f1s), np.std(f1s),
)
append_regression_row(
    "benchmark/regression/Rapeseed.csv", "N_content", "CSSE-MT",
    np.mean(n_mse), np.std(n_mse), np.mean(n_mae), np.std(n_mae),
    np.mean(n_rmse), np.std(n_rmse), np.mean(n_r2), np.std(n_r2),
)
append_regression_row(
    "benchmark/regression/Rapeseed.csv", "C_content", "CSSE-MT",
    np.mean(c_mse), np.std(c_mse), np.mean(c_mae), np.std(c_mae),
    np.mean(c_rmse), np.std(c_rmse), np.mean(c_r2), np.std(c_r2),
)
append_classification_row(
    "benchmark/quality/Rapeseed.csv", "CSSE-MT",
    np.mean(qaccs), np.std(qaccs), np.mean(qprecs), np.std(qprecs),
    np.mean(qrecs), np.std(qrecs), np.mean(qf1s), np.std(qf1s),
)
print("Benchmark CSVs updated.")
