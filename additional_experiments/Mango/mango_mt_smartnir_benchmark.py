"""Measures the classification/regression metrics the Mango benchmark CSVs
track (accuracy/precision/recall/f1 and mse/mae/rmse/r2) for the CSSE
multi-task model (mango_mt_smartnir_engine.py) and appends a "CSSE-MT" row
to each -- those scripts' training loop only logged accuracy/f1 and mae/r2,
not precision/recall/mse/rmse, and never the mean+std-across-folds the
benchmark tables report.

Reconstructs each fold's exact train/val split (same StratifiedKFold seed)
and re-fits normalization on it (deterministic, no randomness) so the
saved checkpoints can be evaluated on the same held-out rows they were
validated on during training, without needing a separate stats-loading path.
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
from dataset.mango_multitask_dataset import MangoMultiTaskDataset
from mango_mt_smartnir_engine import dm_threshold
from model.mt_smartnir_model import MangoCSSEModel

device = "cuda" if torch.cuda.is_available() else "cpu"
k_folds = 5
machine = "Mango"
task = "category_classification_mt_smartnir"

dataset_root = os.environ.get("MANGO_DATASET_ROOT", _ALL + "/Mendeley-Mango")
full_path = f"{dataset_root}/{machine}/ALL.csv"
full_ds = MangoMultiTaskDataset(full_path, target_column="dry_matter")

kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
splits = list(kf.split(np.arange(len(full_ds.y_raw)), full_ds.cultivar_raw))

accs, precs, recs, f1s = [], [], [], []
mses, maes, rmses, r2s = [], [], [], []
qaccs, qprecs, qrecs, qf1s = [], [], [], []

for fold, (train_idx, val_idx) in enumerate(splits):
    full_ds.fit_normalization_and_labels(train_idx, save_dir=None)

    model = MangoCSSEModel(
        n_food_classes=full_ds.n_classes, c_out=64, n_layers=6, n_heads=6,
        h_hidden=128, seq_len=full_ds.signal_length, use_kan=True, dropout=0.1,
    ).to(device)
    ckpt_path = f"checkpoint/{task}/{machine}/checkpoint_fold{fold + 1}.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    food_true, food_pred, reg_true_n, reg_pred_n = [], [], [], []
    with torch.no_grad():
        for start in range(0, len(val_idx), 512):
            chunk = val_idx[start:start + 512]
            X_b = torch.tensor(full_ds.X[chunk], dtype=torch.float32, device=device)
            fl, rp = model(X_b)
            food_pred.extend(fl.argmax(-1).cpu().numpy())
            food_true.extend(full_ds.food_y[chunk])
            reg_pred_n.extend(rp.cpu().numpy())
            reg_true_n.extend(full_ds.reg_y[chunk])

    food_true, food_pred = np.array(food_true), np.array(food_pred)
    accs.append((food_true == food_pred).mean() * 100.0)
    precs.append(precision_score(food_true, food_pred, average="macro", zero_division=0) * 100.0)
    recs.append(recall_score(food_true, food_pred, average="macro", zero_division=0) * 100.0)
    f1s.append(2 * precs[-1] * recs[-1] / (precs[-1] + recs[-1]) if precs[-1] + recs[-1] > 0 else 0.0)

    reg_true_raw = full_ds.inverse_transform_y(np.array(reg_true_n))
    reg_pred_raw = full_ds.inverse_transform_y(np.array(reg_pred_n))
    mse = mean_squared_error(reg_true_raw, reg_pred_raw)
    mses.append(mse)
    maes.append(mean_absolute_error(reg_true_raw, reg_pred_raw))
    rmses.append(np.sqrt(mse))
    r2s.append(r2_score(reg_true_raw, reg_pred_raw) * 100.0)

    cultivar_true = full_ds.label_encoder.inverse_transform(food_true)
    cultivar_pred = full_ds.label_encoder.inverse_transform(food_pred)
    true_grade = reg_true_raw >= dm_threshold(cultivar_true)
    pred_grade = reg_pred_raw >= dm_threshold(cultivar_pred)
    qaccs.append((true_grade == pred_grade).mean() * 100.0)
    qprecs.append(precision_score(true_grade, pred_grade, average="macro", zero_division=0) * 100.0)
    qrecs.append(recall_score(true_grade, pred_grade, average="macro", zero_division=0) * 100.0)
    qf1s.append(2 * qprecs[-1] * qrecs[-1] / (qprecs[-1] + qrecs[-1]) if qprecs[-1] + qrecs[-1] > 0 else 0.0)

    print(f"Fold {fold + 1}/{k_folds}: acc={accs[-1]:.2f} prec={precs[-1]:.2f} "
          f"rec={recs[-1]:.2f} f1={f1s[-1]:.2f} | mse={mses[-1]:.4f} mae={maes[-1]:.4f} "
          f"rmse={rmses[-1]:.4f} r2={r2s[-1]:.2f} | quality_acc={qaccs[-1]:.2f}")

append_classification_row(
    "benchmark/classification/Mango.csv", "CSSE-MT",
    np.mean(accs), np.std(accs), np.mean(precs), np.std(precs),
    np.mean(recs), np.std(recs), np.mean(f1s), np.std(f1s),
)
append_regression_row(
    "benchmark/regression/Mango.csv", "dry_matter", "CSSE-MT",
    np.mean(mses), np.std(mses), np.mean(maes), np.std(maes),
    np.mean(rmses), np.std(rmses), np.mean(r2s), np.std(r2s),
)
append_classification_row(
    "benchmark/quality/Mango.csv", "CSSE-MT",
    np.mean(qaccs), np.std(qaccs), np.mean(qprecs), np.std(qprecs),
    np.mean(qrecs), np.std(qrecs), np.mean(qf1s), np.std(qf1s),
)
print("Benchmark CSVs updated.")
