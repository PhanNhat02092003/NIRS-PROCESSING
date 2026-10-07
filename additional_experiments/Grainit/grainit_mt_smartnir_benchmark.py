"""Measures the classification/regression metrics the Grainit benchmark
CSVs track (accuracy/precision/recall/f1 and mse/mae/rmse/r2) for the CSSE
multi-task model (grainit_mt_smartnir_engine.py) and appends a "CSSE-MT"
row to each -- see mango_mt_smartnir_benchmark.py's docstring for why this
is a separate pass instead of reading it off the training history.
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
from dataset.grainit_multitask_dataset import GrainitMultiTaskDataset
from grainit_mt_smartnir_engine import needs_protein, quality_grade
from model.mt_smartnir_model import GrainitCSSEModel

device = "cuda" if torch.cuda.is_available() else "cpu"
k_folds = 5
machine = "Grainit"
task = "category_classification_mt_smartnir"

dataset_root = os.environ.get("GRAINIT_DATASET_ROOT", _ALL + "/Zenodo-Grainit")
full_path = f"{dataset_root}/{machine}/ALL.csv"
full_ds = GrainitMultiTaskDataset(full_path)

kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
splits = list(kf.split(np.arange(len(full_ds.cultivar_raw)), full_ds.cultivar_raw))

accs, precs, recs, f1s = [], [], [], []
moist_mse, moist_mae, moist_rmse, moist_r2 = [], [], [], []
prot_mse, prot_mae, prot_rmse, prot_r2 = [], [], [], []
qaccs, qprecs, qrecs, qf1s = [], [], [], []

for fold, (train_idx, val_idx) in enumerate(splits):
    full_ds.fit_normalization_and_labels(train_idx, save_dir=None)

    model = GrainitCSSEModel(
        n_food_classes=full_ds.n_classes, c_out=64, n_layers=6, n_heads=6,
        h_hidden=128, seq_len=full_ds.signal_length, use_kan=True, dropout=0.1,
    ).to(device)
    ckpt_path = f"checkpoint/{task}/{machine}/checkpoint_fold{fold + 1}.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    food_true, food_pred = [], []
    moist_true_n, moist_pred_n, moist_mask = [], [], []
    prot_true_n, prot_pred_n, prot_mask = [], [], []
    with torch.no_grad():
        for start in range(0, len(val_idx), 64):
            chunk = val_idx[start:start + 64]
            X_b = torch.tensor(full_ds.X[chunk], dtype=torch.float32, device=device)
            fl, mp, pp = model(X_b)
            food_pred.extend(fl.argmax(-1).cpu().numpy())
            food_true.extend(full_ds.food_y[chunk])
            moist_pred_n.extend(mp.cpu().numpy())
            moist_true_n.extend(full_ds.moisture_y[chunk])
            moist_mask.extend(full_ds.moisture_valid[chunk])
            prot_pred_n.extend(pp.cpu().numpy())
            prot_true_n.extend(full_ds.protein_y[chunk])
            prot_mask.extend(full_ds.protein_valid[chunk])

    food_true, food_pred = np.array(food_true), np.array(food_pred)
    accs.append((food_true == food_pred).mean() * 100.0)
    precs.append(precision_score(food_true, food_pred, average="macro", zero_division=0) * 100.0)
    recs.append(recall_score(food_true, food_pred, average="macro", zero_division=0) * 100.0)
    f1s.append(2 * precs[-1] * recs[-1] / (precs[-1] + recs[-1]) if precs[-1] + recs[-1] > 0 else 0.0)

    moist_mask = np.array(moist_mask) == 1
    moist_true_raw_all = full_ds.inverse_transform_moisture(np.array(moist_true_n))
    moist_pred_raw_all = full_ds.inverse_transform_moisture(np.array(moist_pred_n))
    mse = mean_squared_error(moist_true_raw_all[moist_mask], moist_pred_raw_all[moist_mask])
    moist_mse.append(mse)
    moist_mae.append(mean_absolute_error(moist_true_raw_all[moist_mask], moist_pred_raw_all[moist_mask]))
    moist_rmse.append(np.sqrt(mse))
    moist_r2.append(r2_score(moist_true_raw_all[moist_mask], moist_pred_raw_all[moist_mask]) * 100.0)

    prot_mask = np.array(prot_mask) == 1
    prot_true_raw_all = full_ds.inverse_transform_protein(np.array(prot_true_n))
    prot_pred_raw_all = full_ds.inverse_transform_protein(np.array(prot_pred_n))
    mse = mean_squared_error(prot_true_raw_all[prot_mask], prot_pred_raw_all[prot_mask])
    prot_mse.append(mse)
    prot_mae.append(mean_absolute_error(prot_true_raw_all[prot_mask], prot_pred_raw_all[prot_mask]))
    prot_rmse.append(np.sqrt(mse))
    prot_r2.append(r2_score(prot_true_raw_all[prot_mask], prot_pred_raw_all[prot_mask]) * 100.0)

    cultivar_true = full_ds.label_encoder.inverse_transform(food_true)
    cultivar_pred = full_ds.label_encoder.inverse_transform(food_pred)
    required_valid = moist_mask & (~needs_protein(cultivar_true) | prot_mask)
    true_grade = quality_grade(cultivar_true[required_valid], moist_true_raw_all[required_valid],
                                prot_true_raw_all[required_valid])
    pred_grade = quality_grade(cultivar_pred[required_valid], moist_pred_raw_all[required_valid],
                                prot_pred_raw_all[required_valid])
    qaccs.append((true_grade == pred_grade).mean() * 100.0)
    qprecs.append(precision_score(true_grade, pred_grade, average="macro", zero_division=0) * 100.0)
    qrecs.append(recall_score(true_grade, pred_grade, average="macro", zero_division=0) * 100.0)
    qf1s.append(2 * qprecs[-1] * qrecs[-1] / (qprecs[-1] + qrecs[-1]) if qprecs[-1] + qrecs[-1] > 0 else 0.0)

    print(f"Fold {fold + 1}/{k_folds}: acc={accs[-1]:.2f} prec={precs[-1]:.2f} "
          f"rec={recs[-1]:.2f} f1={f1s[-1]:.2f} | moist_r2={moist_r2[-1]:.2f} "
          f"prot_r2={prot_r2[-1]:.2f} | quality_acc={qaccs[-1]:.2f}")

append_classification_row(
    "benchmark/classification/Grainit.csv", "CSSE-MT",
    np.mean(accs), np.std(accs), np.mean(precs), np.std(precs),
    np.mean(recs), np.std(recs), np.mean(f1s), np.std(f1s),
)
append_regression_row(
    "benchmark/regression/Grainit.csv", "Moisture", "CSSE-MT",
    np.mean(moist_mse), np.std(moist_mse), np.mean(moist_mae), np.std(moist_mae),
    np.mean(moist_rmse), np.std(moist_rmse), np.mean(moist_r2), np.std(moist_r2),
)
append_regression_row(
    "benchmark/regression/Grainit.csv", "Protein", "CSSE-MT",
    np.mean(prot_mse), np.std(prot_mse), np.mean(prot_mae), np.std(prot_mae),
    np.mean(prot_rmse), np.std(prot_rmse), np.mean(prot_r2), np.std(prot_r2),
)
append_classification_row(
    "benchmark/quality/Grainit.csv", "CSSE-MT",
    np.mean(qaccs), np.std(qaccs), np.mean(qprecs), np.std(qprecs),
    np.mean(qrecs), np.std(qrecs), np.mean(qf1s), np.std(qf1s),
)
print("Benchmark CSVs updated.")
