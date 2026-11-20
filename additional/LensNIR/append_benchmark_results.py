"""Appends a "LensNIR" row to additional/{Dataset}/benchmark/{classification,
regression,quality}/{Dataset}.csv for Grainit/Mango/Rapeseed, from the
already-trained "main" (fixed S=4, OURS_ARGS) LensNIR checkpoints' raw
per-fold test predictions at ../../../LensNIR/results/main/{Dataset}/
fold{1..5}_test_pred.npz -- READ ONLY, nothing in ../../../LensNIR is
touched or retrained.

Also supports DanangLeafy6_FLAMENIR/DanangLeafy6_OCEANFX, whose "main"
checkpoints were trained via this repo's own additional/LensNIR/train.py
(not the separate paper-code LensNIR checkout) -- predictions read from
./results/main/{dataset}/ instead, and the row goes to
additional/DanangLeafy6/benchmark/{task}/{FLAMENIR,OCEANFX}.csv (one CSV
per machine, not per dataset, matching that project's convention).

Usage: python3 append_benchmark_results.py [dataset ...]  (default: Grainit Mango Rapeseed)
       python3 append_benchmark_results.py DanangLeafy6_FLAMENIR DanangLeafy6_OCEANFX

LensNIR's own paper/results/{Dataset}_summary.csv already reports food_acc/
food_f1/r2/rmse/mae/qual_acc/qual_f1/qual_bacc for this exact "main" row,
but not precision/recall (not part of that paper's tables); this script
recomputes accuracy/precision_macro/recall_macro/f1_macro itself from the
raw food_true/food_pred (and the same for the derived quality decision) to
match this repo's own benchmark CSV schema exactly, and recomputes
mse/mae/rmse/r2 per regression target the same way -- cross-checked against
the paper's own numbers below before appending.
"""
import os
import sys

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                              mean_absolute_error, mean_squared_error, precision_score,
                              r2_score, recall_score)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # additional/, for benchmark_utils
from benchmark_utils import append_classification_row, append_regression_row

from dataset_benchmark import DATASETS, derive_quality, load_benchmark

LENSNIR_RESULTS = "/home/phannhat/LensNIR/results/main"
# DanangLeafy6_{machine}'s "main" checkpoints were trained via this repo's
# own additional/LensNIR/train.py (not the separate paper-code LensNIR
# checkout Grainit/Mango/Rapeseed use), so their fold predictions live here.
LOCAL_RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/main")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # NIRS-PROCESSING/
N_FOLDS = 5


def results_dir_for(dataset: str) -> str:
    return LOCAL_RESULTS if dataset.startswith("DanangLeafy6_") else LENSNIR_RESULTS


def csv_group_for(dataset: str) -> tuple:
    """(folder under additional/, csv basename) for additional/{folder}/benchmark/*/{basename}.csv.

    Grainit/Mango/Rapeseed have one CSV per dataset. DanangLeafy6 splits by
    machine (FLAMENIR/OCEANFX) instead -- see additional/DanangLeafy6/benchmark/."""
    if dataset.startswith("DanangLeafy6_"):
        return "DanangLeafy6", dataset[len("DanangLeafy6_"):]
    return dataset, dataset


def load_fold_preds(dataset: str):
    preds = []
    results_root = results_dir_for(dataset)
    for fold in range(1, N_FOLDS + 1):
        path = f"{results_root}/{dataset}/fold{fold}_test_pred.npz"
        with np.load(path) as z:
            preds.append({k: z[k] for k in z.files})
    return preds


def clf_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred) * 100,
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0) * 100,
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0) * 100,
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0) * 100,
    }


def reg_metrics(y_true, y_pred):
    valid = ~np.isnan(y_true)
    yt, yp = y_true[valid], y_pred[valid]
    mse = mean_squared_error(yt, yp)
    return {
        "mse": mse, "mae": mean_absolute_error(yt, yp),
        "rmse": np.sqrt(mse), "r2": r2_score(yt, yp) * 100,
    }


def mean_std(dicts, key):
    vals = np.array([d[key] for d in dicts])
    return vals.mean(), vals.std(ddof=1)


def run(dataset: str):
    print(f"\n=== {dataset} ===")
    d = load_benchmark(dataset)
    food_classes = d["food_classes"]
    target_names = d["target_names"]
    spec = DATASETS[dataset]
    preds = load_fold_preds(dataset)
    folder, csv_name = csv_group_for(dataset)

    # --- classification ---
    clf_folds = [clf_metrics(p["food_true"], p["food_pred"]) for p in preds]
    acc_m, acc_s = mean_std(clf_folds, "accuracy")
    prec_m, prec_s = mean_std(clf_folds, "precision")
    rec_m, rec_s = mean_std(clf_folds, "recall")
    f1_m, f1_s = mean_std(clf_folds, "f1")
    print(f"  classification: acc={acc_m:.2f}+-{acc_s:.2f} prec={prec_m:.2f} rec={rec_m:.2f} f1={f1_m:.2f}")
    csv_path = f"{REPO_ROOT}/additional/{folder}/benchmark/classification/{csv_name}.csv"
    append_classification_row(csv_path, "LensNIR", acc_m, acc_s, prec_m, prec_s, rec_m, rec_s, f1_m, f1_s)

    # --- regression, per target ---
    for i, target in enumerate(target_names):
        reg_folds = [reg_metrics(p["reg_true"][:, i], p["reg_pred"][:, i]) for p in preds]
        mse_m, mse_s = mean_std(reg_folds, "mse")
        mae_m, mae_s = mean_std(reg_folds, "mae")
        rmse_m, rmse_s = mean_std(reg_folds, "rmse")
        r2_m, r2_s = mean_std(reg_folds, "r2")
        print(f"  regression[{target}]: r2={r2_m:.2f}+-{r2_s:.2f} rmse={rmse_m:.4f} mae={mae_m:.4f}")
        csv_path = f"{REPO_ROOT}/additional/{folder}/benchmark/regression/{csv_name}.csv"
        append_regression_row(csv_path, target, "LensNIR", mse_m, mse_s, mae_m, mae_s, rmse_m, rmse_s, r2_m, r2_s)

    # --- quality ---
    qual_folds = []
    for p in preds:
        if spec.get("quality_col"):  # Rapeseed: learned head, aux_true/aux_pred already 0/1
            y_true_q, y_pred_q = p["aux_true"], p["aux_pred"]
        else:  # Grainit/Mango: rule-derived from predicted class + predicted regression
            y_true_q = derive_quality(p["reg_true"], p["food_true"], food_classes, target_names, spec["quality_rules"])
            y_pred_q = derive_quality(p["reg_pred"], p["food_pred"], food_classes, target_names, spec["quality_rules"])
        valid = ~(np.isnan(y_true_q) | np.isnan(y_pred_q))
        qual_folds.append(clf_metrics(y_true_q[valid], y_pred_q[valid]))
    qacc_m, qacc_s = mean_std(qual_folds, "accuracy")
    qprec_m, qprec_s = mean_std(qual_folds, "precision")
    qrec_m, qrec_s = mean_std(qual_folds, "recall")
    qf1_m, qf1_s = mean_std(qual_folds, "f1")
    print(f"  quality: acc={qacc_m:.2f}+-{qacc_s:.2f} prec={qprec_m:.2f} rec={qrec_m:.2f} f1={qf1_m:.2f}")
    csv_path = f"{REPO_ROOT}/additional/{folder}/benchmark/quality/{csv_name}.csv"
    append_classification_row(csv_path, "LensNIR", qacc_m, qacc_s, qprec_m, qprec_s, qrec_m, qrec_s, qf1_m, qf1_s)


if __name__ == "__main__":
    datasets = sys.argv[1:] or ["Grainit", "Mango", "Rapeseed"]
    for ds in datasets:
        run(ds)
