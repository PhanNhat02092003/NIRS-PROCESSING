"""Evaluate the Buoc 1 (presence/absence) models of every baseline on the
held-out sample test (HOLDOUT_100.csv, 100 samples per machine) that no
training or cross-validation ever touched.

For each (machine, method, substance) the 5 fold models are applied to the
held-out spectra with exactly the preprocessing and fold-specific artefacts
used in training (scaler / normalisation stats, Platt calibrator, Recall>=0.9
threshold). Two views are reported:
  * per-fold: metrics of each fold model, then averaged over the 5 folds;
  * ensemble: calibrated probabilities averaged over the 5 fold models,
    decision threshold = mean of the 5 fold thresholds.

Held-out spectra get only the per-sample steps (Savitzky-Golay + SNV); the
train-set outlier filter is skipped because it is fitted on the whole set
(median/MAD across rows), which is meaningless on 100 rows.

Usage: python3 evaluate_holdout_stage1.py [FLAMENIR OCEANFX]
Writes results/holdout_stage1.json (merged across runs).
"""
import json
import os
import sys
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)

from dataset.preprocessing import savgol_smooth, snv

warnings.filterwarnings("ignore")
ROOT = os.environ.get("DATASET_ROOT", "../all-dataset/Danang-NIR")
SUBSTANCES = [
    'Thiamethoxam', 'Permethrin', 'Metalaxyl', 'Azoxystrobin',
    'Imidaclopird', 'Difenoconazole', 'Cypermethrin', 'Cyhalothrin',
    'Chlorantraniliprol', 'Chlopyrifos Methyl', 'Emamectin benzoate',
    'Chlorothalonil', 'Triadimefon', 'Cyantraniliprole', 'Flutolanil',
    'Indoxacarb', 'Abamectin', 'Propamocarb.HCL', 'Chlothianidin',
]
METHODS = {  # name -> (artefact tag, kind)
    "XGBoost": ("stage1", "xgb"),
    "LightGBM": ("stage1_lightgbm", "lgb"),
    "SMART-NIR": ("stage1_smartnir_os2", "smartnir"),
    "GuidedDCNet": ("stage1_guideddcnet", "gdc"),
}
SMARTNIR_BIN = {"FLAMENIR": 1, "OCEANFX": 8}  # wavelength binning used at training time


def load_holdout(machine):
    df = pd.read_csv(f"{ROOT}/{machine}/HOLDOUT_100.csv")
    w = [c for c in df.columns if c.startswith("w_")]
    X = snv(savgol_smooth(df[w].values.astype(np.float32))).astype(np.float32)
    food_ids = json.load(open(f"{ROOT}/food_ids.json"))
    n2i = {v["name"]: k for k, v in food_ids.items()}
    F = pd.get_dummies(df["category"].map(n2i)).reindex(
        columns=list(food_ids), fill_value=0).astype(np.float32).values
    return df, X, F


def bin_spectra(X, factor):
    if factor <= 1:
        return X
    n_out = (X.shape[1] // factor) // 8 * 8
    return X[:, :n_out * factor].reshape(len(X), n_out, factor).mean(axis=2).astype(np.float32)


def fold_probs(kind, machine, tag, sub, fold, X, F, dev="cuda"):
    """Calibrated probabilities of one fold model on the held-out rows, plus its threshold."""
    d = f"data/substance_regression/{tag}/{machine}/{sub}"
    ck = f"checkpoint/substance_regression/{tag}/{machine}/{sub}"
    calib = joblib.load(f"{d}/{sub}_fold_{fold}_calibrator.pkl")
    thr = json.load(open(f"{d}/{sub}_fold_{fold}_threshold.json"))["threshold"]
    if kind == "xgb":
        scaler = joblib.load(f"{d}/{sub}_fold_{fold}_scaler.pkl")
        b = xgb.Booster()
        b.load_model(f"{ck}/{sub}_fold_{fold}.json")
        raw = b.predict(xgb.DMatrix(scaler.transform(np.hstack([X, F]))))
        return calib.predict_proba(raw.reshape(-1, 1))[:, 1], thr
    if kind == "lgb":
        b = lgb.Booster(model_file=f"{ck}/{sub}_fold_{fold}.txt")
        raw = b.predict(np.hstack([X, F]))
        return calib.predict_proba(raw.reshape(-1, 1))[:, 1], thr
    nz = np.load(f"{d}/fold_{fold}_norm.npz")
    Xn = torch.tensor((X - nz["mean"]) / nz["std"], dtype=torch.float32, device=dev)
    Ft = torch.tensor(F, device=dev)
    if kind == "smartnir":
        from stage1_detection import SmartNIRWithFood, predict_logit_diff
        m = SmartNIRWithFood(X.shape[1], F.shape[1]).to(dev)
        m.load_state_dict(torch.load(f"{ck}/{sub}_fold_{fold}.pth", map_location=dev))
        idx = torch.arange(len(Xn), device=dev)
        score = predict_logit_diff(m, Xn, Ft, idx, 256)
    else:  # gdc
        from stage1_detection import GuidedDCNetFood, score_rows
        m = GuidedDCNetFood(F.shape[1]).to(dev)
        m.load_state_dict(torch.load(f"{ck}/{sub}_fold_{fold}.pth", map_location=dev))
        score = score_rows(m, Xn, Ft, torch.arange(len(Xn), device=dev), 256)
    return calib.predict_proba(score.reshape(-1, 1))[:, 1], thr


def metrics(y, p, thr):
    pred = (p > thr).astype(int)
    out = {
        "acc": accuracy_score(y, pred),
        "recall_pos": recall_score(y, pred, zero_division=0),
        "precision_pos": precision_score(y, pred, zero_division=0),
        "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
        "n_pos": int(y.sum()), "n_pred_pos": int(pred.sum()),
    }
    if 0 < y.sum() < len(y):
        out["pr_auc"] = average_precision_score(y, p)
        out["roc_auc"] = roc_auc_score(y, p)
    return out


def evaluate(machine):
    df, X0, F = load_holdout(machine)
    results = {}
    for method, (tag, kind) in METHODS.items():
        X = bin_spectra(X0, SMARTNIR_BIN[machine]) if kind == "smartnir" else X0
        for sub in SUBSTANCES:
            ck = f"checkpoint/substance_regression/{tag}/{machine}/{sub}"
            ext = "json" if kind == "xgb" else "txt" if kind == "lgb" else "pth"
            files = [f"{ck}/{sub}_fold_{k}.{ext}" for k in range(1, 6)]
            if not all(os.path.exists(f) for f in files):
                continue
            y = (df[sub] > 0).astype(int).values
            probs, thrs = zip(*[fold_probs(kind, machine, tag, sub, k, X, F) for k in range(1, 6)])
            per_fold = [metrics(y, p, t) for p, t in zip(probs, thrs)]
            ens = metrics(y, np.mean(probs, axis=0), float(np.mean(thrs)))
            avg = {k: float(np.mean([m[k] for m in per_fold if k in m])) for k in per_fold[0] if k in per_fold[0]}
            results[f"{machine}|{method}|{sub}"] = {"per_fold_mean": avg, "ensemble": ens}
            print(f"{machine:8s} {method:12s} {sub:20s} n_pos={ens['n_pos']:3d} "
                  f"ens acc={ens['acc']:.2f} rec={ens['recall_pos']:.2f} prec={ens['precision_pos']:.2f} | "
                  f"fold-mean acc={avg['acc']:.2f}", flush=True)
    return results


if __name__ == "__main__":
    out_path = "results/holdout_stage1.json"
    os.makedirs("results", exist_ok=True)
    allres = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for m in (sys.argv[1:] or ["FLAMENIR", "OCEANFX"]):
        allres.update(evaluate(m))
        json.dump(allres, open(out_path, "w"), indent=1)
