"""Buoc 2 (An toan / Vuot nguong) retrain, SMART-NIR + food one-hot, with the
same batch-level-leakage fix already validated for food_classification and
Buoc 1 (see stage1_detection_downsampled.py): the deployed model
(checkpoint/substance_severity_smartnir) was trained with plain
StratifiedKFold on {machine}/ALL.csv, so one prepared batch (same category +
same 19-substance concentration vector, scanned repeatedly) can straddle
train/val. Fixed here by training on {machine}/ALL_downsampled.csv
(MiniBatchKMeans representative subset, see downsample_training_data.py)
with StratifiedGroupKFold on its `_group_id` column, plus the same per-fold
`assert not overlap` sanity check.

No near-LOD-style concentration filter here (unlike Buoc 1): the severity
label is MRL-relative, not LOD-relative, and stage2_safety.py's existing
food_prior_shrinkage already handles the near-deterministic (food,
pesticide) cells that near-LOD filtering was substituting for in Buoc 1.
Everything else -- architecture, calibration, food_prior_shrinkage,
Recall>=0.9 threshold, POS_FRAC positive-oversampling, BIN for OCEANFX -- is
unchanged from stage2_safety.py's main_smartnir, so this is a fair
apples-to-apples retrain, not a different method. Reuses train_one_fold /
pick_threshold_for_recall from stage1_detection_downsampled.py (generic
binary-classification training loop, no presence/absence-specific logic)
and food_prior_shrinkage from model/smartnir_food_model.py (shared with
deploy_utils.py's production inference).

Usage:
    MACHINE=FLAMENIR SUBSTANCES=Thiamethoxam python3 stage2_safety_downsampled.py
    MACHINE=OCEANFX BIN=8 python3 stage2_safety_downsampled.py   # all 19 substances

Env: MACHINE (default FLAMENIR), SUBSTANCES (comma list; default = all 19),
MAX_EPOCHS, BATCH_SIZE, POS_FRAC, CAL_FRAC, FOLDS, TAG (output subfolder,
default stage2_smartnir_downsampled), BIN (wavelength binning factor).
"""

import json
import os

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                              f1_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from dataset.preprocessing import preprocess_spectra
from model.smartnir_food_model import food_prior_shrinkage, predict_logit_diff
from stage1_detection_downsampled import ALL_SUBSTANCES, pick_threshold_for_recall, train_one_fold

load_dotenv()


def _food_prior(y_tr, food_idx_tr, n_foods):
    prior_by_food = np.full(max(n_foods, 1), 0.5, dtype=np.float64)
    for fidx in range(n_foods):
        mask = food_idx_tr == fidx
        n_food = int(mask.sum())
        n_pos_food = int(y_tr[mask].sum()) if n_food else 0
        prior_by_food[fidx] = (n_pos_food + 1) / (n_food + 2)  # Beta(1,1)
    return prior_by_food


def run_substance(substance, Xnp, Fnp, y, groups, machine, device, max_epochs, bs,
                   pos_frac, cal_frac, folds, tag):
    hist_dir = f"history/substance_severity_smartnir/{tag}/{machine}/{substance}"
    ckpt_dir = f"checkpoint/substance_severity_smartnir/{tag}/{machine}/{substance}"
    data_dir = f"data/substance_severity_smartnir/{tag}/{machine}/{substance}"
    for d in (hist_dir, ckpt_dir, data_dir):
        os.makedirs(d, exist_ok=True)

    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos < 5 or n_neg < 5:
        print(f"Not enough samples for {substance} (An toan={n_neg}, Vuot nguong={n_pos}), skipping.")
        return

    food_idx_all = Fnp.argmax(axis=1)
    n_foods = Fnp.shape[1]
    kf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    Fs = torch.tensor(Fnp, device=device)
    ys = torch.tensor(y, dtype=torch.long, device=device)
    for fold, (tr, va) in enumerate(kf.split(Xnp, y, groups=groups), start=1):
        if folds and fold not in folds:
            continue
        overlap = set(groups[tr]) & set(groups[va])
        assert not overlap, f"{substance} fold {fold}: group leakage {len(overlap)} groups"
        print(f"{substance} fold {fold}/5 (train={len(tr)}, val={len(va)})", flush=True)
        mean = Xnp[tr].mean(axis=0, keepdims=True)
        std = Xnp[tr].std(axis=0, keepdims=True) + 1e-8
        Xs = torch.tensor((Xnp - mean) / std, dtype=torch.float32, device=device)
        np.savez(f"{data_dir}/fold_{fold}_norm.npz", mean=mean, std=std)

        if cal_frac > 0:
            tr_fit, tr_cal = train_test_split(tr, test_size=cal_frac, stratify=y[tr], random_state=fold)
        else:
            tr_fit, tr_cal = tr, tr

        model = train_one_fold(Xs, Fs, ys, tr_fit, tr_cal, va, device, max_epochs, bs, pos_frac=pos_frac)

        prior_by_food = _food_prior(y[tr_cal], food_idx_all[tr_cal], n_foods)

        d_tr = predict_logit_diff(model, Xs, Fs, torch.as_tensor(tr_cal, device=device), bs * 2)
        calib = LogisticRegression().fit(d_tr.reshape(-1, 1), y[tr_cal])
        p_tr = food_prior_shrinkage(calib.predict_proba(d_tr.reshape(-1, 1))[:, 1], food_idx_all[tr_cal], prior_by_food)
        thr = pick_threshold_for_recall(y[tr_cal], p_tr, target_recall=0.9)

        d_va = predict_logit_diff(model, Xs, Fs, torch.as_tensor(va, device=device), bs * 2)
        p_raw = 1.0 / (1.0 + np.exp(-d_va))
        p_va_cal = calib.predict_proba(d_va.reshape(-1, 1))[:, 1]
        p_va = food_prior_shrinkage(p_va_cal, food_idx_all[va], prior_by_food)
        yv = y[va]
        pred = (p_va > thr).astype(int)
        hist = {
            "threshold": [thr],
            "val_acc": [accuracy_score(yv, pred)],
            "val_precision": [precision_score(yv, pred, average="macro", zero_division=0)],
            "val_recall": [recall_score(yv, pred, average="macro", zero_division=0)],
            "val_f1": [f1_score(yv, pred, average="macro", zero_division=0)],
            "val_pr_auc": [average_precision_score(yv, p_va)],
            "brier_before_calibration": [brier_score_loss(yv, p_raw)],
            "brier_after_calibration": [brier_score_loss(yv, p_va_cal)],
            "brier_after_shrinkage": [brier_score_loss(yv, p_va)],
            "val_pos_recall": [recall_score(yv, pred, zero_division=0)],
            "val_pos_precision": [precision_score(yv, pred, zero_division=0)],
        }
        with open(f"{hist_dir}/{substance}_fold_{fold}.json", "w") as f:
            json.dump(hist, f, indent=4)
        torch.save(model.state_dict(), f"{ckpt_dir}/{substance}_fold_{fold}.pth")
        import joblib
        joblib.dump(calib, f"{data_dir}/{substance}_fold_{fold}_calibrator.pkl")
        with open(f"{data_dir}/{substance}_fold_{fold}_threshold.json", "w") as f:
            json.dump({"threshold": thr, "target_recall": 0.9}, f, indent=4)
        with open(f"{data_dir}/{substance}_fold_{fold}_food_prior.json", "w") as f:
            json.dump({"prior_by_food": prior_by_food.tolist()}, f, indent=4)
        print(f"  -> acc={hist['val_acc'][0]:.4f} PR-AUC={hist['val_pr_auc'][0]:.4f} "
              f"pos_recall={hist['val_pos_recall'][0]:.3f} pos_prec={hist['val_pos_precision'][0]:.3f} "
              f"brier_shrink={hist['brier_after_shrinkage'][0]:.4f}", flush=True)
        del Xs
        torch.cuda.empty_cache()


def main():
    machine = os.environ.get("MACHINE", "FLAMENIR")
    max_epochs = int(os.environ.get("MAX_EPOCHS", 50))
    bs = int(os.environ.get("BATCH_SIZE", 2048))
    subs = os.environ.get("SUBSTANCES")
    substances = [s.strip() for s in subs.split(",")] if subs else ALL_SUBSTANCES
    device = "cuda"
    pos_frac = float(os.environ.get("POS_FRAC", 0.25))
    cal_frac = float(os.environ.get("CAL_FRAC", 0.15))
    folds = [int(f) for f in os.environ["FOLDS"].split(",")] if os.environ.get("FOLDS") else None
    tag = os.environ.get("TAG", "stage2_smartnir_downsampled")

    root = os.environ["DATASET_ROOT"]
    df = pd.read_csv(f"{root}/{machine}/ALL_downsampled.csv")
    w_cols = sorted((c for c in df.columns if c.startswith("w_")), key=lambda c: int(c.split("_")[1]))
    X, keep = preprocess_spectra(df[w_cols].values.astype(np.float32))
    df = df[keep].reset_index(drop=True)

    bin_factor = int(os.environ.get("BIN", 8 if machine == "OCEANFX" else 1))
    if bin_factor > 1:
        n_out = (X.shape[1] // bin_factor) // 8 * 8
        X = X[:, :n_out * bin_factor].reshape(len(X), n_out, bin_factor).mean(axis=2).astype(np.float32)
        print(f"binned spectra by {bin_factor}: {len(w_cols)} -> {X.shape[1]} points", flush=True)

    with open(f"{root}/food_ids.json") as f:
        food_ids = json.load(f)
    with open(f"{root}/pesticide_ids.json") as f:
        pesticide_ids = json.load(f)
    with open(f"{root}/thresholds.json") as f:
        thresholds = json.load(f)
    name2id = {v["name"]: k for k, v in food_ids.items()}
    pesticide_name_to_id = {v["name"]: k for k, v in pesticide_ids.items()}
    food_id_per_row = df["category"].map(name2id)
    Fnp = pd.get_dummies(food_id_per_row).reindex(columns=list(food_ids), fill_value=0).astype(np.float32).values
    groups = df["_group_id"].values
    print(f"{machine}: {len(df)} rows, {X.shape[1]} wavelengths, {len(set(groups))} groups, "
          f"substances={substances}", flush=True)

    for s in substances:
        pid = pesticide_name_to_id[s]
        mrl_per_row = food_id_per_row.map(lambda fid: thresholds[fid][pid]["mrl"])
        y = ((df[s] != -1) & (df[s] > mrl_per_row)).astype(int).values
        run_substance(s, X, Fnp, y, groups, machine, device, max_epochs, bs, pos_frac, cal_frac, folds, tag)


if __name__ == "__main__":
    main()
