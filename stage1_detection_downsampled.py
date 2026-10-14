"""Buoc 1 (presence/absence) retrain, SMART-NIR + food one-hot, with the two
data fixes agreed after analyze_substance_detectability_v2.py:

  1. Batch-level leakage: the deployed model (checkpoint/substance_regression/
     stage1_smartnir_os2) was trained with plain StratifiedKFold on
     {machine}/ALL.csv, so one prepared batch (same category + same 19-substance
     concentration vector, scanned repeatedly) can straddle train/val -- same
     leakage diagnosed and fixed for food_classification_downsampled.py. Fixed
     here by training on {machine}/ALL_downsampled.csv (MiniBatchKMeans
     representative subset, see downsample_training_data.py) with
     StratifiedGroupKFold on its `_group_id` column, plus the same per-fold
     `assert not overlap` sanity check.
  2. Near-LOD label noise: a "present" row whose concentration is barely above
     the absent sentinel carries almost no spectral signal (confirmed by
     analyze_substance_detectability.py/_v2.py -- pooled present-vs-absent
     separability ratio 0.05-0.64, but restricting to the top 25% by
     concentration often pushes it past 1.0). Training on every present row
     indiscriminately asks the model to fit noise for the bottom of that
     range. Fixed by dropping the bottom `LOW_CONC_QUANTILE` of each
     substance's present concentrations from the gradient-update pool
     (`tr_fit`) only -- the calibration slice (`tr_cal`) and the validation
     fold keep the FULL concentration range, so reported recall/precision
     stay honest about real-world performance across all concentrations,
     including the ones the model was deliberately not trained to chase.

Everything else (architecture, calibration, Recall>=0.9 threshold, POS_FRAC
positive-oversampling, BIN_FACTOR for OCEANFX, output metric names) is
unchanged from stage1_detection.py's main_smartnir, so this is a fair
apples-to-apples retrain, not a different method.

Usage:
    MACHINE=FLAMENIR SUBSTANCES=Thiamethoxam python3 stage1_detection_downsampled.py
    MACHINE=OCEANFX BIN=8 python3 stage1_detection_downsampled.py   # all 19 substances

Env: MACHINE (default FLAMENIR), SUBSTANCES (comma list; default = all 19),
MAX_EPOCHS, BATCH_SIZE, POS_FRAC, CAL_FRAC, FOLDS, TAG (output subfolder,
default stage1_smartnir_downsampled), BIN (wavelength binning factor),
LOW_CONC_QUANTILE (default 0.25; 0 disables the near-LOD filter).
"""

import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                              f1_score, precision_recall_curve, precision_score, recall_score)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from dataset.preprocessing import preprocess_spectra
from model.smartnir_food_model import SmartNIRWithFood, predict_logit_diff

load_dotenv()

ALL_SUBSTANCES = [
    'Thiamethoxam', 'Permethrin', 'Metalaxyl', 'Azoxystrobin',
    'Imidaclopird', 'Difenoconazole', 'Cypermethrin', 'Cyhalothrin',
    'Chlorantraniliprol', 'Chlopyrifos Methyl', 'Emamectin benzoate',
    'Chlorothalonil', 'Triadimefon', 'Cyantraniliprole', 'Flutolanil',
    'Indoxacarb', 'Abamectin', 'Propamocarb.HCL', 'Chlothianidin',
]


def pick_threshold_for_recall(y_true, y_prob, target_recall=0.9):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    precisions, recalls = precisions[:-1], recalls[:-1]
    eligible = np.where(recalls >= target_recall)[0]
    if len(eligible) == 0:
        return float(thresholds[np.argmax(recalls)])
    best = eligible[np.argmax(precisions[eligible])]
    return float(thresholds[best])


def filter_low_conc_positives(idx: np.ndarray, conc: np.ndarray, quantile: float) -> np.ndarray:
    """Drops the bottom `quantile` of present-row concentrations from `idx`
    (negatives -- conc<=0, i.e. the -1 absent sentinel -- are always kept).
    Threshold computed from `idx`'s own positives only, so it never looks
    outside the caller's fold/split."""
    if quantile <= 0:
        return idx
    conc_idx = conc[idx]
    pos_mask = conc_idx > 0
    pos = idx[pos_mask]
    neg = idx[~pos_mask]
    if len(pos) < 20:
        return idx
    thr = np.quantile(conc[pos], quantile)
    keep_pos = pos[conc[pos] >= thr]
    return np.concatenate([neg, keep_pos])


def train_one_fold(Xs, Fs, ys, tr_fit, tr_cal, va_idx, device, max_epochs, bs,
                    patience=8, min_epochs=10, lr=2e-3, pos_frac=0.0):
    model = SmartNIRWithFood(Xs.shape[1], Fs.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps_per_epoch = (len(tr_fit) + bs - 1) // bs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=max_epochs * steps_per_epoch, pct_start=0.1)
    n_pos = int(ys[tr_fit].sum().item())
    class_w = torch.tensor([1.0, ((len(tr_fit) - n_pos) / max(n_pos, 1)) ** 0.5], device=device)
    tr_fit_t = torch.as_tensor(tr_fit, device=device)
    if pos_frac > 0:
        pos_pool = tr_fit_t[ys[tr_fit_t] == 1]
        neg_pool = tr_fit_t[ys[tr_fit_t] == 0]
        n_pos_b = max(1, int(bs * pos_frac))
        class_w = None
    va_idx_t = torch.as_tensor(va_idx, device=device)

    best, best_state, bad = -1.0, None, 0
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        model.train()
        perm = tr_fit_t[torch.randperm(len(tr_fit_t), device=device)]
        run = 0.0
        for i in range(0, len(perm), bs):
            if pos_frac > 0:
                b = torch.cat([
                    pos_pool[torch.randint(len(pos_pool), (n_pos_b,), device=device)],
                    neg_pool[torch.randint(len(neg_pool), (bs - n_pos_b,), device=device)]])
            else:
                b = perm[i:i + bs]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(Xs[b], Fs[b])
            loss = F.cross_entropy(lg.float(), ys[b], weight=class_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += loss.item() * len(b)
        d = predict_logit_diff(model, Xs, Fs, va_idx_t, bs * 2)
        p = 1.0 / (1.0 + np.exp(-d))
        yv = ys[va_idx_t].cpu().numpy()
        val_loss = float(-np.mean(yv * np.log(p + 1e-7) + (1 - yv) * np.log(1 - p + 1e-7)))
        val_ap = average_precision_score(yv, p)
        print(f"  epoch {epoch:2d} train_loss={run / len(perm):.4f} val_logloss={val_loss:.4f} "
              f"val_PR-AUC={val_ap:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if val_ap > best + 1e-4:
            best, bad = val_ap, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch > min_epochs:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


def run_substance(substance, Xnp, Fnp, y, conc, groups, machine, device, max_epochs, bs,
                   pos_frac, cal_frac, low_conc_quantile, folds, tag):
    hist_dir = f"history/substance_regression/{tag}/{machine}/{substance}"
    ckpt_dir = f"checkpoint/substance_regression/{tag}/{machine}/{substance}"
    data_dir = f"data/substance_regression/{tag}/{machine}/{substance}"
    for d in (hist_dir, ckpt_dir, data_dir):
        os.makedirs(d, exist_ok=True)

    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos < 5 or n_neg < 5:
        print(f"Not enough samples for {substance} (pos={n_pos}, neg={n_neg}), skipping.")
        return

    kf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    Fs = torch.tensor(Fnp, device=device)
    ys = torch.tensor(y, dtype=torch.long, device=device)
    for fold, (tr, va) in enumerate(kf.split(Xnp, y, groups=groups), start=1):
        if folds and fold not in folds:
            continue
        overlap = set(groups[tr]) & set(groups[va])
        assert not overlap, f"{substance} fold {fold}: group leakage {len(overlap)} groups"
        print(f"{substance} fold {fold}/5 (train={len(tr)}, val={len(va)})", flush=True)
        t0 = time.time()
        mean = Xnp[tr].mean(axis=0, keepdims=True)
        std = Xnp[tr].std(axis=0, keepdims=True) + 1e-8
        Xs = torch.tensor((Xnp - mean) / std, dtype=torch.float32, device=device)
        np.savez(f"{data_dir}/fold_{fold}_norm.npz", mean=mean, std=std)

        if cal_frac > 0:
            tr_fit, tr_cal = train_test_split(tr, test_size=cal_frac, stratify=y[tr], random_state=fold)
        else:
            tr_fit, tr_cal = tr, tr
        n_before = len(tr_fit)
        tr_fit = filter_low_conc_positives(tr_fit, conc, low_conc_quantile)
        print(f"  low-conc filter: tr_fit {n_before} -> {len(tr_fit)} rows "
              f"(quantile={low_conc_quantile})", flush=True)

        model = train_one_fold(Xs, Fs, ys, tr_fit, tr_cal, va, device, max_epochs, bs, pos_frac=pos_frac)

        d_tr = predict_logit_diff(model, Xs, Fs, torch.as_tensor(tr_cal, device=device), bs * 2)
        calib = LogisticRegression().fit(d_tr.reshape(-1, 1), y[tr_cal])
        p_tr = calib.predict_proba(d_tr.reshape(-1, 1))[:, 1]
        thr = pick_threshold_for_recall(y[tr_cal], p_tr, target_recall=0.9)

        d_va = predict_logit_diff(model, Xs, Fs, torch.as_tensor(va, device=device), bs * 2)
        p_raw = 1.0 / (1.0 + np.exp(-d_va))
        p_va = calib.predict_proba(d_va.reshape(-1, 1))[:, 1]
        yv = y[va]
        pred = (p_va > thr).astype(int)
        grid = np.linspace(0.05, 0.95, 91)
        accs = [accuracy_score(yv, p_va > t) for t in grid]
        hist = {
            "threshold": [thr],
            "val_acc": [accuracy_score(yv, pred)],
            "val_precision": [precision_score(yv, pred, average="macro", zero_division=0)],
            "val_recall": [recall_score(yv, pred, average="macro", zero_division=0)],
            "val_f1": [f1_score(yv, pred, average="macro", zero_division=0)],
            "val_pr_auc": [average_precision_score(yv, p_va)],
            "brier_before_calibration": [brier_score_loss(yv, p_raw)],
            "brier_after_calibration": [brier_score_loss(yv, p_va)],
            "val_pos_recall": [recall_score(yv, pred, zero_division=0)],
            "val_pos_precision": [precision_score(yv, pred, zero_division=0)],
            "val_acc_best_threshold": [float(max(accs))],
            "seconds": [time.time() - t0],
        }
        with open(f"{hist_dir}/{substance}_fold_{fold}.json", "w") as f:
            json.dump(hist, f, indent=4)
        torch.save(model.state_dict(), f"{ckpt_dir}/{substance}_fold_{fold}.pth")
        import joblib
        joblib.dump(calib, f"{data_dir}/{substance}_fold_{fold}_calibrator.pkl")
        with open(f"{data_dir}/{substance}_fold_{fold}_threshold.json", "w") as f:
            json.dump({"threshold": thr, "target_recall": 0.9}, f, indent=4)
        print(f"  -> acc={hist['val_acc'][0]:.4f} acc@bestthr={hist['val_acc_best_threshold'][0]:.4f} "
              f"PR-AUC={hist['val_pr_auc'][0]:.4f} pos_recall={hist['val_pos_recall'][0]:.3f} "
              f"pos_prec={hist['val_pos_precision'][0]:.3f} ({hist['seconds'][0]:.0f}s)", flush=True)
        del Xs
        torch.cuda.empty_cache()


def main():
    machine = os.environ.get("MACHINE", "FLAMENIR")
    max_epochs = int(os.environ.get("MAX_EPOCHS", 40))
    bs = int(os.environ.get("BATCH_SIZE", 2048))
    subs = os.environ.get("SUBSTANCES")
    substances = [s.strip() for s in subs.split(",")] if subs else ALL_SUBSTANCES
    device = "cuda"
    pos_frac = float(os.environ.get("POS_FRAC", 0.25))
    cal_frac = float(os.environ.get("CAL_FRAC", 0.15))
    low_conc_quantile = float(os.environ.get("LOW_CONC_QUANTILE", 0.25))
    folds = [int(f) for f in os.environ["FOLDS"].split(",")] if os.environ.get("FOLDS") else None
    tag = os.environ.get("TAG", "stage1_smartnir_downsampled")

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
    name2id = {v["name"]: k for k, v in food_ids.items()}
    Fnp = pd.get_dummies(df["category"].map(name2id)).reindex(
        columns=list(food_ids), fill_value=0).astype(np.float32).values
    groups = df["_group_id"].values
    print(f"{machine}: {len(df)} rows, {X.shape[1]} wavelengths, {len(set(groups))} groups, "
          f"substances={substances}, low_conc_quantile={low_conc_quantile}", flush=True)

    for s in substances:
        y = (df[s].values > 0).astype(int)
        conc = df[s].values.astype(np.float32)
        run_substance(s, X, Fnp, y, conc, groups, machine, device, max_epochs, bs,
                      pos_frac, cal_frac, low_conc_quantile, folds, tag)


if __name__ == "__main__":
    main()
