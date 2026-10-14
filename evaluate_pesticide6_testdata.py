"""Evaluates the already-trained MT-SMARTNIR pesticide-6 checkpoints
(multitask_mt_smartnir_engine.py) on a genuine held-out external test set
(DATASET_ROOT/test-data/{Machine}.csv, added 2026-10-13) instead of a CV
validation split.

The test file uses a different schema than the training ALL.csv files:
  - pesticide columns are named "comp_<Substance>" instead of "<Substance>"
  - 0.0 means absent instead of -1 (harmless here: this script's presence
    check is "> 0" either way, same as dataset/pesticide_multitask_dataset.py)
  - category is an abbreviated code (e.g. "CACHUA") instead of the training
    set's full Vietnamese name ("Cà Chua") -- CATEGORY_MAP below was built
    by directly comparing both category columns: 9 codes, 9 training
    categories, an unambiguous 1:1 phonetic match.

Each fold's saved per-fold mean/std (data/multitask_mt_smartnir/{machine}/fold_i/stats.npz)
and label encoder (.../label_encoder.pkl) are reused as-is, NOT re-fit on
this test set -- re-fitting would leak the test set's own distribution
into normalization and scramble the class indices the checkpoint was
actually trained against.
"""

import os

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                              mean_squared_error, precision_score, r2_score,
                              recall_score)

from dataset.preprocessing import preprocess_spectra
from model.mt_smartnir_model import MultiTaskSMARTNIRModel
from xspecmamba_joint_pesticide_engine import DEFAULT_SUBSTANCES

device = "cuda" if torch.cuda.is_available() else "cpu"
substances = DEFAULT_SUBSTANCES
k_folds = 5

CATEGORY_MAP = {
    "CACHUA": "Cà Chua",
    "CARROT": "Cà Rốt",
    "CAIBEX": "Cải Bẹ Xanh",
    "CAITHI": "Cải Thìa",
    "DUALEO": "Dưa Leo",
    "KHOQUA": "Khổ Qua",
    "MONTOI": "Mồng Tơi",
    "XALACH": "Xà Lách",
    "DAUCOV": "Đậu Cove",
}

MACHINE_CFG = {
    "FLAMENIR": {"bin_factor": 1},
    "OCEANFX": {"bin_factor": 8},
}


def load_test_set(machine: str, bin_factor: int):
    path = f"{os.environ['DATASET_ROOT']}/test-data/{machine}.csv"
    df = pd.read_csv(path)

    w_cols = [c for c in df.columns if c.startswith("w_")]
    X_raw = df[w_cols].values.astype(np.float32)

    category_raw = df["category"].map(CATEGORY_MAP).values
    unmapped = pd.isna(category_raw)
    if unmapped.any():
        raise ValueError(f"Unmapped category codes in {machine} test set: "
                          f"{sorted(set(df['category'][unmapped]))}")

    raw_pest = np.stack([df[f"comp_{s}"].values.astype(np.float32) for s in substances], axis=1)

    X_raw, keep_mask = preprocess_spectra(X_raw)
    n_dropped = (~keep_mask).sum()
    if n_dropped:
        print(f"[{machine} test-data] dropped {n_dropped} outlier spectra out of {len(keep_mask)}")

    if bin_factor > 1:
        n_out = (X_raw.shape[1] // bin_factor) // 8 * 8
        X_raw = X_raw[:, :n_out * bin_factor].reshape(len(X_raw), n_out, bin_factor).mean(axis=2).astype(np.float32)

    category_raw = category_raw[keep_mask]
    raw_pest = raw_pest[keep_mask]
    pres = (raw_pest > 0).astype(np.float32)
    conc_log = np.log1p(np.where(raw_pest > 0, raw_pest, 0.0).astype(np.float32))
    return X_raw, category_raw, pres, conc_log


def evaluate_fold(model, X, food_true, pres_true, conc_true_log):
    model.eval()
    food_pred, pest_pred, conc_pred = [], [], []
    with torch.no_grad():
        for start in range(0, len(X), 256):
            X_b = torch.tensor(X[start:start + 256], dtype=torch.float32, device=device)
            food_logits, pest_logits, pest_conc = model(X_b)
            food_pred.extend(food_logits.argmax(-1).cpu().numpy())
            pest_pred.append((torch.sigmoid(pest_logits) >= 0.5).cpu().numpy().astype(int))
            conc_pred.append(pest_conc.cpu().numpy())
    food_pred = np.array(food_pred)
    pest_pred = np.concatenate(pest_pred, axis=0)
    conc_pred = np.concatenate(conc_pred, axis=0)

    food_acc = accuracy_score(food_true, food_pred) * 100.0
    food_prec = precision_score(food_true, food_pred, average="macro", zero_division=0) * 100.0
    food_rec = recall_score(food_true, food_pred, average="macro", zero_division=0) * 100.0
    food_f1 = f1_score(food_true, food_pred, average="macro", zero_division=0) * 100.0

    p_accs, p_precs, p_recs, p_f1s = [], [], [], []
    for j in range(len(substances)):
        p_accs.append(accuracy_score(pres_true[:, j], pest_pred[:, j]))
        p_precs.append(precision_score(pres_true[:, j], pest_pred[:, j], zero_division=0))
        p_recs.append(recall_score(pres_true[:, j], pest_pred[:, j], zero_division=0))
        p_f1s.append(f1_score(pres_true[:, j], pest_pred[:, j], zero_division=0))

    conc_true_mgkg = np.expm1(conc_true_log)
    conc_pred_mgkg = np.expm1(conc_pred)
    reg = {}
    for j, name in enumerate(substances):
        m = pres_true[:, j] == 1
        if m.sum() >= 2:
            mse = mean_squared_error(conc_true_mgkg[m, j], conc_pred_mgkg[m, j])
            reg[name] = {
                "mae": float(mean_absolute_error(conc_true_mgkg[m, j], conc_pred_mgkg[m, j])),
                "rmse": float(np.sqrt(mse)),
                "r2": float(r2_score(conc_true_mgkg[m, j], conc_pred_mgkg[m, j]) * 100.0),
                "n": int(m.sum()),
            }
        else:
            reg[name] = None

    return {
        "food_acc": food_acc, "food_prec": food_prec, "food_rec": food_rec, "food_f1": food_f1,
        "det_acc": float(np.mean(p_accs) * 100.0), "det_prec": float(np.mean(p_precs) * 100.0),
        "det_rec": float(np.mean(p_recs) * 100.0), "det_f1": float(np.mean(p_f1s) * 100.0),
        "reg": reg,
    }


if __name__ == "__main__":
    for machine, cfg in MACHINE_CFG.items():
        print(f"=== {machine} ===")
        X_raw, category_raw, pres, conc_log = load_test_set(machine, cfg["bin_factor"])
        print(f"n test rows: {len(X_raw)}, signal_len: {X_raw.shape[1]}")
        print("category distribution:", pd.Series(category_raw).value_counts().to_dict())
        print("presence rate per substance:", {s: float(pres[:, j].mean()) for j, s in enumerate(substances)})

        fold_results = []
        for fold in range(1, k_folds + 1):
            fold_dir = f"data/multitask_mt_smartnir/{machine}/fold_{fold}"
            stats = np.load(f"{fold_dir}/stats.npz")
            le = joblib.load(f"{fold_dir}/label_encoder.pkl")
            X = (X_raw - stats["mean"]) / stats["std"]
            food_true = le.transform(category_raw)

            model = MultiTaskSMARTNIRModel(
                n_food_classes=len(le.classes_), n_pesticides=len(substances),
                c_out=64, n_layers=6, n_heads=6, h_hidden=128, seq_len=X.shape[1],
                use_kan=True, dropout=0.1,
            ).to(device)
            model.load_state_dict(torch.load(
                f"checkpoint/multitask_mt_smartnir/{machine}/checkpoint_fold{fold}.pth", map_location=device))

            result = evaluate_fold(model, X, food_true, pres, conc_log)
            fold_results.append(result)
            print(f"  fold {fold}: food_acc={result['food_acc']:.2f} det_f1={result['det_f1']:.3f}")

        print(f"\n{machine} MT-SMARTNIR on held-out test-data (mean +/- std across {k_folds} folds):")
        for key in ["food_acc", "food_prec", "food_rec", "food_f1", "det_acc", "det_prec", "det_rec", "det_f1"]:
            vals = [r[key] for r in fold_results]
            print(f"  {key}: {np.mean(vals):.2f} +/- {np.std(vals):.2f}")
        for name in substances:
            vals = [r["reg"][name] for r in fold_results if r["reg"][name] is not None]
            if vals:
                for metric in ["mae", "rmse", "r2"]:
                    m = [v[metric] for v in vals]
                    print(f"  {name} {metric}: {np.mean(m):.3f} +/- {np.std(m):.3f} (n={vals[0]['n']})")
            else:
                print(f"  {name}: no present rows in test set")
        print()
