"""Post-hoc decision threshold at an arbitrary Recall target, for either
stage's already-trained downsampled models (no retraining -- same
models/calibrators, reproduces the same tr_cal split). Merges what were
separate stage1_recalibrate_recall_target.py / stage2_recalibrate_recall_target.py
scripts -- the two stages only differ in how the label is computed (presence
vs MRL-relative severity) and in Stage 2's extra food_prior_shrinkage step
(see STAGE below).

Why this exists: the Stage 1 ablation found that under domain shift between
train and the external test-data, a LOOSER recall floor (0.95/0.99)
consistently beat the original Recall>=0.9 default, because the calibrated
probability scale itself drifts and a stricter in-distribution threshold
doesn't survive that drift -- confirmed to hold for Stage 2 too. Writes
{substance}_fold_{fold}_threshold_recall{target}.json next to the existing
{substance}_fold_{fold}_threshold.json (left untouched); see
promote_recall_threshold.py to make a given target the new official default.

Usage: STAGE=stage1 MACHINE=FLAMENIR TARGET_RECALL=0.95 python3 recalibrate_recall_target.py
       STAGE=stage2 MACHINE=OCEANFX BIN=8 TARGET_RECALL=0.99 python3 recalibrate_recall_target.py

Env: STAGE (stage1 | stage2, default stage1), MACHINE (default FLAMENIR),
TARGET_RECALL (default 0.9), CAL_FRAC (default 0.15), BIN, SUBSTANCES
(comma list; default all 19).
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from dataset.preprocessing import preprocess_spectra
from model.smartnir_food_model import SmartNIRWithFood, food_prior_shrinkage, predict_logit_diff
from stage1_detection_downsampled import ALL_SUBSTANCES, pick_threshold_for_recall

load_dotenv()

STAGE_CONFIGS = {
    "stage1": dict(tag="stage1_smartnir_downsampled",
                    data_dir="data/substance_regression/{tag}/{machine}/{s}",
                    ckpt_dir="checkpoint/substance_regression/{tag}/{machine}/{s}"),
    "stage2": dict(tag="stage2_smartnir_downsampled",
                    data_dir="data/substance_severity_smartnir/{tag}/{machine}/{s}",
                    ckpt_dir="checkpoint/substance_severity_smartnir/{tag}/{machine}/{s}"),
}


def main():
    stage = os.environ.get("STAGE", "stage1")
    cfg = STAGE_CONFIGS[stage]
    tag = cfg["tag"]
    machine = os.environ.get("MACHINE", "FLAMENIR")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cal_frac = float(os.environ.get("CAL_FRAC", 0.15))
    target_recall = float(os.environ.get("TARGET_RECALL", 0.9))
    subs = os.environ.get("SUBSTANCES")
    substances = [s.strip() for s in subs.split(",")] if subs else ALL_SUBSTANCES

    root = os.environ["DATASET_ROOT"]
    df = pd.read_csv(f"{root}/{machine}/ALL_downsampled.csv")
    w_cols = sorted((c for c in df.columns if c.startswith("w_")), key=lambda c: int(c.split("_")[1]))
    X, keep = preprocess_spectra(df[w_cols].values.astype(np.float32))
    df = df[keep].reset_index(drop=True)

    bin_factor = int(os.environ.get("BIN", 8 if machine == "OCEANFX" else 1))
    if bin_factor > 1:
        n_out = (X.shape[1] // bin_factor) // 8 * 8
        X = X[:, :n_out * bin_factor].reshape(len(X), n_out, bin_factor).mean(axis=2).astype(np.float32)

    with open(f"{root}/food_ids.json") as f:
        food_ids = json.load(f)
    name2id = {v["name"]: k for k, v in food_ids.items()}
    Fnp = pd.get_dummies(df["category"].map(name2id)).reindex(
        columns=list(food_ids), fill_value=0).astype(np.float32).values
    groups = df["_group_id"].values
    Fs_full = torch.tensor(Fnp, device=device)
    food_idx_all = Fnp.argmax(axis=1)

    if stage == "stage2":
        with open(f"{root}/pesticide_ids.json") as f:
            pesticide_ids = json.load(f)
        with open(f"{root}/thresholds.json") as f:
            thresholds = json.load(f)
        pesticide_name_to_id = {v["name"]: k for k, v in pesticide_ids.items()}
        food_id_per_row = df["category"].map(name2id)

    print(f"{machine}: {len(df)} rows, {X.shape[1]} wavelengths, stage={stage}, "
          f"target_recall={target_recall}", flush=True)

    suffix = f"threshold_recall{target_recall}"
    for s in substances:
        if stage == "stage2":
            pid = pesticide_name_to_id[s]
            mrl_per_row = food_id_per_row.map(lambda fid: thresholds[fid][pid]["mrl"])
            y = ((df[s] != -1) & (df[s] > mrl_per_row)).astype(int).values
        else:
            y = (df[s].values > 0).astype(int)

        data_dir = cfg["data_dir"].format(tag=tag, machine=machine, s=s)
        ckpt_dir = cfg["ckpt_dir"].format(tag=tag, machine=machine, s=s)
        if not os.path.exists(f"{ckpt_dir}/{s}_fold_1.pth"):
            print(f"  [{s}] no trained checkpoint, skipping")
            continue

        kf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        for fold, (tr, va) in enumerate(kf.split(X, y, groups=groups), start=1):
            if cal_frac > 0:
                _, tr_cal = train_test_split(tr, test_size=cal_frac, stratify=y[tr], random_state=fold)
            else:
                tr_cal = tr

            nz = np.load(f"{data_dir}/fold_{fold}_norm.npz")
            Xs = torch.tensor((X - nz["mean"]) / nz["std"], dtype=torch.float32, device=device)
            model = SmartNIRWithFood(Xs.shape[1], Fnp.shape[1]).to(device)
            model.load_state_dict(torch.load(f"{ckpt_dir}/{s}_fold_{fold}.pth", map_location=device))

            calib = joblib.load(f"{data_dir}/{s}_fold_{fold}_calibrator.pkl")

            idx_t = torch.as_tensor(tr_cal, device=device)
            d_cal = predict_logit_diff(model, Xs, Fs_full, idx_t, 512)
            p_cal = calib.predict_proba(d_cal.reshape(-1, 1))[:, 1]
            if stage == "stage2":
                prior_by_food = np.array(
                    json.load(open(f"{data_dir}/{s}_fold_{fold}_food_prior.json"))["prior_by_food"])
                p_cal = food_prior_shrinkage(p_cal, food_idx_all[tr_cal], prior_by_food)
            y_cal = y[tr_cal]

            thr = pick_threshold_for_recall(y_cal, p_cal, target_recall=target_recall)
            with open(f"{data_dir}/{s}_fold_{fold}_{suffix}.json", "w") as f:
                json.dump({"threshold": thr, "target_recall": target_recall}, f, indent=4)
            del Xs
            torch.cuda.empty_cache()
        print(f"  [{s}] recall={target_recall} thresholds written for all 5 folds", flush=True)


if __name__ == "__main__":
    main()
