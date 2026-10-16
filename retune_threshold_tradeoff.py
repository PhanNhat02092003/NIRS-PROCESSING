"""Explores the recall-floor vs sample-level false-alarm tradeoff that
stage1_detection_downsampled.py/promote_recall_threshold.py's Recall>=0.99
policy ignored: that policy was chosen purely from each SUBSTANCE's own
macro-F1 on the external test-data (see recalibrate_recall_target.py), which
doesn't see that with 19 independent per-substance gates, even a modest
per-substance false-positive rate compounds into an almost-certain false
alarm at the SAMPLE level (P(all 19 clean) ~= prod(1-FPR_i)). Measured on
real test-data: at Recall>=0.99, mean 10.5/19 substances are mislabeled per
row and precisely 0% of rows are fully correct, because Stage 1 false
positives feed Stage 2 substance-severity models with "absent" inputs they
were never trained to see (RegressionNIRSDataset drops absent rows), so
Stage 2 then free-wheels on those and very often also misfires.

This script computes, ONCE per substance/stage/machine, the two probability
arrays needed to evaluate ANY recall-floor candidate for free:
  - p_cal, y_cal per fold on ALL_downsampled.csv's held-back calibration
    split (reproduces recalibrate_recall_target.py's split exactly) --
    used to pick a threshold for a given target recall, per fold.
  - p_test_avg = the SAME 5-fold-averaged calibrated probability the
    deployed service computes (deploy_utils._smartnir_ensemble), on the
    REAL external test-data -- independent of any threshold choice.
Then sweeps a recall-target grid (including values below the current 0.99,
since this script's whole point is to check whether the current floor is
too loose) and reports, for each candidate, per-substance macro-F1 AND
sample-level metrics (mean substances-wrong/row, exact-match rate,
'An toan' false-alarm rate) -- all computed in-memory, no retraining, no
extra GPU forward passes per candidate.

Usage: MACHINE=FLAMENIR python3 retune_threshold_tradeoff.py
       MACHINE=OCEANFX python3 retune_threshold_tradeoff.py
Env: MACHINE (default FLAMENIR), CAL_FRAC (default 0.15), BIN,
     TARGETS (comma list of recall floors; default a broad grid).
"""

import json
import os
import pickle

import joblib
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

load_dotenv()

import deploy_utils
from dataset.preprocessing import preprocess_spectra
from deploy_utils import (FOOD_NAME_TO_INDEX, SUBSTANCE_NAME_TO_ID, SUBSTANCES, _food_onehot,
                           _preprocess, _smartnir_ensemble)
from evaluate_deployed_pipeline_testdata import FOOD_ID_TO_NAME, load_test_set
from model.smartnir_food_model import food_prior_shrinkage, predict_logit_diff
from stage1_detection_downsampled import pick_threshold_for_recall

STAGE_CONFIGS = {
    "stage1": dict(tag="stage1_smartnir_downsampled",
                    data_dir="data/substance_regression/{tag}/{machine}/{s}",
                    ckpt_dir="checkpoint/substance_regression/{tag}/{machine}/{s}"),
    "stage2": dict(tag="stage2_smartnir_downsampled",
                    data_dir="data/substance_severity_smartnir/{tag}/{machine}/{s}",
                    ckpt_dir="checkpoint/substance_severity_smartnir/{tag}/{machine}/{s}"),
}


def build_calibration_arrays(stage, machine, cal_frac=0.15):
    """Returns {substance: {"p_cal": [arr_per_fold], "y_cal": [arr_per_fold]}}
    for the given stage/machine, reproducing recalibrate_recall_target.py's
    split exactly (random_state=42 group split, test_size=cal_frac stratified
    train_test_split per fold with random_state=fold)."""
    cfg = STAGE_CONFIGS[stage]
    tag = cfg["tag"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    out = {}
    for s in SUBSTANCES:
        if stage == "stage2":
            pid = pesticide_name_to_id[s]
            mrl_per_row = food_id_per_row.map(lambda fid: thresholds[fid][pid]["mrl"])
            y = ((df[s] != -1) & (df[s] > mrl_per_row)).astype(int).values
        else:
            y = (df[s].values > 0).astype(int)

        data_dir = cfg["data_dir"].format(tag=tag, machine=machine, s=s)
        ckpt_dir = cfg["ckpt_dir"].format(tag=tag, machine=machine, s=s)
        if not os.path.exists(f"{ckpt_dir}/{s}_fold_1.pth"):
            print(f"  [{stage}/{s}] no trained checkpoint, skipping", flush=True)
            continue

        kf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        p_cal_folds, y_cal_folds = [], []
        for fold, (tr, va) in enumerate(kf.split(X, y, groups=groups), start=1):
            if cal_frac > 0:
                _, tr_cal = train_test_split(tr, test_size=cal_frac, stratify=y[tr], random_state=fold)
            else:
                tr_cal = tr

            nz = np.load(f"{data_dir}/fold_{fold}_norm.npz")
            Xs = torch.tensor((X - nz["mean"]) / nz["std"], dtype=torch.float32, device=device)
            model = deploy_utils.SmartNIRWithFood(Xs.shape[1], Fnp.shape[1]).to(device)
            model.load_state_dict(torch.load(f"{ckpt_dir}/{s}_fold_{fold}.pth", map_location=device))

            calib = joblib.load(f"{data_dir}/{s}_fold_{fold}_calibrator.pkl")
            idx_t = torch.as_tensor(tr_cal, device=device)
            d_cal = predict_logit_diff(model, Xs, Fs_full, idx_t, 512)
            p_cal = calib.predict_proba(d_cal.reshape(-1, 1))[:, 1]
            if stage == "stage2":
                prior_by_food = np.array(
                    json.load(open(f"{data_dir}/{s}_fold_{fold}_food_prior.json"))["prior_by_food"])
                p_cal = food_prior_shrinkage(p_cal, food_idx_all[tr_cal], prior_by_food)
            p_cal_folds.append(p_cal)
            y_cal_folds.append(y[tr_cal])
            del Xs
            torch.cuda.empty_cache()
        out[s] = {"p_cal": p_cal_folds, "y_cal": y_cal_folds}
        print(f"  [{stage}/{s}] calibration arrays built (5 folds)", flush=True)
    return out


def build_test_probs(stage, machine):
    """Returns {substance: avg_prob_test} -- the 5-fold-averaged calibrated
    probability on the real external test-data, exactly as the deployed
    service computes it (deploy_utils._smartnir_ensemble), independent of
    any threshold. Uses the TRUE food category (isolates substance-model
    quality from food-classification noise, same convention as
    evaluate_stage1_downsampled.py / evaluate_stage2_downsampled.py)."""
    cfg = STAGE_CONFIGS[stage]
    tag = cfg["tag"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spectra, food_id_true, presence_true, severity_true, safe_true = load_test_set(machine)
    true_categories = [FOOD_ID_TO_NAME[fid] for fid in food_id_true]
    spectra_pp = _preprocess(spectra, machine)
    Ft = torch.tensor(_food_onehot(true_categories), device=device)
    food_idx = np.array([FOOD_NAME_TO_INDEX.get(c, 0) for c in true_categories]) if stage == "stage2" else None

    out = {}
    for s in SUBSTANCES:
        data_dir = cfg["data_dir"].format(tag=tag, machine=machine, s=s)
        ckpt_dir = cfg["ckpt_dir"].format(tag=tag, machine=machine, s=s)
        avg_prob, _ = _smartnir_ensemble(spectra_pp, Ft, data_dir, ckpt_dir, s, device, food_idx=food_idx)
        if avg_prob is None:
            continue
        out[s] = avg_prob
        print(f"  [{stage}/{s}] test-set probabilities computed", flush=True)
    return out, presence_true, severity_true, safe_true


def pick_threshold_f1opt(y_true, y_prob):
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    precisions, recalls = precisions[:-1], recalls[:-1]
    f1s = np.where(precisions + recalls > 0, 2 * precisions * recalls / (precisions + recalls + 1e-12), 0.0)
    return float(thresholds[np.argmax(f1s)])


def threshold_for_target(p_cal_folds, y_cal_folds, target):
    if target == "f1opt":
        thrs = [pick_threshold_f1opt(y_cal_folds[f], p_cal_folds[f]) for f in range(len(p_cal_folds))]
    else:
        thrs = [pick_threshold_for_recall(y_cal_folds[f], p_cal_folds[f], target_recall=target)
                for f in range(len(p_cal_folds))]
    return float(np.mean(thrs))


def evaluate_candidate(stage, cal_arrays, test_probs, target, true_mat_dict, SUBSTANCES_present):
    """Returns (macro_f1, pred_mat) for one recall-floor candidate."""
    preds = {}
    f1s = []
    for s in SUBSTANCES_present:
        thr = threshold_for_target(cal_arrays[s]["p_cal"], cal_arrays[s]["y_cal"], target)
        pred = test_probs[s] > thr
        preds[s] = pred
        y_true = true_mat_dict[s]
        if y_true.sum() > 0:
            f1s.append(f1_score(y_true, pred, zero_division=0))
    return float(np.mean(f1s)), preds


def main():
    machine = os.environ.get("MACHINE", "FLAMENIR")
    cal_frac = float(os.environ.get("CAL_FRAC", 0.15))
    targets_env = os.environ.get("TARGETS")
    targets = (targets_env.split(",") if targets_env else
               [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, "f1opt"])
    targets = [float(t) if t != "f1opt" else t for t in targets]

    cache_path = f"/tmp/retune_cache_{machine}.pkl"
    if os.path.exists(cache_path) and os.environ.get("NO_CACHE") != "1":
        print(f"Loading cached calibration/test arrays from {cache_path}", flush=True)
        with open(cache_path, "rb") as f:
            cal1, test1, presence_true, severity_true, safe_true, cal2, test2 = pickle.load(f)
    else:
        print(f"=== {machine}: building Stage 1 calibration arrays ===", flush=True)
        cal1 = build_calibration_arrays("stage1", machine, cal_frac)
        print(f"=== {machine}: building Stage 1 test-set probabilities ===", flush=True)
        test1, presence_true, severity_true, safe_true = build_test_probs("stage1", machine)

        print(f"=== {machine}: building Stage 2 calibration arrays ===", flush=True)
        cal2 = build_calibration_arrays("stage2", machine, cal_frac)
        print(f"=== {machine}: building Stage 2 test-set probabilities ===", flush=True)
        test2, _, _, _ = build_test_probs("stage2", machine)

        with open(cache_path, "wb") as f:
            pickle.dump((cal1, test1, presence_true, severity_true, safe_true, cal2, test2), f)
        print(f"Cached calibration/test arrays to {cache_path}", flush=True)

    subs1 = [s for s in SUBSTANCES if s in cal1 and s in test1]
    subs2 = [s for s in SUBSTANCES if s in cal2 and s in test2]
    n = len(next(iter(test1.values())))
    presence_mat = np.stack([presence_true[s] for s in subs1], axis=1)
    severity_mat_by_sub = {s: severity_true[s] for s in subs2}

    print(f"\n{'target':>7} | {'S1 macroF1':>10} | {'S2 macroF1':>10} | "
          f"{'mean#wrong/19':>13} | {'exact-match%':>12} | {'falsealarm%':>11}")
    print("-" * 80)

    results = []
    for target in targets:
        f1_s1, preds1 = evaluate_candidate("stage1", cal1, test1, target, presence_true, subs1)
        f1_s2, preds2 = evaluate_candidate("stage2", cal2, test2, target, severity_mat_by_sub, subs2)

        pred_presence_mat = np.stack([preds1[s] for s in subs1], axis=1)
        wrong_per_row = (pred_presence_mat != presence_mat).sum(axis=1)
        exact_match = np.mean(wrong_per_row == 0) * 100

        # sample-level: a substance only "fires" severity if stage1 predicts it present
        # (matches deploy semantics -- Stage 2 only runs on Stage-1-detected substances)
        over_mat = np.zeros((n, len(subs2)), dtype=bool)
        for j, s in enumerate(subs2):
            stage1_says_present = preds1.get(s, np.zeros(n, dtype=bool))
            over_mat[:, j] = stage1_says_present & preds2[s]
        pred_unsafe = over_mat.any(axis=1)
        pred_safe = ~pred_unsafe
        false_alarm = np.mean(~pred_safe[safe_true]) * 100 if safe_true.sum() > 0 else float("nan")

        print(f"{target:>7} | {f1_s1:>10.3f} | {f1_s2:>10.3f} | "
              f"{wrong_per_row.mean():>13.2f} | {exact_match:>12.3f} | {false_alarm:>11.1f}")
        results.append(dict(target=target, f1_s1=f1_s1, f1_s2=f1_s2,
                             mean_wrong=float(wrong_per_row.mean()), exact_match=exact_match,
                             false_alarm=false_alarm))

    out_path = f"/tmp/retune_tradeoff_{machine}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full sweep to {out_path}")


if __name__ == "__main__":
    main()
