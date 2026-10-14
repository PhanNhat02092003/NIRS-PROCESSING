"""Compares the DEPLOYED Stage 2 (checkpoint/substance_severity_smartnir,
plain StratifiedKFold on full ALL.csv) against the retrained Stage 2
(tag=stage2_smartnir_downsampled, StratifiedGroupKFold on
ALL_downsampled.csv, see stage2_safety_downsampled.py) on the external
test-data set.

Uses the TRUE food label (not GuidedDCNet's predicted one) as the one-hot
input, same methodology as evaluate_stage1_downsampled.py -- isolates the
Stage-2-only change. Also bypasses Stage 1 entirely (unlike
evaluate_stage12_true_labels.py's cascade): severity is scored directly
against the TRUE presence mask, not whatever Stage 1 happened to detect, so
this measures Stage 2's own quality, undiluted by Stage 1 errors.

Usage: python3 evaluate_stage2_downsampled.py
"""

import json
import os

import numpy as np
import torch
from dotenv import load_dotenv
from sklearn.metrics import f1_score, precision_score, recall_score

load_dotenv()

import deploy_utils
from deploy_utils import (FOOD_NAME_TO_INDEX, SUBSTANCE_NAME_TO_ID, SUBSTANCES, _food_onehot,
                           _preprocess, _smartnir_ensemble)
from evaluate_deployed_pipeline_testdata import FOOD_ID_TO_NAME, load_test_set
from model.smartnir_food_model import SmartNIRWithFood, food_prior_shrinkage, predict_logit_diff


def _severity_ensemble_custom_thr(spectra, Ft, food_idx, data_dir, ckpt_dir, substance, device, thr_filename):
    """Same averaging as deploy_utils._smartnir_ensemble (incl. food-prior
    shrinkage), but reads a caller-specified threshold filename (e.g.
    '{s}_fold_{n}_threshold_recall0.95.json', written by
    stage2_recalibrate_recall_target.py) instead of the default
    '{s}_fold_{n}_threshold.json'."""
    import joblib
    probs, thrs = [], []
    idx = torch.arange(len(spectra), device=device)
    for fold in range(1, deploy_utils.K_FOLDS + 1):
        norm_path = f"{data_dir}/fold_{fold}_norm.npz"
        model_path = f"{ckpt_dir}/{substance}_fold_{fold}.pth"
        calib_path = f"{data_dir}/{substance}_fold_{fold}_calibrator.pkl"
        prior_path = f"{data_dir}/{substance}_fold_{fold}_food_prior.json"
        thr_path = f"{data_dir}/{substance}_fold_{fold}_{thr_filename}.json"
        if not all(os.path.exists(p) for p in (norm_path, model_path, calib_path, prior_path, thr_path)):
            continue
        nz = np.load(norm_path)
        Xn = torch.tensor((spectra - nz["mean"]) / nz["std"], dtype=torch.float32, device=device)
        model = SmartNIRWithFood(Xn.shape[1], Ft.shape[1]).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        score = predict_logit_diff(model, Xn, Ft, idx, 256)
        calib = joblib.load(calib_path)
        p = calib.predict_proba(score.reshape(-1, 1))[:, 1]
        prior_by_food = np.array(json.load(open(prior_path))["prior_by_food"])
        p = food_prior_shrinkage(p, food_idx, prior_by_food)
        probs.append(p)
        thrs.append(json.load(open(thr_path))["threshold"])
    if not probs:
        return None, None
    return np.mean(probs, axis=0), float(np.mean(thrs))


def detect_severity_with_tag(spectra, machine, categories, task_dir, chunk_size=4000, thr_filename="threshold"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spectra_pp = _preprocess(spectra, machine)
    Ft_full = torch.tensor(_food_onehot(categories), device=device)
    food_idx = np.array([FOOD_NAME_TO_INDEX.get(c, 0) for c in categories])

    probs = {}
    for substance in SUBSTANCES:
        data_dir = f"data/{task_dir}/{machine}/{substance}"
        ckpt_dir = f"checkpoint/{task_dir}/{machine}/{substance}"
        all_prob = np.full(len(spectra_pp), np.nan, dtype=np.float32)
        all_thr = []
        for start in range(0, len(spectra_pp), chunk_size):
            end = min(start + chunk_size, len(spectra_pp))
            chunk = spectra_pp[start:end]
            Ft = Ft_full[start:end]
            fidx = food_idx[start:end]
            if thr_filename == "threshold":
                avg_prob, avg_thr = _smartnir_ensemble(chunk, Ft, data_dir, ckpt_dir, substance, device, food_idx=fidx)
            else:
                avg_prob, avg_thr = _severity_ensemble_custom_thr(
                    chunk, Ft, fidx, data_dir, ckpt_dir, substance, device, thr_filename)
            if avg_prob is None:
                break
            all_prob[start:end] = avg_prob
            all_thr.append(avg_thr)
        if np.isnan(all_prob).any():
            continue
        probs[substance] = (all_prob, float(np.mean(all_thr)))
    return probs


def evaluate_tag(machine, task_dir, label, chunk_size=4000, thr_filename="threshold"):
    spectra, food_id_true, presence_true, severity_true, safe_true = load_test_set(machine)
    true_categories = [FOOD_ID_TO_NAME[fid] for fid in food_id_true]

    probs = detect_severity_with_tag(spectra, machine, true_categories, task_dir, chunk_size=chunk_size,
                                      thr_filename=thr_filename)

    print(f"\n--- {machine} / {label} ({task_dir}) ---")
    sev_f1s, sev_precs, sev_recs = [], [], []
    for s in SUBSTANCES:
        if s not in probs:
            continue
        present_mask = presence_true[s]
        if present_mask.sum() == 0:
            continue
        all_prob, thr = probs[s]
        y_pred = (all_prob > thr).astype(int)[present_mask]
        y_true = severity_true[s][present_mask]
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        sev_f1s.append(f1); sev_precs.append(prec); sev_recs.append(rec)
        print(f"  {s:<20} n_present={int(present_mask.sum()):>5} precision={prec:.3f} recall={rec:.3f} f1={f1:.3f}")
    print(f"  macro: precision={np.mean(sev_precs):.3f} recall={np.mean(sev_recs):.3f} F1={np.mean(sev_f1s):.3f}")
    return np.mean(sev_f1s)


if __name__ == "__main__":
    for machine in ("FLAMENIR", "OCEANFX"):
        f1_old = evaluate_tag(machine, "substance_severity_smartnir", "DEPLOYED (leaky CV)")
        f1_new = evaluate_tag(machine, "substance_severity_smartnir/stage2_smartnir_downsampled",
                               "RETRAINED (group-safe)")
        print(f"\n=== {machine}: macro-F1 deployed={f1_old:.3f}  retrained(group-safe)={f1_new:.3f} ===\n")
