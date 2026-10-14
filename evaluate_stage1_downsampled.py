"""Compares the DEPLOYED Stage 1 (tag=stage1_smartnir_os2, plain StratifiedKFold
on full ALL.csv) against the retrained Stage 1 (tag=stage1_smartnir_downsampled,
StratifiedGroupKFold on ALL_downsampled.csv + near-LOD positive filtering, see
stage1_detection_downsampled.py) on the external test-data set.

Both use the TRUE food label (not GuidedDCNet's predicted one) as the one-hot
input, same as evaluate_stage12_true_labels.py -- isolates the Stage-1 change
from food-classification accuracy, since that's what changed here.

Usage: python3 evaluate_stage1_downsampled.py
"""

import os

import numpy as np
import torch
from dotenv import load_dotenv
from sklearn.metrics import f1_score, precision_score, recall_score

load_dotenv()

import json

import deploy_utils
from deploy_utils import (FOOD_NAME_TO_INDEX, N_FOOD, SUBSTANCE_NAME_TO_ID, SUBSTANCES,
                           _food_onehot, _preprocess, _smartnir_ensemble)
from evaluate_deployed_pipeline_testdata import FOOD_ID_TO_NAME, load_test_set


def _ensemble_prob_and_thr_custom(spectra, Ft, data_dir, ckpt_dir, substance, device, thr_filename,
                                   calib_filename="calibrator", isotonic=False):
    """Same averaging as deploy_utils._smartnir_ensemble, but reads a
    caller-specified threshold filename (e.g. '{s}_fold_{n}_threshold_f1opt.json',
    written by stage1_recalibrate_f1opt.py) instead of the default
    '{s}_fold_{n}_threshold.json', and optionally a caller-specified
    calibrator filename/kind (e.g. calibrator_isotonic.pkl, an
    IsotonicRegression from stage1_recalibrate_isotonic.py, whose .predict
    takes the raw score directly instead of .predict_proba(score.reshape(-1,1))[:,1])."""
    from model.smartnir_food_model import SmartNIRWithFood, predict_logit_diff
    import joblib
    probs, thrs = [], []
    idx = torch.arange(len(spectra), device=device)
    for fold in range(1, deploy_utils.K_FOLDS + 1):
        norm_path = f"{data_dir}/fold_{fold}_norm.npz"
        model_path = f"{ckpt_dir}/{substance}_fold_{fold}.pth"
        calib_path = f"{data_dir}/{substance}_fold_{fold}_{calib_filename}.pkl"
        thr_path = f"{data_dir}/{substance}_fold_{fold}_{thr_filename}.json"
        if not all(os.path.exists(p) for p in (norm_path, model_path, calib_path, thr_path)):
            continue
        nz = np.load(norm_path)
        Xn = torch.tensor((spectra - nz["mean"]) / nz["std"], dtype=torch.float32, device=device)
        model = SmartNIRWithFood(Xn.shape[1], Ft.shape[1]).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        score = predict_logit_diff(model, Xn, Ft, idx, 256)
        calib = joblib.load(calib_path)
        p = calib.predict(score) if isotonic else calib.predict_proba(score.reshape(-1, 1))[:, 1]
        probs.append(p)
        thrs.append(json.load(open(thr_path))["threshold"])
    if not probs:
        return None, None
    return np.mean(probs, axis=0), float(np.mean(thrs))


def detect_with_tag(spectra: np.ndarray, machine: str, categories, tag: str, chunk_size=4000,
                     thr_filename: str = "threshold", calib_filename: str = "calibrator",
                     isotonic: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spectra_pp = _preprocess(spectra, machine)
    Ft_full = torch.tensor(_food_onehot(categories), device=device)

    detected_sets = [set() for _ in range(len(spectra_pp))]
    confidences = [{} for _ in range(len(spectra_pp))]
    for substance in SUBSTANCES:
        data_dir = f"data/substance_regression/{tag}/{machine}/{substance}"
        ckpt_dir = f"checkpoint/substance_regression/{tag}/{machine}/{substance}"
        all_prob = np.full(len(spectra_pp), np.nan, dtype=np.float32)
        all_thr = []
        for start in range(0, len(spectra_pp), chunk_size):
            end = min(start + chunk_size, len(spectra_pp))
            chunk = spectra_pp[start:end]
            Ft = Ft_full[start:end]
            if thr_filename == "threshold" and calib_filename == "calibrator":
                avg_prob, avg_thr = _smartnir_ensemble(chunk, Ft, data_dir, ckpt_dir, substance, device)
            else:
                avg_prob, avg_thr = _ensemble_prob_and_thr_custom(
                    chunk, Ft, data_dir, ckpt_dir, substance, device, thr_filename,
                    calib_filename=calib_filename, isotonic=isotonic)
            if avg_prob is None:
                break
            all_prob[start:end] = avg_prob
            all_thr.append(avg_thr)
        if np.isnan(all_prob).any():
            continue
        thr = float(np.mean(all_thr))
        pid = SUBSTANCE_NAME_TO_ID[substance]
        for i, detected in enumerate(all_prob > thr):
            if detected:
                detected_sets[i].add(pid)
                confidences[i][pid] = float(all_prob[i])
    return detected_sets, confidences


def _ensemble_prob_and_perfood_thr(spectra, Ft, data_dir, ckpt_dir, substance, device):
    """Same averaging as deploy_utils._smartnir_ensemble, but reads
    {substance}_fold_{n}_threshold_per_food.json (written by
    stage1_recalibrate_per_food.py) and averages the per-food threshold
    VECTOR across folds instead of a single scalar. Returns
    (avg_prob, avg_thr_vec[N_FOOD]) or (None, None) if artefacts are missing."""
    from model.smartnir_food_model import SmartNIRWithFood, predict_logit_diff
    import joblib
    probs, thr_vecs = [], []
    idx = torch.arange(len(spectra), device=device)
    for fold in range(1, deploy_utils.K_FOLDS + 1):
        norm_path = f"{data_dir}/fold_{fold}_norm.npz"
        model_path = f"{ckpt_dir}/{substance}_fold_{fold}.pth"
        calib_path = f"{data_dir}/{substance}_fold_{fold}_calibrator.pkl"
        thr_path = f"{data_dir}/{substance}_fold_{fold}_threshold_per_food.json"
        if not all(os.path.exists(p) for p in (norm_path, model_path, calib_path, thr_path)):
            continue
        nz = np.load(norm_path)
        Xn = torch.tensor((spectra - nz["mean"]) / nz["std"], dtype=torch.float32, device=device)
        model = SmartNIRWithFood(Xn.shape[1], Ft.shape[1]).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        score = predict_logit_diff(model, Xn, Ft, idx, 256)
        calib = joblib.load(calib_path)
        p = calib.predict_proba(score.reshape(-1, 1))[:, 1]
        probs.append(p)
        pf = json.load(open(thr_path))["per_food"]
        thr_vecs.append([pf[str(i)] for i in range(N_FOOD)])
    if not probs:
        return None, None
    return np.mean(probs, axis=0), np.mean(thr_vecs, axis=0)


def detect_with_tag_per_food(spectra: np.ndarray, machine: str, categories, tag: str, chunk_size=4000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spectra_pp = _preprocess(spectra, machine)
    Ft_full = torch.tensor(_food_onehot(categories), device=device)
    food_idx = np.array([FOOD_NAME_TO_INDEX.get(c, 0) for c in categories])

    detected_sets = [set() for _ in range(len(spectra_pp))]
    for substance in SUBSTANCES:
        data_dir = f"data/substance_regression/{tag}/{machine}/{substance}"
        ckpt_dir = f"checkpoint/substance_regression/{tag}/{machine}/{substance}"
        all_prob = np.full(len(spectra_pp), np.nan, dtype=np.float32)
        thr_vecs = []
        for start in range(0, len(spectra_pp), chunk_size):
            end = min(start + chunk_size, len(spectra_pp))
            chunk = spectra_pp[start:end]
            Ft = Ft_full[start:end]
            avg_prob, thr_vec = _ensemble_prob_and_perfood_thr(chunk, Ft, data_dir, ckpt_dir, substance, device)
            if avg_prob is None:
                break
            all_prob[start:end] = avg_prob
            thr_vecs.append(thr_vec)
        if np.isnan(all_prob).any():
            continue
        thr_vec = np.mean(thr_vecs, axis=0)
        thr_per_row = thr_vec[food_idx]
        pid = SUBSTANCE_NAME_TO_ID[substance]
        for i, detected in enumerate(all_prob > thr_per_row):
            if detected:
                detected_sets[i].add(pid)
    return detected_sets


def evaluate_tag(machine: str, tag: str, label: str, per_food: bool = False, thr_filename: str = "threshold",
                  calib_filename: str = "calibrator", isotonic: bool = False):
    spectra, food_id_true, presence_true, severity_true, safe_true = load_test_set(machine)
    true_categories = [FOOD_ID_TO_NAME[fid] for fid in food_id_true]

    if per_food:
        detected_sets = detect_with_tag_per_food(spectra, machine, true_categories, tag)
    else:
        detected_sets, _ = detect_with_tag(spectra, machine, true_categories, tag, thr_filename=thr_filename,
                                            calib_filename=calib_filename, isotonic=isotonic)

    print(f"\n--- {machine} / {label} (tag={tag}) ---")
    det_f1s, det_precs, det_recs = [], [], []
    for s in SUBSTANCES:
        pid = SUBSTANCE_NAME_TO_ID[s]
        y_true = presence_true[s]
        y_pred = np.array([pid in d for d in detected_sets])
        if y_true.sum() == 0:
            continue
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        det_f1s.append(f1); det_precs.append(prec); det_recs.append(rec)
        print(f"  {s:<20} n_pos={int(y_true.sum()):>5} precision={prec:.3f} recall={rec:.3f} f1={f1:.3f}")
    print(f"  macro: precision={np.mean(det_precs):.3f} recall={np.mean(det_recs):.3f} F1={np.mean(det_f1s):.3f}")
    return np.mean(det_f1s)


if __name__ == "__main__":
    for machine in ("FLAMENIR", "OCEANFX"):
        f1_old = evaluate_tag(machine, "stage1_smartnir_os2", "DEPLOYED (leaky CV)")
        f1_new = evaluate_tag(machine, "stage1_smartnir_downsampled", "RETRAINED (group-safe + low-conc filter)")
        f1_pf = evaluate_tag(machine, "stage1_smartnir_downsampled", "RETRAINED + per-food threshold", per_food=True)
        print(f"\n=== {machine}: macro-F1 deployed={f1_old:.3f}  retrained(global thr)={f1_new:.3f}  "
              f"retrained(per-food thr)={f1_pf:.3f} ===\n")
