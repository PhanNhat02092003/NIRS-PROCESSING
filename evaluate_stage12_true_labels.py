"""Evaluates Stage 1 (substance detection) and Stage 2 (severity) of the
deployed pipeline using the GROUND-TRUTH food category as the one-hot
conditioning input, instead of GuidedDCNet's predicted category (which the
earlier evaluation found is only ~74-76% accurate on test-data) -- isolating
how good Stage 1/2 actually are on their own, decoupled from upstream
food-classification errors.

deploy_utils.infer_substances_detection(spectra, machine, categories) and
infer_substances_severity(spectra, machine, categories, detected_list) both
take `categories` (the Vietnamese food names) as a plain argument, so this
just calls them directly with the test set's true food label instead of
routing through infer_category_classification first -- no monkeypatching,
no retraining.

Usage: python3 evaluate_stage12_true_labels.py
"""

import os

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import deploy_utils
from evaluate_deployed_pipeline_testdata import FOOD_ID_TO_NAME, load_test_set
from deploy_utils import SUBSTANCE_NAME_TO_ID, SUBSTANCES


def evaluate_machine(machine: str, chunk_size: int = 4000):
    print(f"=== {machine} (true food labels fed to Stage 1/2) ===")
    spectra, food_id_true, presence_true, severity_true, safe_true = load_test_set(machine)
    n = len(spectra)
    print(f"n test rows: {n}")
    true_categories = [FOOD_ID_TO_NAME[fid] for fid in food_id_true]

    detected_sets = []
    over_threshold_sets = []
    safe_pred = []
    for start in range(0, n, chunk_size):
        chunk = spectra[start:start + chunk_size]
        cat_chunk = true_categories[start:start + chunk_size]

        detected_list, detect_conf = deploy_utils.infer_substances_detection(chunk, machine, cat_chunk)
        severity_list, severity_conf = deploy_utils.infer_substances_severity(
            chunk, machine, cat_chunk, detected_list)

        for detected, severity in zip(detected_list, severity_list):
            over_threshold = [s for s, verdict in severity.items() if verdict == "Vượt ngưỡng"]
            detected_sets.append(set(SUBSTANCE_NAME_TO_ID[s] for s in detected))
            over_threshold_sets.append(set(SUBSTANCE_NAME_TO_ID[s] for s in over_threshold))
            safe_pred.append(len(over_threshold) == 0)
        print(f"  {min(start + chunk_size, n)}/{n} rows done", flush=True)

    safe_pred = np.array(safe_pred, dtype=bool)

    print("\nPer-substance detection (Stage 1) with TRUE food label:")
    det_f1s = []
    for s in SUBSTANCES:
        pid = SUBSTANCE_NAME_TO_ID[s]
        y_true = presence_true[s]
        y_pred = np.array([pid in d for d in detected_sets])
        if y_true.sum() == 0:
            continue
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        det_f1s.append(f1)
        print(f"  {s:<20} n_pos={int(y_true.sum()):>5} precision={prec:.3f} recall={rec:.3f} f1={f1:.3f}")
    print(f"  macro-F1 across substances with >=1 positive: {np.mean(det_f1s):.3f}")

    print("\nPer-substance severity (Stage 2, restricted to truly-present rows) with TRUE food label:")
    sev_accs = []
    for s in SUBSTANCES:
        pid = SUBSTANCE_NAME_TO_ID[s]
        present_mask = presence_true[s]
        if present_mask.sum() == 0:
            continue
        y_true = severity_true[s][present_mask]
        y_pred = np.array([pid in d for d in over_threshold_sets])[present_mask]
        acc = accuracy_score(y_true, y_pred) * 100.0
        sev_accs.append(acc)
        print(f"  {s:<20} n_present={int(present_mask.sum()):>5} accuracy={acc:.2f}%")
    print(f"  mean accuracy across substances with >=1 present row: {np.mean(sev_accs):.2f}%")

    safe_acc = accuracy_score(safe_true, safe_pred) * 100.0
    safe_f1 = f1_score(safe_true, safe_pred, zero_division=0) * 100.0
    print(f"\nSample-level `safe` verdict: accuracy={safe_acc:.2f}% f1={safe_f1:.2f}%\n")


if __name__ == "__main__":
    for machine in ("FLAMENIR", "OCEANFX"):
        evaluate_machine(machine)
