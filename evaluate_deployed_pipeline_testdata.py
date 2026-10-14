"""Evaluates the ACTUAL DEPLOYED pipeline (main branch's app.py/utils.py:
GuidedDCNet food classification -> SMART-NIR+food-onehot Stage 1 detection
-> SMART-NIR+food-onehot+prior-shrinkage Stage 2 severity) on the new
held-out external test set (DATASET_ROOT/test-data/{Machine}.csv, added
2026-10-13), instead of any of this session's experimental multitask
models (MT-SMARTNIR / XSpecMamba-joint).

deploy_utils.py is `git show main:utils.py` copied verbatim into this
(train-branch) working tree; model/smartnir_food_model.py was similarly
pulled from main (train didn't have it as a standalone module). All other
model files main's utils.py imports (guideddcnet_model.py,
classification_model.py, dataset/preprocessing.py) are byte-identical
between branches (checked via `git diff main train -- <file>`), so this
runs the exact same inference code the deployed service runs, against the
exact same checkpoints already on disk -- not a reimplementation.

Ground truth per row:
  - food category: the test CSV's abbreviated code (e.g. "CACHUA") mapped
    to a food id via DATASET_ROOT/food_ids.json's own `code` field
    (case-insensitive) -- the same codes, not a guessed mapping.
  - substance presence: comp_<Substance> > 0 (the test file's own absence
    sentinel is 0.0, not training's -1, but ">0" is a presence check
    either way, so this doesn't need special-casing).
  - substance severity ("Vuot nguong"): present AND comp_<Substance> >
    DATASET_ROOT/thresholds.json[food_id][pesticide_id]["mrl"] -- the same
    label rule stage2_safety.py uses to build its own training targets.
  - sample-level `safe`: True iff no substance is "Vuot nguong" for that
    row's true food category.

Compares the pipeline's end-to-end predictions (which depend on its own
upstream category/detection calls, exactly as a real request would) against
this ground truth: food accuracy, per-substance detection precision/
recall/F1, per-substance severity accuracy (restricted to truly-present
substances), and sample-level `safe` accuracy.
"""

import json
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

load_dotenv()

from deploy_utils import FOOD_NAME_TO_ID, SUBSTANCE_NAME_TO_ID, SUBSTANCES, analyze_spectrum

DATASET_ROOT = os.environ["DATASET_ROOT"]
FOOD_IDS = json.load(open(f"{DATASET_ROOT}/food_ids.json"))
THRESHOLDS = json.load(open(f"{DATASET_ROOT}/thresholds.json"))

# code (e.g. "Cachua") -> F01.. id, built from food_ids.json itself rather
# than a hand-typed table, then matched case-insensitively against the test
# file's upper-cased codes (e.g. "CACHUA").
CODE_TO_FOOD_ID = {v["code"].upper(): k for k, v in FOOD_IDS.items()}
FOOD_ID_TO_NAME = {k: v["name"] for k, v in FOOD_IDS.items()}
NAME_TO_FOOD_ID = {v["name"]: k for k, v in FOOD_IDS.items()}


def comp_column(df: pd.DataFrame, substance: str) -> str:
    """comp_<Substance> columns in the test file carry stray trailing
    whitespace on at least one name ("comp_Emamectin benzoate ") -- match by
    stripped name instead of an exact f-string."""
    target = f"comp_{substance}".strip()
    for c in df.columns:
        if c.strip() == target:
            return c
    raise KeyError(f"No comp_ column found for substance {substance!r}")


def load_test_set(machine: str):
    path = f"{DATASET_ROOT}/test-data/{machine}.csv"
    df = pd.read_csv(path)

    w_cols = sorted((c for c in df.columns if c.startswith("w_")), key=lambda c: int(c.split("_")[1]))
    spectra = df[w_cols].values.astype(np.float32)

    food_id_true = df["category"].str.upper().map(CODE_TO_FOOD_ID)
    unmapped = food_id_true.isna()
    if unmapped.any():
        raise ValueError(f"Unmapped category codes in {machine} test set: "
                          f"{sorted(set(df['category'][unmapped]))}")
    food_id_true = food_id_true.values

    presence_true = {}
    severity_true = {}
    for s in SUBSTANCES:
        col = comp_column(df, s)
        vals = df[col].values.astype(np.float32)
        present = vals > 0
        presence_true[s] = present
        pid = SUBSTANCE_NAME_TO_ID[s]
        mrl = np.array([THRESHOLDS[fid][pid]["mrl"] for fid in food_id_true])
        severity_true[s] = present & (vals > mrl)

    safe_true = ~np.any(np.stack(list(severity_true.values()), axis=1), axis=1)
    return spectra, food_id_true, presence_true, severity_true, safe_true


def evaluate_machine(machine: str, chunk_size: int = 4000):
    print(f"=== {machine} ===")
    spectra, food_id_true, presence_true, severity_true, safe_true = load_test_set(machine)
    n = len(spectra)
    print(f"n test rows: {n}")

    food_id_pred = []
    detected_sets = []
    over_threshold_sets = []
    safe_pred = []
    for start in range(0, n, chunk_size):
        chunk = spectra[start:start + chunk_size]
        results = analyze_spectrum(chunk, machine)
        for r in results:
            food_id_pred.append(r["category"]["code"])
            detected_sets.append({d["code"] for d in r["substances_detected"]})
            over_threshold_sets.append({d["code"] for d in r["substances_over_threshold"]})
            safe_pred.append(r["safe"])
        print(f"  {min(start + chunk_size, n)}/{n} rows done")

    food_id_pred = np.array(food_id_pred)
    safe_pred = np.array(safe_pred, dtype=bool)

    food_acc = accuracy_score(food_id_true, food_id_pred) * 100.0
    print(f"\nFood classification accuracy: {food_acc:.2f}%")

    print("\nPer-substance detection (Stage 1):")
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

    print("\nPer-substance severity (Stage 2, restricted to truly-present rows):")
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
    print(f"\nSample-level `safe` verdict: accuracy={safe_acc:.2f}% f1={safe_f1:.2f}%")
    print()


if __name__ == "__main__":
    for machine in ("FLAMENIR", "OCEANFX"):
        evaluate_machine(machine)
