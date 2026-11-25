"""Trains the model that resolves which of the 4 leafy-green categories
(Xa Lach, Cai Be Xanh, Cai Thia, Mong Toi) a sample actually is, for use
when the production 6-class GuidedDCNet classifier (see
food_classification_6class.py) predicts the merged "Rau La" label.

Trained on {DATASET_ROOT}/leafy-position-scan/{machine}/processed/ALL.csv,
position == "La 2" only (the inner leaf face -- found to give the best
separation among the 4 categories in analyze_leafy_positions.py). This CSV
now holds two scanning sessions, distinguished by its "source" column
("original" vs "new-test-data-2026-11", the later one folded in by
../new-test-data + this file's own merge step -- see that folder's git
history / the .bak-pre-newtestdata-merge sibling of this CSV for the
pre-merge data): having a second, independent session in training turned
out to matter a lot for cross-session generalization (see below), and the
"source" column is what lets held-out CV below hold out an entire session
rather than a random split that could let two scans of the same physical
leaf land on both sides.

OCEANFX: plain SG+SNV + LDA, same approach that already generalized well
session-to-session.

FLAMENIR: SNV + 1st-derivative + RandomForest, NOT plain SG+SNV + LDA.
Plain SNV+LDA scored ~85% in-session CV but only ~24% when validated
against a held-out *second* scanning session (../new-test-data, before it
was merged in here) -- FLAMENIR's much coarser 128-point spectrum carries
far less margin between these 4 very similar leafy greens, so day-to-day
instrument/sample drift (which SNV's per-sample scatter correction doesn't
remove -- it's a whole-curve-level correction, not a local-shape one)
wrecks LDA's linear boundary specifically on FLAMENIR. SNV +
1st-derivative (local curve *shape*, not level) + a RandomForest
(nonlinear, more robust to a shifted-but-similarly-shaped boundary than
LDA) recovered most of that: ~80% cross-session CV vs ~24% for the old
recipe on the exact same held-out session.

Usage: python3 train_leafy_pca_lda.py [FLAMENIR OCEANFX]
Output: checkpoint/leafy_pca_lda/{machine}/model.joblib, a dict:
  {"pipeline": sklearn Pipeline,
   "classes": sorted list of the 4 category names,
   "preprocess": "plain" | "derivative" -- which per-sample transform
                 deploy_utils.py's _resolve_leafy_subtype() must apply,
   "n_pca": chosen PCA n_components,
   "cv_accuracy": (mean, std) -- cross-SESSION estimate (held-out source=
                  "new-test-data-2026-11" folds), not a blind in-pool CV,
                  which would overstate accuracy once the same session
                  appears on both sides of the split.}
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dataset.preprocessing import preprocess_spectra, savgol_smooth, snv

load_dotenv()

N_PCA_CANDIDATES = [10, 20, 30, 50, 80, 120, 180]
SEED = 42
LEAFY_SCAN_DIR = f"{os.environ['DATASET_ROOT']}/leafy-position-scan"
OUT_DIR = "checkpoint/leafy_pca_lda"
# Cross-session CV below always holds out whichever scan session is most
# recent -- the actual production scenario (a session never seen during
# training) -- not a blind split of the full pool. Update this whenever a
# newer scanning session is folded into the CSV (merge_*.py scripts tag
# each one's rows with a distinct "source" value).
NEWEST_SOURCE = "new-test-data-2026-11-25"
FLAMENIR_N_PCA = 80

RECIPES = {
    # preprocess_fn(X_raw) -> features; make_clf() -> a fresh untrained classifier.
    "OCEANFX": {
        "preprocess": "plain",
        "preprocess_fn": lambda X: snv(savgol_smooth(X)).astype(np.float32),
        "n_pca": None,  # picked via pick_best_n_pca
        "make_clf": LinearDiscriminantAnalysis,
    },
    "FLAMENIR": {
        "preprocess": "derivative",
        "preprocess_fn": lambda X: savgol_filter(snv(X), window_length=11, polyorder=2, deriv=1, axis=1).astype(np.float32),
        "n_pca": FLAMENIR_N_PCA,
        "make_clf": lambda: RandomForestClassifier(n_estimators=300, random_state=SEED),
    },
}


def pick_best_n_pca(X, y, make_clf):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    max_n_pca = min(X.shape[1], int(X.shape[0] * 0.8)) - 1
    best = None
    for n_pca in N_PCA_CANDIDATES:
        if n_pca >= max_n_pca:
            continue
        model = make_pipeline(StandardScaler(), PCA(n_components=n_pca, random_state=SEED), make_clf())
        scores = cross_val_score(model, X, y, cv=skf)
        if best is None or scores.mean() > best[0]:
            best = (scores.mean(), scores.std(), n_pca)
    return best


def cross_session_cv(X_raw_old, y_old, X_raw_new, y_new, recipe):
    """5-fold CV over the NEWEST session only (the "old" session(s) are
    always fully in train) -- estimates accuracy on a scanning session
    never seen at all during training/fitting, the actual production
    scenario, rather than a blind CV over the full pool."""
    preprocess_fn, make_clf, n_pca = recipe["preprocess_fn"], recipe["make_clf"], recipe["n_pca"]
    X_old = preprocess_fn(X_raw_old)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    accs = []
    for tr_idx, te_idx in skf.split(X_raw_new, y_new):
        X_train = np.concatenate([X_old, preprocess_fn(X_raw_new[tr_idx])])
        y_train = np.concatenate([y_old, y_new[tr_idx]])
        X_test = preprocess_fn(X_raw_new[te_idx])
        this_n_pca = n_pca or pick_best_n_pca(X_train, y_train, make_clf)[2]
        pipe = make_pipeline(StandardScaler(), PCA(n_components=this_n_pca, random_state=SEED), make_clf())
        pipe.fit(X_train, y_train)
        accs.append((pipe.predict(X_test) == y_new[te_idx]).mean())
    return float(np.mean(accs)), float(np.std(accs))


def run(machine: str):
    recipe = RECIPES[machine]
    csv_path = f"{LEAFY_SCAN_DIR}/{machine}/processed/ALL.csv"
    df = pd.read_csv(csv_path)
    sub = df[df["position"] == "Lá 2"]
    w_cols = [c for c in sub.columns if c.startswith("w_")]
    is_newest = sub.get("source", "original") == NEWEST_SOURCE

    X_old_raw = sub.loc[~is_newest, w_cols].values.astype(np.float32)
    y_old = sub.loc[~is_newest, "category"].values
    X_new_raw = sub.loc[is_newest, w_cols].values.astype(np.float32)
    y_new = sub.loc[is_newest, "category"].values
    print(f"{machine}: n={len(sub)} (older session(s)={len(y_old)}, {NEWEST_SOURCE}={len(y_new)})")
    print("  " + sub["category"].value_counts().to_string().replace("\n", "\n  "))

    if len(y_new) > 0:
        acc_mean, acc_std = cross_session_cv(X_old_raw, y_old, X_new_raw, y_new, recipe)
        print(f"  cross-session 5-fold CV (held-out {NEWEST_SOURCE} folds): "
              f"{acc_mean*100:.2f}% (+/- {acc_std*100:.2f})")
    else:
        acc_mean = acc_std = float("nan")

    # Shipped model: fit on everything (outlier filter still applies as a
    # data-quality step, same as before merging in a second session).
    X_raw_all = sub[w_cols].values.astype(np.float32)
    y_all = sub["category"].values
    _, keep = preprocess_spectra(X_raw_all, apply_savgol_snv=False)  # outlier filter only
    X_all = recipe["preprocess_fn"](X_raw_all[keep])
    y_all = y_all[keep]

    n_pca = recipe["n_pca"] or pick_best_n_pca(X_all, y_all, recipe["make_clf"])[2]
    pipeline = make_pipeline(StandardScaler(), PCA(n_components=n_pca, random_state=SEED), recipe["make_clf"]())
    pipeline.fit(X_all, y_all)

    out_dir = f"{OUT_DIR}/{machine}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/model.joblib"
    joblib.dump({
        "pipeline": pipeline,
        "classes": sorted(set(y_all)),
        "preprocess": recipe["preprocess"],
        "n_pca": n_pca,
        "cv_accuracy": (acc_mean, acc_std),
    }, out_path)
    print(f"  Wrote {out_path}")


if __name__ == "__main__":
    machines = sys.argv[1:] or ["FLAMENIR", "OCEANFX"]
    for m in machines:
        run(m)
