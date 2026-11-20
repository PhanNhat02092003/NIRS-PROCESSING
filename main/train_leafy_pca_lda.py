"""Trains the PCA+LDA model that resolves which of the 4 leafy-green
categories (Xa Lach, Cai Be Xanh, Cai Thia, Mong Toi) a sample actually is,
for use when the production 6-class GuidedDCNet classifier (see
food_classification_6class.py) predicts the merged "Rau La" label.

Trained on {DATASET_ROOT}/leafy-position-scan/{machine}/processed/ALL.csv,
position == "La 2" only
(the inner leaf face -- found to give the best separation among the 4
categories in analyze_leafy_positions.py: ~97% 5-fold CV accuracy on both
machines, vs ~85-95% for La 1 and worse for Cuong). This means the
deployed scanning protocol for leafy-green samples must use the La 2
position for this model to be accurate -- a new operational requirement,
not just an internal implementation detail.

For each machine: SG+SNV preprocessing (dataset.preprocessing.
preprocess_spectra, full wavelength resolution -- NOT the BIN=8 reduction
GuidedDCNet uses for OCEANFX), best PCA n_components picked by 5-fold CV
over N_PCA_CANDIDATES (same method as analyze_leafy_positions.py), then
the final StandardScaler+PCA+LDA pipeline is refit on the FULL La-2 leafy
set (no held-out split -- this is the model that ships, not an accuracy
estimate; the CV accuracy from the selection step is printed as the best
available estimate of out-of-sample accuracy).

Usage: python3 train_leafy_pca_lda.py [FLAMENIR OCEANFX]
Output: checkpoint/leafy_pca_lda/{machine}/model.joblib, a dict:
  {"pipeline": sklearn Pipeline (StandardScaler, PCA, LDA),
   "classes": sorted list of the 4 category names,
   "n_pca": chosen PCA n_components,
   "cv_accuracy": (mean, std) from the selection step}
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

from dataset.preprocessing import preprocess_spectra

load_dotenv()

N_PCA_CANDIDATES = [10, 20, 30, 50, 80, 120, 180]
SEED = 42
LEAFY_SCAN_DIR = f"{os.environ['DATASET_ROOT']}/leafy-position-scan"
OUT_DIR = "checkpoint/leafy_pca_lda"


def pick_best_n_pca(X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    max_n_pca = min(X.shape[1], int(X.shape[0] * 0.8)) - 1
    best = None
    for n_pca in N_PCA_CANDIDATES:
        if n_pca >= max_n_pca:
            continue
        model = make_pipeline(StandardScaler(), PCA(n_components=n_pca, random_state=SEED),
                               LinearDiscriminantAnalysis())
        scores = cross_val_score(model, X, y, cv=skf)
        if best is None or scores.mean() > best[0]:
            best = (scores.mean(), scores.std(), n_pca)
    return best


def run(machine: str):
    csv_path = f"{LEAFY_SCAN_DIR}/{machine}/processed/ALL.csv"
    df = pd.read_csv(csv_path)
    sub = df[df["position"] == "Lá 2"]
    w_cols = [c for c in sub.columns if c.startswith("w_")]
    X_raw = sub[w_cols].values.astype(np.float32)
    y = sub["category"].values

    X, keep = preprocess_spectra(X_raw)
    y = y[keep]

    acc_mean, acc_std, n_pca = pick_best_n_pca(X, y)
    print(f"{machine}: n={len(y)}, n_features={X.shape[1]}, best PCA n_components={n_pca}, "
          f"CV accuracy={acc_mean*100:.2f}% (+/- {acc_std*100:.2f})")
    print("  " + sub["category"].value_counts().to_string().replace("\n", "\n  "))

    pipeline = make_pipeline(StandardScaler(), PCA(n_components=n_pca, random_state=SEED),
                              LinearDiscriminantAnalysis())
    pipeline.fit(X, y)

    out_dir = f"{OUT_DIR}/{machine}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/model.joblib"
    joblib.dump({
        "pipeline": pipeline,
        "classes": sorted(set(y)),
        "n_pca": n_pca,
        "cv_accuracy": (float(acc_mean), float(acc_std)),
    }, out_path)
    print(f"  Wrote {out_path}")


if __name__ == "__main__":
    machines = sys.argv[1:] or ["FLAMENIR", "OCEANFX"]
    for m in machines:
        run(m)
