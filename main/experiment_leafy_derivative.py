"""One-off experiment: does SNV + 1st-derivative preprocessing (dataset.
preprocessing.preprocess_spectra_derivative -- already in the codebase,
built for exactly this leafy-green confusability problem, but unused by
the currently deployed train_leafy_pca_lda.py which uses plain SG+SNV)
generalize better than the deployed model to the new-test-data holdout,
especially for FLAMENIR (deployed model: 21/84 -> ~25% on new-test-data,
vs ~85% in-session CV -- a large session-to-session drop that plain SNV,
a per-sample scatter correction, doesn't fix since it's whole-curve-level,
not local-shape).

This is a genuine held-out test: train only on leafy-position-scan (the
original "Lá 2" collection session), evaluate only on new-test-data (a
separate, later collection session) -- never seen during fitting or PCA
n_components selection.

Usage: python3 experiment_leafy_derivative.py
"""
import glob
import os

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
from sklearn.svm import SVC

from dataset.preprocessing import preprocess_spectra, preprocess_spectra_derivative, savgol_smooth, snv

load_dotenv()
DATASET_ROOT = os.environ["DATASET_ROOT"]
N_PCA_CANDIDATES = [10, 20, 30, 50, 80, 120, 180]
SEED = 42

FOLDER_TO_LABEL = {"Caibex": "Cải Bẹ Xanh", "Caithi": "Cải Thìa", "Montoi": "Mồng Tơi", "Xalach": "Xà Lách"}
EXCLUDE_BASENAMES = {
    "DL_Montoi_FLMN032701__1__00001.txt", "HC2_Montoi_FLMN032701__5__00005.txt",
    "CT_Caibex_OFX034251__5__00005.txt", "CT_Caibex_OFX034251__6__00006.txt",
}


def load_txt_spectrum(path: str) -> np.ndarray:
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    start = next(i for i, l in enumerate(lines) if "Begin Spectral Data" in l) + 1
    return np.array([float(l.split()[1]) for l in lines[start:] if l.strip()], dtype=np.float32)


def load_new_test_data(machine: str, root: str):
    X_raw, y = [], []
    for folder, label in FOLDER_TO_LABEL.items():
        for fp in sorted(glob.glob(f"{root}/{machine}/{folder}/*.txt")):
            if os.path.basename(fp) in EXCLUDE_BASENAMES:
                continue
            X_raw.append(load_txt_spectrum(fp))
            y.append(label)
    return np.stack(X_raw), np.array(y)


def apply_inference_style(X_raw: np.ndarray, mode: str) -> np.ndarray:
    """Mirrors deploy_utils.py's convention: per-sample steps only, no
    dataset-level outlier filter (not meaningful for a single live batch)."""
    if mode == "plain":
        return snv(savgol_smooth(X_raw)).astype(np.float32)
    if mode == "derivative":
        return savgol_filter(snv(X_raw), window_length=11, polyorder=2, deriv=1, axis=1).astype(np.float32)
    raise ValueError(mode)


def pick_best_n_pca(X, y, candidates):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    max_n_pca = min(X.shape[1], int(X.shape[0] * 0.8)) - 1
    best = None
    for n_pca in candidates:
        if n_pca >= max_n_pca:
            continue
        model = make_pipeline(StandardScaler(), PCA(n_components=n_pca, random_state=SEED),
                               LinearDiscriminantAnalysis())
        scores = cross_val_score(model, X, y, cv=skf)
        if best is None or scores.mean() > best[0]:
            best = (scores.mean(), scores.std(), n_pca)
    return best


def run(machine: str):
    print(f"\n{'='*70}\n{machine}\n{'='*70}")
    train_csv = f"{DATASET_ROOT}/leafy-position-scan/{machine}/processed/ALL.csv"
    df = pd.read_csv(train_csv)
    sub = df[df["position"] == "Lá 2"]
    w_cols = [c for c in sub.columns if c.startswith("w_")]
    X_raw_train = sub[w_cols].values.astype(np.float32)
    y_train_full = sub["category"].values

    new_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "new-test-data")
    X_raw_new, y_new = load_new_test_data(machine, new_root)
    print(f"train (leafy-position-scan, Lá 2): n={len(X_raw_train)}  "
          f"new-test-data (holdout session): n={len(X_raw_new)}")

    for mode in ("plain", "derivative"):
        if mode == "plain":
            X_train, keep = preprocess_spectra(X_raw_train)
        else:
            X_train, keep = preprocess_spectra_derivative(X_raw_train)
        y_train = y_train_full[keep]
        X_new = apply_inference_style(X_raw_new, mode)

        acc_cv, std_cv, n_pca = pick_best_n_pca(X_train, y_train, N_PCA_CANDIDATES)
        pipe = make_pipeline(StandardScaler(), PCA(n_components=n_pca, random_state=SEED),
                              LinearDiscriminantAnalysis())
        pipe.fit(X_train, y_train)
        pred_new = pipe.predict(X_new)
        acc_new = (pred_new == y_new).mean() * 100

        print(f"  [{mode:<10}] in-session 5-fold CV={acc_cv*100:.2f}%±{std_cv*100:.2f} "
              f"(n_pca={n_pca})  ->  NEW-SESSION holdout acc={acc_new:.2f}%  "
              f"({(pred_new == y_new).sum()}/{len(y_new)})")

    # Also try: same derivative features, but a less-linear classifier
    # (RandomForest, SVM-RBF) in case LDA's linear boundary itself (not the
    # feature space) is the fragile part under session drift.
    X_train_d, keep = preprocess_spectra_derivative(X_raw_train)
    y_train_d = y_train_full[keep]
    X_new_d = apply_inference_style(X_raw_new, "derivative")
    for name, clf in [
        ("RandomForest", RandomForestClassifier(n_estimators=300, random_state=SEED)),
        ("SVM-RBF", SVC(kernel="rbf", C=10, gamma="scale")),
    ]:
        pipe = make_pipeline(StandardScaler(), PCA(n_components=30, random_state=SEED), clf)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        cv_scores = cross_val_score(pipe, X_train_d, y_train_d, cv=skf)
        pipe.fit(X_train_d, y_train_d)
        pred_new = pipe.predict(X_new_d)
        acc_new = (pred_new == y_new).mean() * 100
        print(f"  [derivative+{name:<12}] in-session CV={cv_scores.mean()*100:.2f}% "
              f"-> NEW-SESSION holdout acc={acc_new:.2f}% ({(pred_new == y_new).sum()}/{len(y_new)})")


if __name__ == "__main__":
    for m in ["FLAMENIR", "OCEANFX"]:
        run(m)
