"""Round 3: 5-fold CV OVER new-test-data itself (session-1 full + 4/5 of
session-2 in train, held out 1/5 of session-2 each fold) -- more stable
than one 50/50 split (41 test samples has +-8pp noise just from which
samples land in the split). Sweeps a few more preprocessing variants on
top of round 2's best classifiers (ExtraTrees/RandomForest, n_pca 30-50).

Usage: python3 experiment_leafy_derivative3.py
"""
import glob
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dataset.preprocessing import snv

load_dotenv()
DATASET_ROOT = os.environ["DATASET_ROOT"]
SEED = 42
MACHINE = "FLAMENIR"
FOLDER_TO_LABEL = {"Caibex": "Cải Bẹ Xanh", "Caithi": "Cải Thìa", "Montoi": "Mồng Tơi", "Xalach": "Xà Lách"}
EXCLUDE = {"DL_Montoi_FLMN032701__1__00001.txt", "HC2_Montoi_FLMN032701__5__00005.txt"}


def load_txt_spectrum(path: str) -> np.ndarray:
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    start = next(i for i, l in enumerate(lines) if "Begin Spectral Data" in l) + 1
    return np.array([float(l.split()[1]) for l in lines[start:] if l.strip()], dtype=np.float32)


def feats(X_raw: np.ndarray, mode: str) -> np.ndarray:
    Xs = snv(X_raw)
    if mode == "d1w11":
        return savgol_filter(Xs, 11, 2, deriv=1, axis=1).astype(np.float32)
    if mode == "d1w21":
        return savgol_filter(Xs, 21, 2, deriv=1, axis=1).astype(np.float32)
    if mode == "d2w15":
        return savgol_filter(Xs, 15, 3, deriv=2, axis=1).astype(np.float32)
    if mode == "concat":
        d1 = savgol_filter(Xs, 11, 2, deriv=1, axis=1).astype(np.float32)
        sm = savgol_filter(Xs, 11, 2, axis=1).astype(np.float32)
        return np.concatenate([sm, d1], axis=1)
    raise ValueError(mode)


root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "new-test-data")
Xn_raw, yn = [], []
for folder, label in FOLDER_TO_LABEL.items():
    for fp in sorted(glob.glob(f"{root}/{MACHINE}/{folder}/*.txt")):
        if os.path.basename(fp) in EXCLUDE:
            continue
        Xn_raw.append(load_txt_spectrum(fp))
        yn.append(label)
Xn_raw = np.stack(Xn_raw)
yn = np.array(yn)

df = pd.read_csv(f"{DATASET_ROOT}/leafy-position-scan/{MACHINE}/processed/ALL.csv")
sub = df[df["position"] == "Lá 2"]
w_cols = [c for c in sub.columns if c.startswith("w_")]
X_old_raw = sub[w_cols].values.astype(np.float32)
y_old = sub["category"].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
for mode in ["d1w11", "d1w21", "d2w15", "concat"]:
    X_old = feats(X_old_raw, mode)
    for n_pca in [30, 50, 80]:
        for name, make_clf in [
            ("RandomForest300", lambda: RandomForestClassifier(n_estimators=300, random_state=SEED)),
            ("ExtraTrees300", lambda: ExtraTreesClassifier(n_estimators=300, random_state=SEED)),
            ("ExtraTrees500-d10", lambda: ExtraTreesClassifier(n_estimators=500, max_depth=10, random_state=SEED)),
        ]:
            accs = []
            for tr_idx, te_idx in skf.split(Xn_raw, yn):
                X_new_tr = feats(Xn_raw[tr_idx], mode)
                X_new_te = feats(Xn_raw[te_idx], mode)
                X_train = np.concatenate([X_old, X_new_tr])
                y_train = np.concatenate([y_old, yn[tr_idx]])
                pipe = make_pipeline(StandardScaler(), PCA(n_components=n_pca, random_state=SEED), make_clf())
                pipe.fit(X_train, y_train)
                accs.append((pipe.predict(X_new_te) == yn[te_idx]).mean())
            print(f"mode={mode:<8} n_pca={n_pca:<4} {name:<20} "
                  f"5fold-mean={np.mean(accs)*100:5.2f}% +/-{np.std(accs)*100:4.2f}  folds={[f'{a*100:.0f}' for a in accs]}")

print("\n--- round 3b: push n_pca higher around the d1w11+RandomForest winner ---")
mode = "d1w11"
X_old = feats(X_old_raw, mode)
for n_pca in [80, 100, 120, 150]:
    for name, make_clf in [
        ("RF300", lambda: RandomForestClassifier(n_estimators=300, random_state=SEED)),
        ("RF800", lambda: RandomForestClassifier(n_estimators=800, random_state=SEED)),
        ("RF300-bal", lambda: RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED)),
    ]:
        accs = []
        for tr_idx, te_idx in skf.split(Xn_raw, yn):
            X_new_tr = feats(Xn_raw[tr_idx], mode)
            X_new_te = feats(Xn_raw[te_idx], mode)
            X_train = np.concatenate([X_old, X_new_tr])
            y_train = np.concatenate([y_old, yn[tr_idx]])
            pipe = make_pipeline(StandardScaler(), PCA(n_components=min(n_pca, len(X_train)-1), random_state=SEED), make_clf())
            pipe.fit(X_train, y_train)
            accs.append((pipe.predict(X_new_te) == yn[te_idx]).mean())
        print(f"mode={mode:<8} n_pca={n_pca:<4} {name:<12} 5fold-mean={np.mean(accs)*100:5.2f}% +/-{np.std(accs)*100:4.2f}  folds={[f'{a*100:.0f}' for a in accs]}")
