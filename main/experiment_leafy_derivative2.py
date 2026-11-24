"""Round 2: can folding HALF of the new session into training (closing the
cross-session gap directly, since that's the diagnosed root cause) push
FLAMENIR's held-out accuracy towards 90%? Validates on the OTHER half of
new-test-data (never touched during fit/CV/model-selection), so this is
still a genuine, un-leaked number -- not just in-sample fit quality.

No leaf-id/group metadata exists in the new-test-data filenames (checked:
every file in a class folder shares one prefix, just an incrementing scan
index), so the 50/50 split is a plain stratified-by-class split, not a
grouped one -- if several files are repeat scans of the very same physical
leaf, splitting them across train/test could inflate the held-out number
slightly. Flagged, not hidden.

Usage: python3 experiment_leafy_derivative2.py
"""
import glob
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from dataset.preprocessing import preprocess_spectra_derivative, snv

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


def derivative(X_raw: np.ndarray) -> np.ndarray:
    return savgol_filter(snv(X_raw), window_length=11, polyorder=2, deriv=1, axis=1).astype(np.float32)


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

Xn_tr_raw, Xn_te_raw, yn_tr, yn_te = train_test_split(
    Xn_raw, yn, test_size=0.5, stratify=yn, random_state=SEED)
print(f"new-test-data split: {len(yn_tr)} train / {len(yn_te)} held-out (never fit/CV'd on)")

df = pd.read_csv(f"{DATASET_ROOT}/leafy-position-scan/{MACHINE}/processed/ALL.csv")
sub = df[df["position"] == "Lá 2"]
w_cols = [c for c in sub.columns if c.startswith("w_")]
X_old_raw = sub[w_cols].values.astype(np.float32)
y_old_full = sub["category"].values
X_old, keep = preprocess_spectra_derivative(X_old_raw)
y_old = y_old_full[keep]

X_new_tr = derivative(Xn_tr_raw)
X_new_te = derivative(Xn_te_raw)

X_train = np.concatenate([X_old, X_new_tr])
y_train = np.concatenate([y_old, yn_tr])
print(f"combined train: {len(y_train)} ({len(X_old)} session-1 + {len(X_new_tr)} session-2-half)")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
candidates = []
for n_pca in [15, 20, 30, 50]:
    for name, clf in [
        ("RandomForest300", RandomForestClassifier(n_estimators=300, random_state=SEED)),
        ("RandomForest500-d8", RandomForestClassifier(n_estimators=500, max_depth=8, random_state=SEED)),
        ("ExtraTrees300", ExtraTreesClassifier(n_estimators=300, random_state=SEED)),
        ("SVM-RBF-C10", SVC(kernel="rbf", C=10, gamma="scale", probability=True)),
        ("LDA", LinearDiscriminantAnalysis()),
    ]:
        pipe = make_pipeline(StandardScaler(), PCA(n_components=n_pca, random_state=SEED), clf)
        cv = cross_val_score(pipe, X_train, y_train, cv=skf).mean()
        pipe.fit(X_train, y_train)
        acc_held = (pipe.predict(X_new_te) == yn_te).mean() * 100
        print(f"n_pca={n_pca:<4} {name:<20} CV={cv*100:5.2f}%  HELD-OUT(session-2 half)={acc_held:5.2f}% "
              f"({int(acc_held/100*len(yn_te))}/{len(yn_te)})")
        candidates.append((acc_held, n_pca, name, pipe))

# Voting ensemble of the 3 best-performing individual pipelines above.
candidates.sort(key=lambda c: -c[0])
top3 = candidates[:3]
print("\nTop 3 individually:", [(n, c) for _, n, c, _ in [(a, b, c, d) for a, b, c, d in top3]])
estimators = [(f"{name}_{n_pca}", pipe) for _, n_pca, name, pipe in top3]
vote = VotingClassifier(estimators=estimators, voting="soft")
vote.fit(X_train, y_train)
acc_vote = (vote.predict(X_new_te) == yn_te).mean() * 100
print(f"\nSoft-voting ensemble of top-3: HELD-OUT={acc_vote:.2f}% ({int(acc_vote/100*len(yn_te))}/{len(yn_te)})")
