"""Shrinks each machine's ALL.csv by collapsing massively repeated
same-label groups to a smaller, spectrally diverse subset.

Why: groupby(category + the 19-substance concentration vector) shows
~320-340 distinct label combinations per machine but ~335k/380k total
rows -- i.e. an average of ~1000-1100 rows (up to ~5800) share the exact
same label. There is no session/date/batch id in ALL.csv to confirm it
directly, but this is almost certainly one prepared batch scanned
hundreds of times, not hundreds of independent physical specimens.
Exact-duplicate check (dedup_conflicting_spectra.py) found 0 bit-identical
rows, and nearest-neighbor distances in SG+SNV space within the largest
group are continuously spread (not clumped near 0), so this isn't
trivial repeat-scan duplication either -- it's just heavy, fine-grained
oversampling of a handful of batches, which both (a) adds training cost
without adding real diversity, and (b) risks a model learning batch-level
acquisition quirks (lamp drift, session temperature, ...) as a shortcut
for the label, since ordinary row-level StratifiedKFold can split one
batch's 1000+ near-identical-context rows across train and validation.

What this does, per machine:
  1. Group rows by (category, 19-substance concentration vector).
  2. Groups with <= CAP rows are kept unchanged.
  3. Groups with > CAP rows: run MiniBatchKMeans (k=CAP) on the group's
     Savitzky-Golay + SNV preprocessed spectrum, then keep -- for each
     cluster -- the single real row closest to that cluster's centroid.
     This spreads the kept rows across the group's actual spectral
     diversity instead of taking the first N rows (which could all be
     one contiguous scanning run) or a uniform random sample (which
     would keep the same redundancy ratio, just smaller).
  4. Every kept row is tagged with `_group_id` (a stable integer per
     label group), so a later change to food_classification.py /
     stage1_detection.py / stage2_safety.py can swap StratifiedKFold for
     GroupKFold on this column to guarantee no single batch straddles
     train and validation -- not done here, this script only prepares
     the data.

Output: {DATASET_ROOT}/{machine}/ALL_downsampled.csv (raw ALL.csv is
never modified). Prints before/after row counts per machine.

Usage: python3 downsample_training_data.py [--cap N] [FLAMENIR OCEANFX]
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

from dataset.preprocessing import preprocess_spectra

DATASET_ROOT = os.environ.get("DATASET_ROOT", "../all-dataset/Danang-NIR")
SUBSTANCES = [
    'Thiamethoxam', 'Permethrin', 'Metalaxyl', 'Azoxystrobin',
    'Imidaclopird', 'Difenoconazole', 'Cypermethrin', 'Cyhalothrin',
    'Chlorantraniliprol', 'Chlopyrifos Methyl', 'Emamectin benzoate',
    'Chlorothalonil', 'Triadimefon', 'Cyantraniliprole', 'Flutolanil',
    'Indoxacarb', 'Abamectin', 'Propamocarb.HCL', 'Chlothianidin',
]
DEFAULT_CAP = 200
SEED = 42


def representative_subset(spectra: np.ndarray, cap: int, seed: int) -> np.ndarray:
    """Returns row indices (into `spectra`) of the `cap` rows closest to
    each of `cap` MiniBatchKMeans cluster centroids fit on the SG+SNV
    preprocessed spectrum."""
    pp, _ = preprocess_spectra(spectra)
    km = MiniBatchKMeans(n_clusters=cap, random_state=seed, n_init=3, batch_size=1024)
    labels = km.fit_predict(pp)
    chosen = np.empty(cap, dtype=np.int64)
    for c in range(cap):
        members = np.flatnonzero(labels == c)
        if len(members) == 0:
            continue
        d = np.linalg.norm(pp[members] - km.cluster_centers_[c], axis=1)
        chosen[c] = members[d.argmin()]
    # a cluster can be empty (rare, MiniBatchKMeans); drop its slot
    valid = np.array([c for c in range(cap) if len(np.flatnonzero(labels == c)) > 0])
    return chosen[valid]


def run(machine: str, cap: int) -> None:
    d = f"{DATASET_ROOT}/{machine}"
    df = pd.read_csv(f"{d}/ALL.csv")
    w_cols = [c for c in df.columns if c.startswith("w_")]
    n = len(df)

    group_key = df[["category"] + SUBSTANCES].astype(str).agg("|".join, axis=1)
    group_id, uniques = pd.factorize(group_key)
    df["_group_id"] = group_id

    keep_idx = []
    n_groups_capped = 0
    for gid in range(len(uniques)):
        members = np.flatnonzero(group_id == gid)
        if len(members) <= cap:
            keep_idx.extend(members.tolist())
            continue
        n_groups_capped += 1
        spectra = df.iloc[members][w_cols].values.astype(np.float32)
        rel = representative_subset(spectra, cap, SEED)
        keep_idx.extend(members[rel].tolist())
        print(f"  {machine} group {gid} ({df.iloc[members[0]]['category']}): "
              f"{len(members)} -> {len(rel)}", flush=True)

    keep_idx = sorted(keep_idx)
    out = df.iloc[keep_idx].drop(columns=[]).copy()
    out_path = f"{d}/ALL_downsampled.csv"
    out.to_csv(out_path, index=False)
    print(f"{machine}: {n} rows ({len(uniques)} label groups, {n_groups_capped} capped) "
          f"-> {len(out)} rows -> wrote {out_path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=DEFAULT_CAP)
    ap.add_argument("machines", nargs="*", default=["FLAMENIR", "OCEANFX"])
    args = ap.parse_args()
    for m in args.machines:
        run(m, args.cap)
