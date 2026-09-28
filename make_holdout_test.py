"""Set aside a fixed, independent test set (N_HOLDOUT samples per machine) from
each machine's ALL.csv, before any training/cross-validation touches the data.

For each machine this
  1. picks N_HOLDOUT rows stratified by food category (proportional
     allocation, largest-remainder rounding, fixed seed),
  2. writes them, byte-for-byte, to {machine}/HOLDOUT_{N}.csv plus a small
     .meta.json recording the seed and the 0-based row indices they had in
     the pre-split ALL.csv,
  3. rewrites {machine}/ALL.csv WITHOUT those rows, so every existing engine
     (which all read ALL.csv) automatically trains/validates on data that
     never includes the held-out samples,
  4. keeps the pre-split file as ALL.csv.bak-preholdout-<timestamp>.

Rows are moved as raw lines (no float re-formatting), so values are exactly
what was in ALL.csv. Refuses to run if HOLDOUT_{N}.csv already exists, so
the held-out set can never be silently re-drawn from an already-split file.

Usage: python3 make_holdout_test.py [FLAMENIR OCEANFX]
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

DATASET_ROOT = os.environ.get("DATASET_ROOT", "../all-dataset/Danang-NIR")
N_HOLDOUT = 100
SEED = 42


def allocate(counts: pd.Series, n_total: int) -> pd.Series:
    quota = counts / counts.sum() * n_total
    alloc = np.floor(quota).astype(int)
    remainder = n_total - int(alloc.sum())
    for cat in (quota - alloc).sort_values(ascending=False).index[:remainder]:
        alloc[cat] += 1
    return alloc


def split_machine(machine: str) -> None:
    d = f"{DATASET_ROOT}/{machine}"
    all_path = f"{d}/ALL.csv"
    holdout_path = f"{d}/HOLDOUT_{N_HOLDOUT}.csv"
    meta_path = f"{d}/HOLDOUT_{N_HOLDOUT}.meta.json"
    if os.path.exists(holdout_path):
        raise SystemExit(f"{holdout_path} already exists -- refusing to re-draw the held-out set.")

    cats = pd.read_csv(all_path, usecols=["category"])["category"]
    n_rows = len(cats)
    rng = np.random.default_rng(SEED)
    alloc = allocate(cats.value_counts(), N_HOLDOUT)
    chosen = []
    for cat, k in alloc.items():
        idx = np.flatnonzero(cats.values == cat)
        chosen.extend(rng.choice(idx, size=int(k), replace=False).tolist())
    chosen = sorted(chosen)
    assert len(chosen) == N_HOLDOUT
    chosen_lines = {i + 1 for i in chosen}  # +1: line 0 is the header

    tmp_path = all_path + ".tmp"
    n_out = n_held = 0
    with open(all_path, "rb") as fin, open(tmp_path, "wb") as fout, open(holdout_path, "wb") as fh:
        header = fin.readline()
        fout.write(header)
        fh.write(header)
        for i, line in enumerate(fin, start=1):
            if i in chosen_lines:
                fh.write(line)
                n_held += 1
            else:
                fout.write(line)
                n_out += 1
    assert n_held == N_HOLDOUT and n_out == n_rows - N_HOLDOUT, (n_held, n_out, n_rows)

    ts = time.strftime("%Y%m%d_%H%M%S")
    os.rename(all_path, f"{all_path}.bak-preholdout-{ts}")
    os.rename(tmp_path, all_path)

    with open(meta_path, "w") as f:
        json.dump({
            "machine": machine, "seed": SEED, "n_holdout": N_HOLDOUT,
            "rows_in_pre_split_all_csv": chosen, "n_rows_pre_split": n_rows,
            "per_category": {k: int(v) for k, v in alloc.items()},
        }, f, ensure_ascii=False, indent=2)
    print(f"{machine}: {n_rows} -> {n_out} train-pool rows + {n_held} held out "
          f"(per category: {dict(alloc)})", flush=True)


if __name__ == "__main__":
    for m in (sys.argv[1:] or ["FLAMENIR", "OCEANFX"]):
        split_machine(m)
