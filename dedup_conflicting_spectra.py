"""Clean duplicated scans out of each machine's ALL.csv.

Two spectra are treated as the same scan when all wavelength values are equal
at float32 precision (~7 significant digits) -- independent re-scans never
coincide that closely, so such pairs are the same scan recorded twice.

Per group of identical spectra:
  * different food categories                     -> drop the whole group
  * same category, any of the 19 concentration
    values differs (label conflict)               -> drop the whole group
  * same category, all 19 values identical        -> keep the first row only
Any group whose spectrum also occurs in HOLDOUT_100.csv is dropped from
ALL.csv entirely, so the held-out set stays independent of the training pool.

Default is a dry run. With --write the original ALL.csv is renamed to
ALL.csv.bak-predup-<timestamp> and the cleaned file takes its place (raw
lines are copied, no float re-formatting).

Usage: python3 dedup_conflicting_spectra.py [--write] [FLAMENIR OCEANFX]
"""
import os
import sys
import time

import numpy as np
import pandas as pd

DATASET_ROOT = os.environ.get("DATASET_ROOT", "../all-dataset/Danang-NIR")
SUBSTANCES = [
    'Thiamethoxam', 'Permethrin', 'Metalaxyl', 'Azoxystrobin',
    'Imidaclopird', 'Difenoconazole', 'Cypermethrin', 'Cyhalothrin',
    'Chlorantraniliprol', 'Chlopyrifos Methyl', 'Emamectin benzoate',
    'Chlorothalonil', 'Triadimefon', 'Cyantraniliprole', 'Flutolanil',
    'Indoxacarb', 'Abamectin', 'Propamocarb.HCL', 'Chlothianidin',
]


def spectrum_hash(df, w_cols):
    return pd.util.hash_pandas_object(df[w_cols].astype(np.float32), index=False).values


def run(machine, write):
    d = f"{DATASET_ROOT}/{machine}"
    df = pd.read_csv(f"{d}/ALL.csv")
    w_cols = [c for c in df.columns if c.startswith("w_")]
    hold = pd.read_csv(f"{d}/HOLDOUT_100.csv")
    df["_h"] = spectrum_hash(df, w_cols)
    hold_h = set(spectrum_hash(hold, w_cols))
    n = len(df)

    size = df.groupby("_h")["_h"].transform("size")
    n_cat = df.groupby("_h")["category"].transform("nunique")
    # number of distinct concentration vectors inside the group
    conc_hash = pd.util.hash_pandas_object(df[SUBSTANCES].round(6), index=False).values
    df["_c"] = conc_hash
    n_vec = df.groupby("_h")["_c"].transform("nunique")

    is_dup = size > 1
    drop_cat = is_dup & (n_cat > 1)
    drop_conflict = is_dup & (n_cat == 1) & (n_vec > 1)
    drop_holdout = df["_h"].isin(hold_h)
    drop_group = drop_cat | drop_conflict | drop_holdout
    # agreeing duplicates: keep first, drop the rest
    redundant = is_dup & ~drop_group & df.duplicated("_h", keep="first")
    drop = drop_group | redundant

    biggest = size.max()
    big_rows = df[size == biggest]
    print(f"{machine}: {n} rows; rows in duplicate groups {int(is_dup.sum())} "
          f"({is_dup.mean():.3%}); largest group {biggest} rows "
          f"(categories={big_rows.category.nunique()}, "
          f"spectrum std={big_rows[w_cols].iloc[0].std():.3g})")
    print(f"  drop: different-category groups {int(drop_cat.sum())} | label-conflict groups "
          f"{int(drop_conflict.sum())} | overlap with hold-out {int(drop_holdout.sum())} "
          f"(union of group drops {int(drop_group.sum())}) | redundant agreeing copies {int(redundant.sum())}")
    print(f"  total dropped {int(drop.sum())} -> keep {n - int(drop.sum())}")
    # holdout rows that are themselves in ambiguous groups
    hh = pd.Series(spectrum_hash(hold, w_cols))
    amb = set(df.loc[drop_cat | drop_conflict, "_h"])
    print(f"  hold-out rows whose spectrum is in a conflicting/mixed-category group: {int(hh.isin(amb).sum())}")

    if not write:
        return
    drop_lines = {i + 1 for i in np.flatnonzero(drop.values)}
    del df
    ts = time.strftime("%Y%m%d_%H%M%S")
    src, tmp = f"{d}/ALL.csv", f"{d}/ALL.csv.tmp"
    kept = 0
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        fo.write(fi.readline())
        for i, line in enumerate(fi, start=1):
            if i not in drop_lines:
                fo.write(line)
                kept += 1
    assert kept == n - len(drop_lines), (kept, n, len(drop_lines))
    os.rename(src, f"{src}.bak-predup-{ts}")
    os.rename(tmp, src)
    print(f"  wrote {src} ({kept} rows); backup ALL.csv.bak-predup-{ts}", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    write = "--write" in args
    machines = [a for a in args if a != "--write"] or ["FLAMENIR", "OCEANFX"]
    for m in machines:
        run(m, write)
