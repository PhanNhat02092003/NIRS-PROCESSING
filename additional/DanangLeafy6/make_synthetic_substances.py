"""Model-sanity-check variant of the 6 chosen substances (Thiamethoxam,
Permethrin, Azoxystrobin, Difenoconazole, Cypermethrin, Chlothianidin):
keeps everything about the real data/{machine}/ALL.csv unchanged --
category (6-class, "Rau La" merged), _group_id, spectra, and each
substance's real presence/-1 pattern (so group counts, CV folds and the
per-substance filtering in RegressionNIRSDataset all behave exactly as
with the real data) -- but REPLACES the concentration value, wherever
present, with a synthetic signal (fixed random combo of real spectral
bands) + Gaussian noise calibrated to a chosen ceiling R2 (the R2 a
perfect model would get). safe_label is then RECOMPUTED (same MRL rule
as prepare_danangleafy6_dataset.py's compute_safe_label) from these new
synthetic concentrations, not left over from the real ones -- otherwise
the quality task would describe a safety reality that no longer matches
the regression targets in the same row.

This is NOT a pesticide-detectability claim (see this session's
permutation-test findings: real concentrations here have at best a weak,
substance-specific spectral correlation). It answers "can EBAR/SMART-NIR/
NirMACNet/XSpecMamba/LensNIR recover a real, controlled-strength signal
under these exact spectra, noise and group/presence structure" -- a
pipeline capability check, kept in a separate file so it's never mistaken
for a real-substance result.

Output: additional/DanangLeafy6/data/{machine}/ALL_synthetic.csv --
drop-in replacement for ALL.csv (same columns), so every existing
classification/quality/regression engine runs on it unchanged via a path
override.

Usage: python3 make_synthetic_substances.py [--ceiling R2] [FLAMENIR OCEANFX]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(_HERE)), "main"))
sys.path.insert(0, _HERE)
from dataset.preprocessing import preprocess_spectra
from prepare_danangleafy6_dataset import (LEAFY_MERGED_LABEL, LEAFY_REPRESENTATIVE,
                                          compute_safe_label, load_maps)

SUBSTANCES = ['Thiamethoxam', 'Permethrin', 'Azoxystrobin',
              'Difenoconazole', 'Cypermethrin', 'Chlothianidin']
SEED = 42
N_INFORMATIVE_BANDS = 16
DEFAULT_CEILING = 0.70


def make_target(X_pp: np.ndarray, ceiling_r2: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_bands = X_pp.shape[1]
    band_idx = rng.choice(n_bands, size=min(N_INFORMATIVE_BANDS, n_bands), replace=False)
    weights = rng.normal(size=len(band_idx))

    signal = X_pp[:, band_idx] @ weights
    signal = (signal - signal.mean()) / (signal.std() + 1e-8)

    noise_std = np.sqrt(max(1.0 / ceiling_r2 - 1.0, 0.0))
    noise = rng.normal(scale=noise_std, size=len(signal))

    y = signal + noise
    y = (y - y.min()) + 1.0  # positive, log1p-safe
    return y.astype(np.float32)


def run(machine: str, ceiling: float):
    # The real DanangLeafy6 engines (classification/quality/regression) all
    # read data/{machine}/ALL.csv -- the 6-class-merged ("Rau La"),
    # safe_label-added file built by prepare_danangleafy6_dataset.py -- not
    # the raw 9-category ALL_downsampled3.csv. Must match it so this stays
    # a true drop-in replacement (same category labels, same safe_label
    # column for the quality task, which is untouched/still real here).
    src = f"{_HERE}/data/{machine}/ALL.csv"
    df = pd.read_csv(src)
    w_cols = [c for c in df.columns if c.startswith("w_")]
    X_raw = df[w_cols].values.astype(np.float32)
    X_pp, keep = preprocess_spectra(X_raw)
    df = df.iloc[keep].reset_index(drop=True)

    out = df.copy()
    for i, substance in enumerate(SUBSTANCES):
        present = out[substance].values != -1
        seed = SEED + hash((machine, substance)) % 10_000
        synth_present = make_target(X_pp[present], ceiling, seed)
        new_col = out[substance].values.copy()
        new_col[present] = synth_present
        out[substance] = new_col
        print(f"  {machine} {substance:<16} n_present={present.sum():5d} "
              f"(real presence pattern kept, values replaced)")

    # safe_label is derived FROM the substance concentrations (vs MRL) --
    # now that those are synthetic, the real safe_label would describe a
    # reality that no longer exists. Recompute it with the same rule
    # (compute_safe_label, from prepare_danangleafy6_dataset.py) over the
    # new synthetic values, so quality and regression stay consistent
    # within this synthetic dataset. compute_safe_label needs the
    # *original* food name for MRL lookup ("Rau La" here is Cai Thia's
    # rows -- same convention as analyze_dataset.py's CLASS_TO_THRESHOLD_NAME).
    name_to_fid, name_to_pid, thresholds = load_maps()
    mrl_lookup_category = out["category"].replace({LEAFY_MERGED_LABEL: LEAFY_REPRESENTATIVE})
    out_for_label = out.assign(category=mrl_lookup_category)
    old_safe_label = out["safe_label"].value_counts().to_dict() if "safe_label" in out.columns else None
    out["safe_label"] = compute_safe_label(out_for_label, name_to_fid, name_to_pid, thresholds)
    print(f"  {machine} safe_label recomputed from synthetic concentrations: "
          f"{old_safe_label} -> {out['safe_label'].value_counts().to_dict()}")

    out_dir = f"{_HERE}/data/{machine}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/ALL_synthetic.csv"
    out.to_csv(out_path, index=False)
    print(f"{machine}: wrote {len(out)} rows -> {out_path}  (ceiling R2={ceiling})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ceiling", type=float, default=DEFAULT_CEILING)
    p.add_argument("machines", nargs="*", default=["FLAMENIR", "OCEANFX"])
    args = p.parse_args()
    for m in args.machines:
        run(m, args.ceiling)
