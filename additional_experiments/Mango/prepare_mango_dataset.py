"""Build an ALL.csv for the Mango DMC and NIR Spectra dataset (Mendeley
DOI 10.17632/46htwnp833 -- 10 mango cultivars, dry matter content) in the
same schema the pipeline already expects (w_* wavelength columns,
`category`, plus one target column), so it can be used with the existing
ClassificationNIRSDataset / RegressionNIRSDataset and engines unchanged --
just point MACHINE=Mango / DATASET_ROOT at its parent directory.

Uses the v4 file (MangoDMC_NIR_Data_v4.csv, 85,401 rows, 10 cultivars, 31
instruments, 2015-2021 seasons) rather than the smaller v3 file (12,011
rows, single-schema) -- v4 is the newer, larger superset described in the
dataset's own metadata as augmenting v3 with more instruments/seasons, and
ships its own outlier flag for quality control.

Like OSSL and Grainit, there is no Stage 1 (presence/absence) equivalent:
dry matter content is an always-measured continuous fruit property, not an
analyte that can be "undetected" -- only classification (predict cultivar)
+ Stage-2 regression (dry_matter) are supported for this dataset.

Wavelength range: the 306 raw columns (285-1200nm, 3nm step) are zero-padded
outside each instrument's native sensing range (31 different instruments,
each covering a different sub-range) -- using the full range would silently
feed fake zeros into SNV normalization for narrower-range instruments. Only
the 507-1023nm band is real (nonzero) for every single row regardless of
instrument (the longest contiguous 100%-real-coverage span found in the raw
data, trimmed by 2 bands for the SMART-NIR MultiKernelBlock constraint).

Source file (download from https://data.mendeley.com/datasets/46htwnp833/4
first): MangoDMC_NIR_Data_v4.csv.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))  # repo root: model/, dataset/, shared engines
_ALL = os.path.join(_ROOT, "..", "all-dataset")
sys.path.insert(0, _ROOT)
os.chdir(_HERE)  # relative outputs (history/, checkpoint/, data/) land next to this script
import os

import numpy as np
import pandas as pd

RAW_DIR = os.environ.get("MANGO_RAW_DIR", _ALL + "/Mendeley-Mango")
OUT_DIR = os.environ.get("MANGO_OUT_DIR", _ALL + "/Mendeley-Mango/Mango")
RAW_FILE = "MangoDMC_NIR_Data_v4.csv"

META_COLS = [
    "partition_1", "outlier_flag_1", "subsequent_flag_1", "train_partition_1",
    "sample_order_1", "partition_ext", "origin", "population", "date", "season",
    "region", "cultivar", "physio_stage", "temp", "reference_no", "dry_matter",
    "instrument", "spectra_no",
]
WMIN, WMAX = 507, 1023  # nm, see module docstring


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(f"{RAW_DIR}/{RAW_FILE}", low_memory=False)
    print("raw rows:", len(df))

    df = df[df["outlier_flag_1"] == 0].reset_index(drop=True)
    print("rows after outlier_flag_1==0 filter:", len(df))

    wave_cols = [c for c in df.columns if c not in META_COLS]
    wnum = np.array([float(c) for c in wave_cols])
    keep_cols = [c for c, w in zip(wave_cols, wnum) if WMIN <= w <= WMAX]
    n_bands = len(keep_cols)
    print(f"n bands in [{WMIN},{WMAX}]nm: {n_bands}")
    assert n_bands % 4 in (0, 1), (
        f"signal_len={n_bands} fails SMART-NIR MultiKernelBlock's len%4 in {{0,1}} "
        f"constraint -- adjust WMIN/WMAX"
    )

    X = df[keep_cols].values.astype(np.float32)
    valid_mask = ~np.isnan(X).any(axis=1)
    n_dropped = int((~valid_mask).sum())
    if n_dropped:
        print(f"dropping {n_dropped} row(s) with NaN spectra")
    df = df[valid_mask].reset_index(drop=True)
    X = X[valid_mask]

    w_cols = [f"w_{i}" for i in range(n_bands)]
    out = pd.DataFrame(X, columns=w_cols)
    out["category"] = df["cultivar"].values
    out["dry_matter"] = df["dry_matter"].values.astype(np.float64)

    for col in ["region", "season", "physio_stage", "temp", "instrument", "reference_no"]:
        out[col] = df[col].values

    print("\ncultivar distribution:")
    print(out["category"].value_counts())
    print("\ndry_matter: n=", len(out), "range=[{:.3f}, {:.3f}]".format(
        out["dry_matter"].min(), out["dry_matter"].max()))

    out_path = f"{OUT_DIR}/ALL.csv"
    out.to_csv(out_path, index=False)
    print("\nwritten:", out_path, out.shape)


if __name__ == "__main__":
    main()
