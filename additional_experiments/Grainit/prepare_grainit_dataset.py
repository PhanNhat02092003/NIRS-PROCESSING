"""Build an ALL.csv for the sensAIfood Cereal NIR Dataset (Grainit set,
Zenodo DOI 10.5281/zenodo.15838272 -- barley/corn/wheat, AuroraNIR
950-1650nm, reference protein % and moisture %) in the same schema the
pipeline already expects (w_* wavelength columns, `category`, plus one
-1-sentinel target column per property), so it can be used with the
existing ClassificationNIRSDataset / RegressionNIRSDataset and engines
unchanged -- just point MACHINE=Grainit / DATASET_ROOT at its parent
directory.

Like OSSL, there is no Stage 1 (presence/absence) equivalent: protein and
moisture are always-measured continuous grain properties, not analytes that
can be "undetected" -- only classification (predict cereal type) + Stage-2
regression (protein, moisture) are supported for this dataset.

Source files (download from https://zenodo.org/records/15838272 and unzip
first): Barley_sensAIfood_Grainit.csv, Corn_sensAIfood_Grainit.csv,
Wheat_sensAIfood_Grainit.csv.
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

RAW_DIR = os.environ.get("GRAINIT_RAW_DIR", _ALL + "/Zenodo-Grainit")
OUT_DIR = os.environ.get("GRAINIT_OUT_DIR", _ALL + "/Zenodo-Grainit/Grainit")

FILES = {
    "barley": "Barley_sensAIfood_Grainit.csv",
    "corn": "Corn_sensAIfood_Grainit.csv",
    "wheat": "Wheat_sensAIfood_Grainit.csv",
}
# 0.0 is used in the raw wheat file as a missing-reading placeholder for
# some Italy/2016 rows (0% moisture or 0% protein in a real grain sample is
# physically impossible) -- treat it as missing (-1), not a real zero.
TARGETS = ["Moisture", "Protein"]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    frames = []
    for cereal, fname in FILES.items():
        df = pd.read_csv(f"{RAW_DIR}/{fname}")
        df["category"] = cereal
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    print("combined rows:", len(df), "by cereal:")
    print(df["category"].value_counts())

    wave_cols = sorted([c for c in df.columns if c.isdigit()], key=int)
    print("n bands (raw):", len(wave_cols), "nm range:", wave_cols[0], wave_cols[-1])

    # SMART-NIR's MultiKernelBlock concatenates 4 parallel Conv1d branches
    # (kernel=4/8/16/32, all stride=4, padding=0/3/7/15); their output
    # lengths only line up when signal_len % 4 in {0, 1}. Trim the fewest
    # high-wavelength bands needed to satisfy this.
    n_bands = len(wave_cols)
    rem = n_bands % 4
    if rem not in (0, 1):
        trim = rem - 1
        print(f"trimming {trim} band(s) off the high-wavelength end so "
              f"signal_len % 4 in {{0,1}} (SMART-NIR MultiKernelBlock constraint)")
        wave_cols = wave_cols[:-trim]
    print("n bands (final):", len(wave_cols), "nm range:", wave_cols[0], wave_cols[-1])

    X = df[wave_cols].values.astype(np.float32)
    w_cols = [f"w_{i}" for i in range(X.shape[1])]
    out = pd.DataFrame(X, columns=w_cols)
    out["category"] = df["category"].values

    for name in TARGETS:
        vals = df[name].values.astype(np.float64)
        assert np.all(vals >= 0), f"{name} has a negative value, -1 sentinel unsafe"
        n_zero = int((vals == 0).sum())
        if n_zero:
            print(f"{name}: nulling {n_zero} row(s) with value == 0 (missing-reading placeholder)")
            vals = np.where(vals == 0, -1.0, vals)
        out[name] = vals
        n_valid = int((vals != -1).sum())
        print(f"{name}: n_valid={n_valid} ({100 * n_valid / len(out):.1f}%), "
              f"range=[{vals[vals != -1].min():.3f}, {vals[vals != -1].max():.3f}]")

    for col in ["ID", "Spectrometer", "Variety", "Country", "Year"]:
        out[col] = df[col].values

    out_path = f"{OUT_DIR}/ALL.csv"
    out.to_csv(out_path, index=False)
    print("\nwritten:", out_path, out.shape)


if __name__ == "__main__":
    main()
