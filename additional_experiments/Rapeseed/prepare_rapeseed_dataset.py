"""Build an ALL.csv for the INRAE winter oilseed rape (rapeseed) tissue NIR
dataset (Recherche Data Gouv, DOI 10.57745/6VYUQN -- 2,427 samples of
roots/leaves/stems/flowers/pods, nitrogen and carbon content) in the same
schema the pipeline already expects (w_* wavelength columns, `category`,
plus one -1-sentinel target column per property), so it can be used with
the existing ClassificationNIRSDataset / RegressionNIRSDataset and engines
unchanged -- just point MACHINE=Rapeseed / DATASET_ROOT at its parent
directory.

Like OSSL/Grainit/Mango, there is no Stage 1 (presence/absence) equivalent:
N and C content are always-measured continuous tissue properties, not
analytes that can be "undetected" -- only classification (predict tissue
type) + Stage-2 regression (N_content, C_content) are supported.

The source file uses ';' as field separator and ',' as decimal separator
(European CSV convention) and ships its own per-target outlier flags
(N_model_set-type / C_model_set-type == "Outlier") -- a sample can be
flagged an outlier for one target's model but not the other's, so each
target is nulled (-1) independently rather than dropping the row outright.

Source files (download from
https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.57745/6VYUQN
first): Full_dataset.csv.
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

RAW_DIR = os.environ.get("RAPESEED_RAW_DIR", _ALL + "/RechercheDataGouv-Rapeseed")
OUT_DIR = os.environ.get("RAPESEED_OUT_DIR", _ALL + "/RechercheDataGouv-Rapeseed/Rapeseed")
RAW_FILE = "Full_dataset.csv"

META_COLS = [
    "Sample_ID", "Accession", "Growth season", "Growth condition", "N regime", "W regime",
    "Tissue", "BBCH dev. stage", "N content (%DM)", "N_model_set-type", "C content (%DM)",
    "C_model_set-type", "Spectra_acquisition_year", "Cup_type",
]
TARGETS = {
    # (value column, outlier-flag column)
    "N_content": ("N content (%DM)", "N_model_set-type"),
    "C_content": ("C content (%DM)", "C_model_set-type"),
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(f"{RAW_DIR}/{RAW_FILE}", sep=";", decimal=",")
    print("raw rows:", len(df))

    wave_cols = [c for c in df.columns if c not in META_COLS]
    wnum = np.array([float(c.replace(",", ".")) for c in wave_cols])
    order = np.argsort(wnum)  # ascending nm == descending wavenumber
    wave_cols = [wave_cols[i] for i in order]
    wnum = wnum[order]
    print("n bands (raw):", len(wave_cols), "wavenumber range:", wnum.min(), wnum.max())

    # SMART-NIR's MultiKernelBlock concatenates 4 parallel Conv1d branches
    # (kernel=4/8/16/32, all stride=4, padding=0/3/7/15); their output
    # lengths only line up when signal_len % 4 in {0, 1}. Trim the fewest
    # high-wavenumber (low-nm) bands needed to satisfy this.
    n_bands = len(wave_cols)
    rem = n_bands % 4
    if rem not in (0, 1):
        trim = rem - 1
        print(f"trimming {trim} band(s) so signal_len % 4 in {{0,1}} "
              f"(SMART-NIR MultiKernelBlock constraint)")
        wave_cols = wave_cols[trim:]
    print("n bands (final):", len(wave_cols))

    X = df[wave_cols].values.astype(np.float32)
    w_cols = [f"w_{i}" for i in range(X.shape[1])]
    out = pd.DataFrame(X, columns=w_cols)
    out["category"] = df["Tissue"].values

    for name, (val_col, flag_col) in TARGETS.items():
        vals = df[val_col].values.astype(np.float64)
        assert np.all(vals >= 0), f"{name} has a negative value, -1 sentinel unsafe"
        is_outlier = (df[flag_col] == "Outlier").values
        n_outlier = int(is_outlier.sum())
        print(f"{name}: nulling {n_outlier} row(s) flagged 'Outlier' in {flag_col}")
        vals = np.where(is_outlier, -1.0, vals)
        out[name] = vals
        n_valid = int((vals != -1).sum())
        print(f"{name}: n_valid={n_valid} ({100 * n_valid / len(out):.1f}%), "
              f"range=[{vals[vals != -1].min():.3f}, {vals[vals != -1].max():.3f}]")

    for col in ["Accession", "Growth season", "Growth condition", "N regime", "W regime",
                "BBCH dev. stage", "Cup_type"]:
        out[col] = df[col].values

    print("\ntissue distribution:")
    print(out["category"].value_counts())

    out_path = f"{OUT_DIR}/ALL.csv"
    out.to_csv(out_path, index=False)
    print("\nwritten:", out_path, out.shape)


if __name__ == "__main__":
    main()
