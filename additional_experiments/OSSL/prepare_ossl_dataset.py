"""Build an ALL.csv for the OSSL (Open Soil Spectral Library) VisNIR data in
the same schema the pipeline already expects (w_* wavelength columns,
`category`, plus one -1-sentinel target column per property), so it can be
used with the existing ClassificationNIRSDataset / RegressionNIRSDataset and
engines unchanged -- just point MACHINE=OSSL / DATASET_ROOT at its parent
directory.

There is no Stage 1 (presence/absence) equivalent here: soil properties are
always-measured continuous values, not analytes that can be "undetected"
like the pesticide substances, so only classification + Stage-2-style
regression are supported for this dataset.

Source: https://explorer.soilspectroscopy.org/ via the `soilspecdata`
package, which downloads+caches ossl_all_L0_v1.2.csv.gz (~946MB) to
~/.soilspecdata/ on first call.
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
from soilspecdata.datasets.ossl import get_ossl

OUT_DIR = os.environ.get("OSSL_OUT_DIR", _ALL + "/OSSL-Soil/OSSL")

# properties are stored under different method-registry column names
# depending on which sub-collection measured them (LUCAS.SSL uses ISO
# methods, KSSL.SSL/ICRAF.ISRIC use USDA methods) -- coalesce so a sample is
# only marked missing (-1) if truly neither method measured it.
TARGETS = {
    "Organic_Carbon": ("oc_iso.10694_w.pct", "oc_usda.c1059_w.pct"),
    "pH_H2O": ("ph.h2o_iso.10390_index", "ph.h2o_usda.a268_index"),
    "Clay_Content": ("clay.tot_iso.11277_w.pct", "clay.tot_usda.a334_w.pct"),
    "Sand_Content": ("sand.tot_iso.11277_w.pct", "sand.tot_usda.c60_w.pct"),
    "Silt_Content": ("silt.tot_iso.11277_w.pct", "silt.tot_usda.c62_w.pct"),
}
# these are % by weight -- a value > 100 is a data error, not a real reading
PCT_TARGETS = {"Clay_Content", "Sand_Content", "Silt_Content"}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    ossl = get_ossl()
    df = ossl.df

    wmin, wmax = 4000, 25000  # cm^-1 == 400-2500 nm (get_visnir's own defaults)
    wavenumbers = ossl._extract_wavenumbers(ossl.visnir_cols)
    order = np.argsort(-wavenumbers)  # descending wavenumber == ascending nm
    mask_w = (wavenumbers >= wmin) & (wavenumbers <= wmax)
    cols_sorted = [c for c, m in zip(np.array(ossl.visnir_cols)[order], mask_w[order]) if m]
    nm_sorted = 1e7 / wavenumbers[order][mask_w[order]]
    print("n bands (raw):", len(cols_sorted), "nm range:", nm_sorted.min(), nm_sorted.max())

    # SMART-NIR's MultiKernelBlock concatenates 4 parallel Conv1d branches
    # (kernel=4/8/16/32, all stride=4, padding=0/3/7/15); their output
    # lengths only line up when signal_len % 4 in {0, 1}. Trim the fewest
    # high-wavelength bands needed to satisfy this.
    n_bands = len(cols_sorted)
    rem = n_bands % 4
    if rem not in (0, 1):
        trim = rem - 1
        print(f"trimming {trim} band(s) off the high-wavelength end so "
              f"signal_len % 4 in {{0,1}} (SMART-NIR MultiKernelBlock constraint)")
        cols_sorted = cols_sorted[:-trim]
        nm_sorted = nm_sorted[:-trim]
    print("n bands (final):", len(cols_sorted), "nm range:", nm_sorted.min(), nm_sorted.max())

    valid_mask = df[cols_sorted].notna().all(axis=1)
    sub = df[valid_mask].copy()
    print("valid rows:", len(sub))

    X = sub[cols_sorted].values.astype(np.float32)
    w_cols = [f"w_{i}" for i in range(X.shape[1])]
    out = pd.DataFrame(X, columns=w_cols)
    out["category"] = sub["dataset.code_ascii_txt"].values

    for name, (iso_c, usda_c) in TARGETS.items():
        vals = sub[iso_c].combine_first(sub[usda_c]).values.astype(np.float64)
        assert np.all(np.isnan(vals) | (vals >= 0)), f"{name} has a negative real value, -1 sentinel unsafe"
        vals = np.where(np.isnan(vals), -1.0, vals)
        if name in PCT_TARGETS:
            bad = (vals != -1) & (vals > 100)
            if bad.any():
                print(f"{name}: nulling {bad.sum()} row(s) with value > 100 (data error)")
                vals = np.where(bad, -1.0, vals)
        out[name] = vals
        n_valid = int((vals != -1).sum())
        print(f"{name}: n_valid={n_valid} ({100 * n_valid / len(out):.1f}%), "
              f"range=[{vals[vals != -1].min():.3f}, {vals[vals != -1].max():.3f}]")

    out["id_layer"] = sub["id.layer_local_c"].values if "id.layer_local_c" in sub.columns else np.arange(len(sub))

    print("\ncategory distribution:")
    print(out["category"].value_counts())

    out_path = f"{OUT_DIR}/ALL.csv"
    out.to_csv(out_path, index=False)
    print("\nwritten:", out_path, out.shape)


if __name__ == "__main__":
    main()
