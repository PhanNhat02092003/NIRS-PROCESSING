"""Builds the DanangLeafy6 benchmark dataset: a per-machine copy of the
Danang pesticide dataset's CAP=25 downsampled, group-tagged data
(../all-dataset/Danang-NIR/{machine}/ALL_downsampled3.csv, built by
main/downsample_training_data.py --cap 25 --suffix 3), with the 4
spectrally-similar leafy-green categories (Xa Lach, Cai Be Xanh, Cai Thia,
Mong Toi) collapsed into one class by DROPPING the 3 non-representative
ones and keeping only Cai Thia's rows (relabeled "Rau La") -- NOT the same
merge main/build_6class_labels.py uses (which keeps every row from all 4,
making that class ~3-4x larger than the other 5). This is a separate,
standalone benchmark variant of the classification problem, kept apart
from the deployed main/ pipeline's own 6-class data.

Full 3-task benchmark (classification + regression + quality), matching
Grainit/Mango/Rapeseed's own shape, not a classification-only variant:
  - classification: the 6-class category above.
  - regression: 6 pesticide concentrations (SUBSTANCES below -- a subset
    of the 19 tracked for the main pipeline, picked for this benchmark;
    -1 = absent, LensNIR's own MISSING_VALUE sentinel -- no transform
    needed).
  - quality: per-sample safety decision ("An toan" / "Vuot nguong"), a
    LEARNED head (like Rapeseed's "N regime"), not a rule derived from a
    single target -- computed here from the *original* (pre-merge)
    category + measured concentrations against DATASET_ROOT/thresholds.json
    MRLs, same row-level rule as main/stage2_safety_downsampled.py
    ((conc != -1) & (conc > mrl) => unsafe for that substance), aggregated
    over just these 6 substances: unsafe iff ANY of them is unsafe. A
    rule-derived "quality_rules" dict (LensNIR's other mechanism) doesn't
    fit here -- it treats a NaN/missing target as "undefined", but an
    absent pesticide must count as trivially safe for that substance, not
    undefined.

Output: additional/DanangLeafy6/data/{machine}/ALL.csv -- `category`,
`_group_id`, `safe_label`, the 6 substance columns, `w_0..w_{n-1}`.

Usage: python3 prepare_danangleafy6_dataset.py [FLAMENIR OCEANFX]
"""
import json
import os
import sys

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

_HERE = os.path.dirname(os.path.abspath(__file__))

DATASET_ROOT = os.environ.get("DATASET_ROOT", "/home/phannhat/all-dataset/Danang-NIR")
LEAFY_REPRESENTATIVE = "Cải Thìa"
LEAFY_DROPPED = {"Xà Lách", "Cải Bẹ Xanh", "Mồng Tơi"}
LEAFY_MERGED_LABEL = "Rau Lá"

SUBSTANCES = [
    'Thiamethoxam', 'Permethrin', 'Azoxystrobin',
    'Difenoconazole', 'Cypermethrin', 'Chlothianidin',
]


def load_maps():
    food_ids = json.load(open(f"{DATASET_ROOT}/food_ids.json"))
    pest_ids = json.load(open(f"{DATASET_ROOT}/pesticide_ids.json"))
    thresholds = json.load(open(f"{DATASET_ROOT}/thresholds.json"))
    name_to_fid = {v["name"]: k for k, v in food_ids.items()}
    name_to_pid = {v["name"]: k for k, v in pest_ids.items()}
    return name_to_fid, name_to_pid, thresholds


def compute_safe_label(df: pd.DataFrame, name_to_fid: dict, name_to_pid: dict, thresholds: dict) -> pd.Series:
    """'An toan' unless at least one substance is truly present (!= -1)
    AND its concentration exceeds that (food, substance)'s MRL -- same
    row-level rule as main/stage2_safety_downsampled.py, aggregated over
    all 19 substances (unsafe iff ANY one of them is unsafe)."""
    fid_per_row = df["category"].map(name_to_fid)
    unsafe = pd.Series(False, index=df.index)
    for s in SUBSTANCES:
        pid = name_to_pid[s]
        mrl_per_row = fid_per_row.map(lambda fid: thresholds[fid][pid]["mrl"])
        unsafe |= (df[s] != -1) & (df[s] > mrl_per_row)
    return unsafe.map({True: "Vuot nguong", False: "An toan"})


def run(machine: str):
    src = f"{DATASET_ROOT}/{machine}/ALL_downsampled3.csv"
    out_dir = f"{_HERE}/data/{machine}"
    os.makedirs(out_dir, exist_ok=True)
    dst = f"{out_dir}/ALL.csv"

    name_to_fid, name_to_pid, thresholds = load_maps()
    df = pd.read_csv(src)
    before = df["category"].value_counts()

    df["safe_label"] = compute_safe_label(df, name_to_fid, name_to_pid, thresholds)

    df = df[~df["category"].isin(LEAFY_DROPPED)].copy()
    df["category"] = df["category"].apply(lambda c: LEAFY_MERGED_LABEL if c == LEAFY_REPRESENTATIVE else c)

    w_cols = [c for c in df.columns if c.startswith("w_")]
    out = df[["category", "_group_id", "safe_label"] + SUBSTANCES + w_cols]
    out.to_csv(dst, index=False)

    after = out["category"].value_counts()
    print(f"{machine}: {len(out)} rows ({out['_group_id'].nunique()} groups) -> {dst}")
    print(f"  before (9-class): {before.to_dict()}")
    print(f"  after  (6-class, Cai Thia-only leafy): {after.to_dict()}")
    print(f"  safe_label: {out['safe_label'].value_counts().to_dict()}")


if __name__ == "__main__":
    machines = sys.argv[1:] or ["FLAMENIR", "OCEANFX"]
    for m in machines:
        run(m)
