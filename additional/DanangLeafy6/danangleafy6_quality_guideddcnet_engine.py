"""GuidedDCNet quality-decision benchmark on DanangLeafy6: predicts
`safe_label` ("An toan" / "Vuot nguong") directly from spectra -- see
danangleafy6_quality_smartnir_engine.py's docstring for the label.

Reuses run_guideddcnet() (main/food_classification.py) completely
unchanged by writing a throwaway copy of ALL.csv with `category` replaced
by `safe_label`'s values (run_guideddcnet's ClassificationNIRSDataset
hardcodes the `category` column name) -- same trick already proven safe
by danangleafy6_guideddcnet_classification_engine.py's sibling file, just
swapping which column becomes `category`. StratifiedGroupKFold on
`_group_id` falls out of run_guideddcnet(group_col=...) automatically.

Env: MACHINE (FLAMENIR | OCEANFX), BIN (OCEANFX: average wavelengths in
groups of 8, matching every other deep model on this machine).

Usage: MACHINE=FLAMENIR python3 danangleafy6_quality_guideddcnet_engine.py
       MACHINE=OCEANFX BIN=8 python3 danangleafy6_quality_guideddcnet_engine.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))  # repo root: model/, dataset/, shared engines
sys.path.insert(0, os.path.join(_ROOT, "main"))
sys.path.insert(0, os.path.dirname(_HERE))  # additional/: lets Python merge additional/dataset/ into the dataset namespace package
os.chdir(_HERE)  # relative outputs (history/, checkpoint/, data/) land next to this script

import pandas as pd

from food_classification import run_guideddcnet

os.environ.setdefault("GDC_BATCH_SIZE", "256")  # small dataset, see main/food_classification_6class.py

TASK = "category_classification_quality_guideddcnet"

if __name__ == "__main__":
    machine = os.environ["MACHINE"]
    full_path = f"{_HERE}/data/{machine}/ALL_synthetic.csv"
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"{full_path} not found -- run prepare_danangleafy6_dataset.py first")

    quality_path = f"{_HERE}/data/{machine}/ALL_quality.csv"
    if not os.path.exists(quality_path):
        df = pd.read_csv(full_path)
        df = df.drop(columns=["category"]).rename(columns={"safe_label": "category"})
        df.to_csv(quality_path, index=False)
        print(f"Wrote {quality_path} ({len(df)} rows, category <- safe_label)")

    run_guideddcnet(quality_path, TASK, machine, group_col="_group_id")
