"""GuidedDCNet classification benchmark on DanangLeafy6 (prepare_
danangleafy6_dataset.py: Danang pesticide dataset, CAP=25 downsampled, the
4 leafy greens collapsed into one Cai Thia-only "Rau La" class). Reuses
run_guideddcnet() (main/food_classification.py) unchanged, with
StratifiedGroupKFold on `_group_id` (same leakage-safe protocol as
main/food_classification_hierarchical.py).

Env: MACHINE (FLAMENIR | OCEANFX), BIN (OCEANFX: average wavelengths in
groups of 8, matching every other deep model on this machine).

Usage: MACHINE=FLAMENIR python3 danangleafy6_guideddcnet_classification_engine.py
       MACHINE=OCEANFX BIN=8 python3 danangleafy6_guideddcnet_classification_engine.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))  # repo root: model/, dataset/, shared engines
sys.path.insert(0, os.path.join(_ROOT, "main"))
sys.path.insert(0, os.path.dirname(_HERE))  # additional/: lets Python merge additional/dataset/ into the dataset namespace package
os.chdir(_HERE)  # relative outputs (history/, checkpoint/, data/) land next to this script

from food_classification import run_guideddcnet

os.environ.setdefault("GDC_BATCH_SIZE", "256")  # small dataset (~5,700-6,000 rows), see main/food_classification_6class.py

TASK = "category_classification_guideddcnet"

if __name__ == "__main__":
    machine = os.environ["MACHINE"]
    full_path = f"{_HERE}/data/{machine}/ALL.csv"
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"{full_path} not found -- run prepare_danangleafy6_dataset.py first")
    run_guideddcnet(full_path, TASK, machine, group_col="_group_id")
