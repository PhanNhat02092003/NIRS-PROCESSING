"""Trains the deployed GuidedDCNet food classifier on the merged 6-class
labels (build_6class_labels.py's ALL_downsampled_6class.csv: the 4 leafy
greens collapsed into one "Rau Lá" label, the other 5 categories
unchanged), replacing the 9-class model as the production classifier. A
separate PCA+LDA model (train_leafy_pca_lda.py) resolves which of the 4
leafy greens a "Rau Lá" prediction actually is -- see deploy_utils.py's
infer_category_classification()/analyze_spectrum().

Trains on ALL_downsampled3_6class.csv (CAP=25, see build_6class_labels.py)
with StratifiedGroupKFold on `_group_id`, same leakage-safe protocol as
food_classification_hierarchical.py: one scanned batch can't straddle
train and validation.

GDC_BATCH_SIZE defaults to 3072 in run_guideddcnet() -- tuned for the
full ~335k/380k-row ALL.csv. At CAP=25 (~8,000 rows total, ~80% per
StratifiedGroupKFold train split), that leaves only ~2 batches/epoch,
too few for stable BatchNorm statistics or meaningful gradient updates.
Override it here to 256 (~25-27 batches/epoch) unless the caller already
set GDC_BATCH_SIZE explicitly.

Reuses run_guideddcnet() (food_classification.py) unchanged -- same
training loop, just a different data file/task name/group column/batch
size, so checkpoints/history/data land under
task="category_classification_6way_guideddcnet" without touching the
original 9-class checkpoints.

Usage: MACHINE=FLAMENIR python3 food_classification_6class.py
       MACHINE=OCEANFX BIN=8 python3 food_classification_6class.py
"""
import os

from dotenv import load_dotenv

from food_classification import run_guideddcnet

load_dotenv()
os.environ.setdefault("GDC_BATCH_SIZE", "256")

TASK = "category_classification_6way_guideddcnet"

if __name__ == "__main__":
    machine = os.environ["MACHINE"]
    full_path = f"{os.environ['DATASET_ROOT']}/{machine}/ALL_downsampled3_6class.csv"
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"{full_path} not found -- run build_6class_labels.py first")
    run_guideddcnet(full_path, TASK, machine, group_col="_group_id")
