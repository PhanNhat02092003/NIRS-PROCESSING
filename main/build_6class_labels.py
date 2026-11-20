"""Builds {DATASET_ROOT}/{machine}/ALL_downsampled{suffix}_6class.csv: a
copy of ALL_downsampled{suffix}.csv (downsample_training_data.py's de-
duplicated, group-tagged training set) with the 4 spectrally-similar
leafy-green categories (Xa Lach, Cai Be Xanh, Cai Thia, Mong Toi) collapsed
into one merged label, LEAFY_MERGED_LABEL. The other 5 categories (Ca Chua,
Ca Rot, Dua Leo, Kho Qua, Dau Cove) are left untouched, and so is
`_group_id` -- it was assigned per (original category, 19-substance vector)
combination, which merging the display category doesn't change, so it's
still valid for StratifiedGroupKFold (see food_classification_6class.py).

Uses a downsampled file (not the raw ALL.csv) for the same reason
food_classification_hierarchical.py does: training on the raw file with a
plain (non-grouped) split risks one scanned batch (sometimes 1000+ rows)
straddling train and validation, letting the model learn batch-specific
acquisition quirks instead of the real spectral signal.

Default source is ALL_downsampled3.csv (downsample_training_data.py
--cap 25 --suffix 3) -- chosen over the default ALL_downsampled.csv
(CAP=200) because GuidedDCNet's batch_size=3072 default leaves too few
batches/epoch at that larger size's ~80% train split. See
food_classification_6class.py's GDC_BATCH_SIZE override for the matching
smaller batch size.

Replaces the old design (9-way classifier, then always display "Cai Thia"
for any of the 4 leafy guesses -- LEAFY_DISPLAY_OVERRIDE in deploy_utils.py)
with: a 6-way classifier (this script's data) that only needs to recognize
"this is some leafy green", then a separate PCA+LDA model (see
train_leafy_pca_lda.py) resolves which of the 4 it actually is.

Usage: python3 build_6class_labels.py [--suffix SUFFIX] [FLAMENIR OCEANFX]
"""
import argparse
import os

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DATASET_ROOT = os.environ["DATASET_ROOT"]
LEAFY_GREENS = {"Xà Lách", "Cải Bẹ Xanh", "Cải Thìa", "Mồng Tơi"}
LEAFY_MERGED_LABEL = "Rau Lá"


def run(machine: str, suffix: str):
    src = f"{DATASET_ROOT}/{machine}/ALL_downsampled{suffix}.csv"
    dst = f"{DATASET_ROOT}/{machine}/ALL_downsampled{suffix}_6class.csv"
    df = pd.read_csv(src)
    before = df["category"].value_counts()
    df["category"] = df["category"].apply(lambda c: LEAFY_MERGED_LABEL if c in LEAFY_GREENS else c)
    after = df["category"].value_counts()
    df.to_csv(dst, index=False)
    print(f"{machine}: {len(df)} rows ({df['_group_id'].nunique()} groups) -> {dst}")
    print(f"  before (9-class): {before.to_dict()}")
    print(f"  after  (6-class): {after.to_dict()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="3")  # "3" = ALL_downsampled3.csv (CAP=25)
    ap.add_argument("machines", nargs="*", default=["FLAMENIR", "OCEANFX"])
    args = ap.parse_args()
    for m in args.machines:
        run(m, args.suffix)
