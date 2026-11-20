"""Parses the raw instrument .txt exports under
{DATASET_ROOT}/leafy-position-scan/{machine}/raw/{category}/ (4 leafy-green
categories x 2 machines, each scanned at up to 3 physical positions per
sample: la 1 = outer leaf face, la 2 = inner leaf face, cuong = stem) into
tidy CSVs under {DATASET_ROOT}/leafy-position-scan/{machine}/processed/,
adding a `position` column parsed from the filename.

Category folder names differ slightly per machine (OCEANFX: "Mongtoi",
FLAMENIR: "Montoi") and are matched case-insensitively via
FOLDER_NAME_TO_CATEGORY below rather than hardcoded per machine.

Position parsing (checked against the part of the filename before the
first "__<replicate>__" token, so the trailing numeric file counter never
leaks in -- the instrument serial code, e.g. "OFX034251"/"FLMN032701", is
safe to leave in place since its digits are always flanked by other
digits and so never match the standalone-1/2 lookaround below):
  - contains "cuong"            -> "Cuống"
  - contains a standalone "2"   -> "Lá 2"
  - contains a standalone "1", or neither token (OCEANFX Mongtoi's bare
    "Montoi_OFX..." files, the only group with no explicit "1" variant)
    -> "Lá 1"

Output, per machine (under {DATASET_ROOT}/leafy-position-scan/{machine}/processed/):
  {category}.csv    one row per scan, columns: file, position,
      w_0 .. w_{n-1} (raw instrument intensity, in the project's standard
      w_<index> column convention; n = 2136 for OCEANFX, 128 for FLAMENIR)
  ALL.csv           same rows for all 4 categories, with an extra leading
      `category` column (Vietnamese name, matching food_ids.json)
  wavelengths.csv   the shared wavelength axis (nm) for that machine, one row

Usage: python3 build_data_new_processed.py [MACHINE ...]
"""
import os
import re
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

RAW_DIR = f"{os.environ['DATASET_ROOT']}/leafy-position-scan"
OUT_DIR = f"{os.environ['DATASET_ROOT']}/leafy-position-scan"

FOLDER_NAME_TO_CATEGORY = {
    "xalach": "Xà Lách",
    "caibex": "Cải Bẹ Xanh",
    "caithi": "Cải Thìa",
    "mongtoi": "Mồng Tơi",
    "montoi": "Mồng Tơi",
}


def parse_spectrum(text: str):
    lines = text.splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if "Begin Spectral Data" in l) + 1
    except StopIteration:
        return []
    rows = []
    for line in lines[start:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    return rows


def parse_position(filename: str) -> str:
    prefix = filename.split("__")[0].lower()
    if "cuong" in prefix:
        return "Cuống"
    if re.search(r"(?<!\d)2(?!\d)", prefix):
        return "Lá 2"
    return "Lá 1"


def run(machine: str):
    machine_raw_dir = f"{RAW_DIR}/{machine}/raw"
    if not os.path.isdir(machine_raw_dir):
        print(f"SKIP {machine}: no {machine_raw_dir}")
        return

    out_dir = f"{OUT_DIR}/{machine}/processed"
    os.makedirs(out_dir, exist_ok=True)
    wavelengths = None
    all_rows = []

    print(f"=== {machine} ===")
    for folder in sorted(os.listdir(machine_raw_dir)):
        cat_dir = os.path.join(machine_raw_dir, folder)
        if not os.path.isdir(cat_dir):
            continue
        category = FOLDER_NAME_TO_CATEGORY.get(folder.lower())
        if category is None:
            print(f"  SKIP folder (unknown category): {folder}")
            continue

        rows = []
        for fname in sorted(os.listdir(cat_dir)):
            if not fname.endswith(".txt"):
                continue
            path = os.path.join(cat_dir, fname)
            with open(path, encoding="utf-8", errors="replace") as f:
                text = f.read()
            spectrum = parse_spectrum(text)
            if not spectrum:
                print(f"  SKIP (no spectrum): {path}")
                continue
            wl = np.array([w for w, _ in spectrum])
            vals = np.array([v for _, v in spectrum])
            if wavelengths is None:
                wavelengths = wl
            elif len(wl) != len(wavelengths) or not np.allclose(wl, wavelengths):
                print(f"  WARNING: wavelength axis mismatch in {path}, skipping")
                continue
            position = parse_position(fname)
            rows.append([fname, position] + vals.tolist())

        w_cols = [f"w_{i}" for i in range(len(wavelengths))]
        df = pd.DataFrame(rows, columns=["file", "position"] + w_cols)
        out_path = f"{out_dir}/{folder}.csv"
        df.to_csv(out_path, index=False)
        print(f"  {category:<14} ({folder}): {len(df)} rows -> {out_path}")
        print("    " + df["position"].value_counts().to_string().replace("\n", "\n    "))

        df_all = df.copy()
        df_all.insert(0, "category", category)
        all_rows.append(df_all)

    pd.concat(all_rows, ignore_index=True).to_csv(f"{out_dir}/ALL.csv", index=False)
    pd.DataFrame([wavelengths], columns=[f"w_{i}" for i in range(len(wavelengths))]).to_csv(
        f"{out_dir}/wavelengths.csv", index=False)
    print(f"  Wrote {out_dir}/ALL.csv and {out_dir}/wavelengths.csv")


if __name__ == "__main__":
    machines = sys.argv[1:] or ["FLAMENIR", "OCEANFX"]
    for m in machines:
        run(m)
