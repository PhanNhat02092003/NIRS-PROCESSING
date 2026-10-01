"""Cleans the raw instrument exports under DATASET_ROOT/spectrum-differencce
(NIR absorbance of each pesticide's solution across a concentration series,
FLAMENIR + OCEANFX) into one tidy long-format CSV per (machine, substance):

    columns: rank, ppm, replicate, wavelength, absorbance

rank is the concentration level as labelled in the raw filenames (C1 =
highest .. C5 = lowest); ppm is the explicit concentration when the filename
states one, else blank (some sessions only ever recorded the C1..C5 rank).

Only the instrument's own precomputed "*Absorbance*" exports are used (raw
"*_view_*"/"*_OFX...*"/"*_FLMN...*" intensity files and acquisition PDFs are
ignored); each export is already dark/reference-corrected by the instrument
software.

Excluded on purpose (not single-substance solution concentration data):
  - "Đo tán xạ (Mau rắn)" (OCEANFX): solid-sample scattering/reflectance,
    not a liquid concentration series.
  - "HC1&8", "HC4&6" (FLAMENIR): two substances mixed in one solution, so a
    peak can't be attributed to a single pesticide.
  - ACN / Ethanol / Acetonitril / dark / reference folders and files: pure
    solvent or dark-reference scans, not a pesticide spectrum.
  - the empty stray "HC17_C1-5000ppm" duplicate directory under OCEANFX.

HCxx -> substance name follows the same P01..P19 order as
DATASET_ROOT/pesticide_ids.json (HC03 = Metalaxyl = P03, etc.).

Usage: python3 clean_spectrum_difference.py
Writes DATASET_ROOT/spectrum-differencce-clean/{machine}/{substance}.csv
and a manifest.csv summarizing what was kept/skipped and why.
"""
import io
import json
import os
import re
import zipfile

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

ROOT = os.environ["DATASET_ROOT"]
RAW_DIR = f"{ROOT}/spectrum-differencce"
OUT_DIR = f"{ROOT}/spectrum-differencce-clean"

with open(f"{ROOT}/pesticide_ids.json") as f:
    PESTICIDE_IDS = json.load(f)
HC_TO_SUBSTANCE = {i + 1: v["name"] for i, v in enumerate(PESTICIDE_IDS.values())}

NOT_A_CHEMICAL = {"acn", "ethanol", "acetonitril", "ref", "reference", "dark"}


def hc_folder_to_substance(folder_name: str):
    """Returns the substance name for a clean 'HC<n>' folder, or None if
    this folder isn't a single-substance chemical folder (mixture, solvent
    reference, or anything else)."""
    m = re.fullmatch(r"HC(\d+)", folder_name.strip())
    if not m:
        return None
    return HC_TO_SUBSTANCE.get(int(m.group(1)))


def parse_concentration(filename: str):
    """Returns (rank:int 1-5, ppm:float|None) parsed from a filename, or
    (None, None) if no C<n> rank token is found."""
    ppm = None
    m = re.search(r"(\d{2,6})\s*ppm", filename, re.IGNORECASE)
    if m:
        ppm = float(m.group(1))
    # Negative lookbehind for "H" so "HC2-C1_..." doesn't match the "C2"
    # inside the chemical code "HC2" itself before ever reaching the real
    # concentration token "C1".
    m = re.search(r"(?<!H)[Cc](\d)(?=[-_ ]|$)", filename)
    if not m:
        return None, ppm
    return int(m.group(1)), ppm


def parse_replicate(filename: str):
    m = re.search(r"__(\d+)__\d+\.txt$", filename)
    return int(m.group(1)) if m else 0


def parse_spectrum(text: str):
    """Returns list of (wavelength, value) from an instrument export's body."""
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


def is_absorbance_file(filename: str) -> bool:
    return filename.lower().endswith(".txt") and "absorbance" in filename.lower()


def collect_oceanfx():
    """Yields (substance, rank, ppm, replicate, wavelength, absorbance, source_path)."""
    base = f"{RAW_DIR}/OCEANFX"
    for session in sorted(os.listdir(base)):
        if session == "Đo tán xạ (Mau rắn)":
            continue
        session_path = os.path.join(base, session)
        if not os.path.isdir(session_path):
            continue
        for hc_folder in sorted(os.listdir(session_path)):
            substance = hc_folder_to_substance(hc_folder)
            if substance is None:
                continue
            hc_path = os.path.join(session_path, hc_folder)
            if not os.path.isdir(hc_path):
                continue
            for fname in sorted(os.listdir(hc_path)):
                if not is_absorbance_file(fname):
                    continue
                rank, ppm = parse_concentration(fname)
                if rank is None:
                    continue
                replicate = parse_replicate(fname)
                fpath = os.path.join(hc_path, fname)
                with open(fpath, encoding="utf-8", errors="replace") as f:
                    text = f.read()
                for wl, val in parse_spectrum(text):
                    yield substance, rank, ppm, replicate, wl, val, fpath


def collect_flamenir():
    zpath = f"{RAW_DIR}/FLAMENIIR/FLAMENIR.zip"
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            if name.endswith("/"):
                continue
            fname = os.path.basename(name)
            if not is_absorbance_file(fname):
                continue
            # parent folder of this file inside the zip, e.g. "FLAMENIR/260822/HC2"
            hc_folder = os.path.basename(os.path.dirname(name))
            substance = hc_folder_to_substance(hc_folder)
            if substance is None:
                continue
            rank, ppm = parse_concentration(fname)
            if rank is None:
                continue
            replicate = parse_replicate(fname)
            text = z.read(name).decode("utf-8", errors="replace")
            for wl, val in parse_spectrum(text):
                yield substance, rank, ppm, replicate, wl, val, name


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest_rows = []

    for machine, collector in (("OCEANFX", collect_oceanfx), ("FLAMENIR", collect_flamenir)):
        rows = list(collector())
        if not rows:
            continue
        df = pd.DataFrame(rows, columns=["substance", "rank", "ppm", "replicate", "wavelength", "absorbance", "source"])
        machine_dir = f"{OUT_DIR}/{machine}"
        os.makedirs(machine_dir, exist_ok=True)
        for substance, g in df.groupby("substance"):
            out_path = f"{machine_dir}/{substance}.csv"
            g.drop(columns=["source", "substance"]).sort_values(["rank", "replicate", "wavelength"]).to_csv(out_path, index=False)
            n_ranks = g["rank"].nunique()
            n_replicates = g.groupby("rank")["replicate"].nunique().to_dict()
            manifest_rows.append({
                "machine": machine, "substance": substance, "n_ranks": n_ranks,
                "replicates_per_rank": n_replicates, "n_points": len(g),
                "ppm_values": sorted(g["ppm"].dropna().unique().tolist()),
            })
            print(f"{machine:9s} {substance:20s} ranks={n_ranks} points={len(g):6d} -> {out_path}")

    pd.DataFrame(manifest_rows).to_csv(f"{OUT_DIR}/manifest.csv", index=False)
    print(f"\nWrote manifest: {OUT_DIR}/manifest.csv ({len(manifest_rows)} machine/substance pairs)")


if __name__ == "__main__":
    main()
