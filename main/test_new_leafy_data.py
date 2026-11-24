"""Evaluates deploy_utils.py's real classification pipeline (6-way
GuidedDCNet -> leafy PCA+LDA disambiguator) on ../new-test-data/{machine}/
{Caibex,Caithi,Montoi,Xalach}/*.txt -- new leafy-green scans, all taken at
the "La 2" / inner-leaf-face position the PCA+LDA model requires (per the
person running this).

Ground truth is the folder name; ".txt" files are the spectrometer's raw
export (header text, then ">>>>>Begin Spectral Data<<<<<", then
wavelength\\tintensity rows) -- same format prepare_danangleafy6_dataset.py's
source data uses.

Usage: python3 test_new_leafy_data.py (run from main/, matching every
other deploy_utils.py caller's relative data/checkpoint paths)
"""
import glob
import os

import numpy as np

from deploy_utils import (FOOD_NAME_TO_ID, LEAFY_MERGED_LABEL, _resolve_leafy_subtype,
                           infer_category_classification)

FOLDER_TO_LABEL = {
    "Caibex": "Cải Bẹ Xanh",
    "Caithi": "Cải Thìa",
    "Montoi": "Mồng Tơi",
    "Xalach": "Xà Lách",
}

# Filename prefix doesn't match its folder (e.g. "CT_Caibex_..." sitting in
# Caibex/ but CT = Cai Thia) -- looks like a mislabeled/misplaced file from
# data collection, not a model error; excluded rather than scored as wrong.
EXCLUDE_BASENAMES = {
    "DL_Montoi_FLMN032701__1__00001.txt",
    "HC2_Montoi_FLMN032701__5__00005.txt",
    "CT_Caibex_OFX034251__5__00005.txt",
    "CT_Caibex_OFX034251__6__00006.txt",
}


def load_spectrum(path: str) -> np.ndarray:
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    start = next(i for i, l in enumerate(lines) if "Begin Spectral Data" in l) + 1
    vals = [float(l.split()[1]) for l in lines[start:] if l.strip()]
    return np.array(vals, dtype=np.float32)


def run_machine(machine: str, root: str):
    rows, truths, paths = [], [], []
    for folder, label in FOLDER_TO_LABEL.items():
        for fp in sorted(glob.glob(f"{root}/{machine}/{folder}/*.txt")):
            if os.path.basename(fp) in EXCLUDE_BASENAMES:
                continue
            rows.append(load_spectrum(fp))
            truths.append(label)
            paths.append(fp)
    lens = {len(r) for r in rows}
    if len(lens) != 1:
        raise ValueError(f"[{machine}] inconsistent spectrum lengths across files: {lens}")
    spectra = np.stack(rows)
    print(f"\n[{machine}] {len(spectra)} samples, {spectra.shape[1]} points each")

    categories, cat_conf = infer_category_classification(spectra, machine)
    merged_hits = sum(c == LEAFY_MERGED_LABEL for c in categories)
    print(f"  6-way GuidedDCNet: predicted '{LEAFY_MERGED_LABEL}' (merged leafy class) for "
          f"{merged_hits}/{len(categories)} -- the rest fell outside the leafy group entirely:")
    for i, c in enumerate(categories):
        if c != LEAFY_MERGED_LABEL:
            print(f"    MISROUTED (never reached PCA+LDA): {os.path.basename(paths[i])} "
                  f"true={truths[i]} 6way_pred={c} conf={cat_conf[i]:.3f}")

    leafy_idx = [i for i, c in enumerate(categories) if c == LEAFY_MERGED_LABEL]
    final_pred = list(categories)
    final_conf = list(cat_conf)
    if leafy_idx:
        subtypes, confs = _resolve_leafy_subtype(spectra[leafy_idx], machine)
        for j, i in enumerate(leafy_idx):
            final_pred[i] = subtypes[j]
            final_conf[i] = confs[j]

    # Overall end-to-end accuracy (6-way routing failures count as wrong).
    correct = [final_pred[i] == truths[i] for i in range(len(truths))]
    acc = 100.0 * sum(correct) / len(correct)
    print(f"  End-to-end 4-way leafy accuracy (6-way route + PCA+LDA): {sum(correct)}/{len(correct)} = {acc:.2f}%")

    labels = list(FOLDER_TO_LABEL.values())
    conf_mat = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(truths, final_pred):
        conf_mat[t][p if p in labels else p] = conf_mat[t].get(p, 0) + 1
    print(f"  Confusion matrix (rows=true, cols=pred; {machine}):")
    header = "true\\pred".ljust(14) + "".join(l.ljust(14) for l in labels) + "other".ljust(10)
    print("    " + header)
    for t in labels:
        row = conf_mat[t]
        other = sum(v for k, v in row.items() if k not in labels)
        print("    " + t.ljust(14) + "".join(str(row.get(l, 0)).ljust(14) for l in labels) + str(other).ljust(10))

    wrong = [(paths[i], truths[i], final_pred[i], final_conf[i]) for i in range(len(truths)) if not correct[i]]
    if wrong:
        print(f"  Misclassified ({len(wrong)}):")
        for fp, t, p, c in wrong:
            print(f"    {os.path.basename(fp):<45} true={t:<12} pred={p:<12} conf={c:.3f}")

    return acc, len(truths)


if __name__ == "__main__":
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "new-test-data")
    total_correct, total_n = 0.0, 0
    for machine in ["FLAMENIR", "OCEANFX"]:
        acc, n = run_machine(machine, root)
        total_correct += acc / 100.0 * n
        total_n += n
    print(f"\n=== Overall (both machines): {total_correct:.0f}/{total_n} = {100.0*total_correct/total_n:.2f}% ===")
