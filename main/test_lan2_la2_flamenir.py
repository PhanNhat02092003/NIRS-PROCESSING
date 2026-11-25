"""Evaluates the deployed FLAMENIR leafy-disambiguator fix against a THIRD,
genuinely independent scanning session: ../Lan2_La2(mat trong)/{Caibex,
Caithi,Montoi,Xalach}/*.txt (dated 2026-11-25, one day after the
new-test-data-2026-11 session that's already folded into training --
this one has never been seen by the model in any way).

Usage: python3 test_lan2_la2_flamenir.py (run from main/)
"""
import glob
import os

import numpy as np

from deploy_utils import FOOD_NAME_TO_ID, LEAFY_MERGED_LABEL, _resolve_leafy_subtype, infer_category_classification

FOLDER_TO_LABEL = {
    "Caibex": "Cải Bẹ Xanh",
    "Caithi": "Cải Thìa",
    "Montoi": "Mồng Tơi",
    "Xalach": "Xà Lách",
}
MACHINE = "FLAMENIR"


def load_spectrum(path: str) -> np.ndarray:
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    start = next(i for i, l in enumerate(lines) if "Begin Spectral Data" in l) + 1
    vals = [float(l.split()[1]) for l in lines[start:] if l.strip()]
    return np.array(vals, dtype=np.float32)


if __name__ == "__main__":
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Lan2_La2(mat trong)")
    rows, truths, paths = [], [], []
    for folder, label in FOLDER_TO_LABEL.items():
        for fp in sorted(glob.glob(f"{root}/{folder}/*.txt")):
            rows.append(load_spectrum(fp))
            truths.append(label)
            paths.append(fp)
    lens = {len(r) for r in rows}
    if len(lens) != 1:
        raise ValueError(f"inconsistent spectrum lengths across files: {lens}")
    spectra = np.stack(rows)
    print(f"[{MACHINE}] Lan2_La2(mat trong): {len(spectra)} samples, {spectra.shape[1]} points each")
    print("  " + ", ".join(f"{l}={truths.count(l)}" for l in FOLDER_TO_LABEL.values()))

    categories, cat_conf = infer_category_classification(spectra, MACHINE)
    merged_hits = sum(c == LEAFY_MERGED_LABEL for c in categories)
    print(f"  6-way GuidedDCNet: predicted '{LEAFY_MERGED_LABEL}' for {merged_hits}/{len(categories)}")
    for i, c in enumerate(categories):
        if c != LEAFY_MERGED_LABEL:
            print(f"    MISROUTED (never reached leafy resolver): {os.path.basename(paths[i])} "
                  f"true={truths[i]} 6way_pred={c} conf={cat_conf[i]:.3f}")

    leafy_idx = [i for i, c in enumerate(categories) if c == LEAFY_MERGED_LABEL]
    final_pred = list(categories)
    final_conf = list(cat_conf)
    if leafy_idx:
        subtypes, confs = _resolve_leafy_subtype(spectra[leafy_idx], MACHINE)
        for j, i in enumerate(leafy_idx):
            final_pred[i] = subtypes[j]
            final_conf[i] = confs[j]

    correct = [final_pred[i] == truths[i] for i in range(len(truths))]
    acc = 100.0 * sum(correct) / len(correct)
    print(f"  End-to-end 4-way leafy accuracy: {sum(correct)}/{len(correct)} = {acc:.2f}%")

    labels = list(FOLDER_TO_LABEL.values())
    conf_mat = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(truths, final_pred):
        conf_mat[t][p] = conf_mat[t].get(p, 0) + 1
    print("  Confusion matrix (rows=true, cols=pred):")
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
