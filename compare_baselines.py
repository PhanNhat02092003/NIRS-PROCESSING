"""Side-by-side comparison of SMART-NIR vs GuidedDCNet on the 9-class
vegetable classification task, for both machines. Standalone script, not
part of the progress report -- run after both engines have produced their
5-fold history files:
    history/category_classification/{machine}/smart_nir_classification_fold{n}.json
    history/category_classification_guideddcnet/{machine}/guideddcnet_classification_fold{n}.json

Usage: python3 compare_baselines.py
"""
import json
import os
import statistics as st

MACHINES = ["FLAMENIR", "OCEANFX"]
METRICS = [("val_acc", "Accuracy"), ("val_precision", "Precision"),
           ("val_recall", "Recall"), ("val_f1", "F1-Score")]
MODELS = [
    ("SMART-NIR", "history/category_classification/{machine}/smart_nir_classification_fold{fold}.json"),
    ("GuidedDCNet", "history/category_classification_guideddcnet/{machine}/guideddcnet_classification_fold{fold}.json"),
]


def load_fold_best(path_template, machine, n_folds=5):
    """For each fold, the model-selection criterion is best val_acc (both
    engines checkpoint on best val_acc) -- read the metrics at that epoch."""
    per_metric = {key: [] for key, _ in METRICS}
    missing = []
    for fold in range(1, n_folds + 1):
        path = path_template.format(machine=machine, fold=fold)
        if not os.path.exists(path):
            missing.append(fold)
            continue
        with open(path) as f:
            h = json.load(f)
        best_idx = h["val_acc"].index(max(h["val_acc"]))
        for key, _ in METRICS:
            per_metric[key].append(h[key][best_idx])
    return per_metric, missing


def fmt(values):
    if not values:
        return "  --  "
    mean = sum(values) / len(values) * 100
    std = st.pstdev(values) * 100 if len(values) > 1 else 0.0
    return f"{mean:5.2f}±{std:4.2f}%"


def main():
    for machine in MACHINES:
        print(f"\n=== {machine} ===")
        header = f"{'Metric':<12}" + "".join(f"{name:>18}" for name, _ in MODELS)
        print(header)
        print("-" * len(header))

        results = {}
        for name, template in MODELS:
            per_metric, missing = load_fold_best(template, machine)
            results[name] = per_metric
            if missing:
                print(f"  [{name}] missing folds: {missing}")

        for key, label in METRICS:
            row = f"{label:<12}"
            for name, _ in MODELS:
                row += f"{fmt(results[name][key]):>18}"
            print(row)


if __name__ == "__main__":
    main()
