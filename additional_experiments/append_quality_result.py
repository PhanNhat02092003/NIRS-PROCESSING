"""Generic appender for benchmark/quality/{Dataset}.csv: loads each fold's
history JSON (written by food_classification.py's train()/train_diffusion(),
same val_acc/val_precision/val_recall/val_f1 keys for both SMART-NIR and
GuidedDCNet), takes the metrics at each fold's best-val_acc epoch (matching
how the engine itself picks its best checkpoint), and appends one
mean+std-across-folds row -- used by run_remaining_quality_benchmarks.sh
after each quality-grade training run finishes.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_utils import append_classification_row

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--history-glob", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--method", required=True)
    args = p.parse_args()

    files = sorted(glob.glob(args.history_glob))
    if not files:
        raise SystemExit(f"No history files matched: {args.history_glob}")

    accs, precs, recs, f1s = [], [], [], []
    for fp in files:
        with open(fp) as f:
            h = json.load(f)
        idx = int(np.argmax(h["val_acc"]))
        accs.append(h["val_acc"][idx] * 100.0)
        precs.append(h["val_precision"][idx] * 100.0)
        recs.append(h["val_recall"][idx] * 100.0)
        f1s.append(h["val_f1"][idx] * 100.0)

    append_classification_row(
        args.csv, args.method,
        np.mean(accs), np.std(accs), np.mean(precs), np.std(precs),
        np.mean(recs), np.std(recs), np.mean(f1s), np.std(f1s),
    )
    print(f"{args.method}: acc={np.mean(accs):.2f}+-{np.std(accs):.2f} "
          f"(n={len(files)} folds) -> {args.csv}")
