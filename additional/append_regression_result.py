"""Generic appender for benchmark/regression/{Dataset}.csv.

Two history shapes exist in this project, both handled here:
  - EBAR (regression.py's train_for_substance): one history file per fold,
    single-value lists {"val_mae":[x], "val_rmse":[x], "val_r2":[x]}.
  - Deep models (regression.py's train(), or the XSpecMamba joint engine's
    train_joint_masked()): one history file per fold, PER-EPOCH lists --
    picks the epoch with the lowest "val_loss" if that key exists, else the
    last epoch.

mse isn't stored anywhere in either shape; computed here as rmse**2 to
match the mse_mean/mse_std columns benchmark_utils.append_regression_row
already expects (same units/scale as every existing committed
benchmark/regression/*.csv row).

--mae-key/--rmse-key/--r2-key default to "val_mae"/"val_rmse"/"val_r2"
(EBAR/SMART-NIR/NirMACNet's single-substance history); the XSpecMamba
joint engine's history instead has one key per substance, e.g.
"val_mae_Thiamethoxam" -- pass the full key name explicitly for that case.

Usage:
  python3 append_regression_result.py --history-glob "..." --csv "..." \
    --method "EBAR" --substance "Thiamethoxam"
  python3 append_regression_result.py --history-glob ".../xspecmamba_joint_fold*.json" \
    --csv "..." --method "XSpecMamba" --substance "Thiamethoxam" \
    --mae-key val_mae_Thiamethoxam --rmse-key val_rmse_Thiamethoxam --r2-key val_r2_Thiamethoxam
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_utils import append_regression_row


def best_epoch_values(h: dict, mae_key: str, rmse_key: str, r2_key: str):
    """Single-value list (EBAR) -> index 0. Per-epoch list (deep models) ->
    the epoch with lowest val_loss if present, else the last epoch."""
    values = h[r2_key]
    if len(values) == 1:
        idx = 0
    elif "val_loss" in h:
        idx = int(np.argmin(h["val_loss"]))
    else:
        idx = -1
    return h[mae_key][idx], h[rmse_key][idx], h[r2_key][idx]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--history-glob", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--substance", required=True)
    p.add_argument("--mae-key", default="val_mae")
    p.add_argument("--rmse-key", default="val_rmse")
    p.add_argument("--r2-key", default="val_r2")
    p.add_argument("--mae-log-key", default=None,
                    help="Optional: history key for the log1p-space MAE (e.g. val_mae_log). "
                         "Only meaningful alongside --rmse-log-key/--r2-log-key; omit all three "
                         "(the default) to keep the row exactly as before -- unaffected datasets "
                         "(Grainit/Mango/Rapeseed) never pass these.")
    p.add_argument("--rmse-log-key", default=None)
    p.add_argument("--r2-log-key", default=None)
    args = p.parse_args()

    files = sorted(glob.glob(args.history_glob))
    if not files:
        raise SystemExit(f"No history files matched: {args.history_glob}")

    maes, rmses, r2s = [], [], []
    log_keys_given = args.mae_log_key and args.rmse_log_key and args.r2_log_key
    maes_log, rmses_log, r2s_log = [], [], []
    for fp in files:
        with open(fp) as f:
            h = json.load(f)
        mae, rmse, r2 = best_epoch_values(h, args.mae_key, args.rmse_key, args.r2_key)
        maes.append(mae)
        rmses.append(rmse)
        r2s.append(r2 * 100.0)  # match benchmark_utils convention (r2 stored as a percentage)
        if log_keys_given:
            mae_l, rmse_l, r2_l = best_epoch_values(h, args.mae_log_key, args.rmse_log_key, args.r2_log_key)
            maes_log.append(mae_l)
            rmses_log.append(rmse_l)
            r2s_log.append(r2_l * 100.0)

    mses = [r ** 2 for r in rmses]

    log_kwargs = {}
    if log_keys_given:
        log_kwargs = dict(
            mae_log_mean=np.mean(maes_log), mae_log_std=np.std(maes_log),
            rmse_log_mean=np.mean(rmses_log), rmse_log_std=np.std(rmses_log),
            r2_log_mean=np.mean(r2s_log), r2_log_std=np.std(r2s_log),
        )

    append_regression_row(
        args.csv, args.substance, args.method,
        np.mean(mses), np.std(mses), np.mean(maes), np.std(maes),
        np.mean(rmses), np.std(rmses), np.mean(r2s), np.std(r2s),
        **log_kwargs,
    )
    log_suffix = f" r2(log)={np.mean(r2s_log):.2f}+-{np.std(r2s_log):.2f}" if log_keys_given else ""
    print(f"{args.substance}/{args.method}: rmse={np.mean(rmses):.4f}+-{np.std(rmses):.4f} "
          f"r2={np.mean(r2s):.2f}+-{np.std(r2s):.2f}{log_suffix} (n={len(files)} folds) -> {args.csv}")
