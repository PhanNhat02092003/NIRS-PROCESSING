"""
Detailed multi-task evaluation for a saved SMART-NIR checkpoint.

Evaluates on the test split of the given fold (same 5-fold CV splits as
train.py), or on the whole dataset with --fold 0.

Reports:
  Task 1  — Classification accuracy, per-class F1, confusion matrix
  Task 2  — Regression: per-target R², RMSE, MAE, RPD (original units)
  Task 3  — Quality: quality head (Rapeseed N-/N+) or PASS/FAIL derived from
            the predicted class + regression outputs (Grainit, Mango)
  Scales  — selected streams, per-task gates and encoder MACs (MFSE models)
"""

import argparse
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, mean_absolute_error,
                             mean_squared_error, r2_score)

from dataset_benchmark import (DATASET_NAMES, DEFAULT_DATA_ROOT, QUALITY_CLASSES,
                               derive_quality, load_benchmark)
from train import build_model, make_splitter


# ─── Collect predictions ──────────────────────────────────────────────────────

@torch.no_grad()
def collect(model, X, device, batch_size):
    food_pred, reg_pred, aux_pred = [], [], []
    for start in range(0, len(X), batch_size):
        xb = torch.tensor(X[start:start + batch_size], dtype=torch.float32, device=device)
        fp, rp, ap = model.predict(xb)
        food_pred.append(fp.cpu().numpy())
        reg_pred.append(rp.cpu().numpy())
        if ap is not None:
            aux_pred.append(ap.cpu().numpy())
    aux_pred = np.concatenate(aux_pred) if aux_pred else None
    return np.concatenate(food_pred), np.vstack(reg_pred), aux_pred


# ─── Report printers ──────────────────────────────────────────────────────────

def print_food_report(food_true, food_pred, class_names, title="TASK 1 — CLASSIFICATION"):
    print("\n" + "═"*60)
    print(f"  {title}")
    print("═"*60)
    labels = list(range(len(class_names)))
    print(classification_report(food_true, food_pred, labels=labels,
                                target_names=class_names, digits=4, zero_division=0))
    print("Confusion matrix (rows = true, cols = predicted):")
    print(confusion_matrix(food_true, food_pred, labels=labels))


def print_regression_report(reg_true, reg_pred, target_names):
    print("\n" + "═"*60)
    print("  TASK 2 — REGRESSION  (original units)")
    print("═"*60)
    print(f"  {'Target':<14} {'N':>7} {'R²':>8} {'RMSE':>9} {'MAE':>9} {'RPD':>7}")
    print("  " + "-"*58)
    for j, t in enumerate(target_names):
        obs = ~np.isnan(reg_true[:, j])
        yt, yp = reg_true[obs, j], reg_pred[obs, j]
        rmse = np.sqrt(mean_squared_error(yt, yp))
        print(f"  {t:<14} {obs.sum():>7} {r2_score(yt, yp):>8.4f} {rmse:>9.4f} "
              f"{mean_absolute_error(yt, yp):>9.4f} {yt.std(ddof=1) / rmse:>7.3f}")


def print_quality_report(d, food_true, food_pred, reg_true, reg_pred, aux_true, aux_pred):
    if aux_pred is not None:
        q_true, q_pred, names = aux_true, aux_pred, d["aux_classes"]
        title = "TASK 3 — QUALITY (quality head)"
    else:
        args_ = (d["food_classes"], d["target_names"], d["quality_rules"])
        q_true = derive_quality(reg_true, food_true, *args_)
        q_pred = derive_quality(reg_pred, food_pred, *args_)
        keep = ~np.isnan(q_true)
        if (~keep).any():
            print(f"\n  [quality] {(~keep).sum()} samples skipped (required target missing)")
        q_true, q_pred = q_true[keep].astype(int), q_pred[keep].astype(int)
        names = QUALITY_CLASSES
        title = "TASK 3 — QUALITY (derived PASS/FAIL, predicted class + targets)"
    print_food_report(q_true, q_pred, names, title=title)
    print(f"Balanced accuracy: {balanced_accuracy_score(q_true, q_pred) * 100:.2f}%")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",      required=True, choices=DATASET_NAMES)
    p.add_argument("--checkpoint",   required=True)
    p.add_argument("--fold",         type=int, default=None,
                   help="1-5: evaluate on that fold's test split (default: parsed from "
                        "the checkpoint name fold<N>.pt); 0: whole dataset")
    p.add_argument("--data_root",    type=str, default=str(DEFAULT_DATA_ROOT))
    p.add_argument("--model_type",   type=str, default="lensnir",
                   choices=["lensnir", "csse", "flat", "spectraformer", "resnet1d",
                            "cnn1d", "deepspectra", "mocnn", "assadzadeh"])
    p.add_argument("--c_out",        type=int, default=64)
    p.add_argument("--n_layers",     type=int, default=6)
    p.add_argument("--n_heads",      type=int, default=6)
    p.add_argument("--h_hidden",     type=int, default=128)
    p.add_argument("--use_kan",      action="store_true", default=True)
    p.add_argument("--no_kan",       dest="use_kan", action="store_false")
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--ffn_type",     type=str, default="local_gated",
                   choices=["local_gated", "local_gated_iso", "dual_mlp"],
                   help="FFN variant — must match the training run")
    p.add_argument("--kernels",      type=str, default="5,9,17,33",
                   help="MFSE stem kernels — must match the training run")
    p.add_argument("--d_model",      type=int, default=None)
    p.add_argument("--scale_select", type=str, default="none",
                   choices=["none", "task", "shared"],
                   help="Must match the training run (gates are restored from the checkpoint)")
    p.add_argument("--gate_struct",  type=str, default="hier", choices=["hier", "flat"],
                   help="Must match the training run")
    p.add_argument("--stem",         type=str, default="linear", choices=["linear", "deep"],
                   help="Must match the training run")
    p.add_argument("--sim",          type=str, default="summary", choices=["summary", "none"],
                   help="Must match the training run")
    p.add_argument("--token_pool",   type=int, default=1, help="Must match the training run")
    p.add_argument("--readout",      type=str, default="mean", choices=["mean", "flat"],
                   help="Must match the training run")
    p.add_argument("--readout_rank", type=int, default=8, help="Must match the training run")
    p.add_argument("--cond_reg",     type=str, default="none", choices=["none", "soft", "detach"],
                   help="Must match the training run")
    p.add_argument("--task_linear",  type=str, default="none", nargs="?", const="all",
                   choices=["none", "reg", "all"], help="Must match the training run")
    p.add_argument("--export_pruned", type=str, default=None,
                   help="Optional path: save the physically pruned model "
                        "(only the selected streams) to this file")
    p.add_argument("--batch_size",   type=int, default=256)
    p.add_argument("--device",       default="auto")
    args = p.parse_args()

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))

    d = load_benchmark(args.dataset, args.data_root)
    X, y_food, y_reg, y_aux = d["X"], d["y_food"], d["y_reg"], d["y_aux"]

    fold = args.fold
    if fold is None:
        stem = args.checkpoint.rsplit("/", 1)[-1]
        fold = int(stem[4]) if stem.startswith("fold") and stem[4].isdigit() else 0
    if fold:
        splits = list(make_splitter(d["groups"]).split(X, y_food, d["groups"]))
        te_idx = splits[fold - 1][1]
        X, y_food, y_reg = X[te_idx], y_food[te_idx], y_reg[te_idx]
        y_aux = y_aux[te_idx] if y_aux is not None else None
        print(f"Evaluating fold {fold} test split: {len(te_idx):,} samples")
    else:
        print(f"Evaluating whole dataset: {len(X):,} samples")

    model_args = SimpleNamespace(**vars(args), n_food=len(d["food_classes"]),
                                 n_targets=len(d["target_names"]),
                                 n_aux=len(d["aux_classes"]), seq_len=d["seq_len"])
    model = build_model(model_args).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device,
                                     weights_only=True))

    food_pred, reg_pred, aux_pred = collect(model, X, device, args.batch_size)

    print_food_report(y_food, food_pred, d["food_classes"])
    print_regression_report(y_reg, reg_pred, d["target_names"])
    if aux_pred is not None or d["quality_rules"]:
        print_quality_report(d, y_food, food_pred, y_reg, reg_pred, y_aux, aux_pred)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params/1e3:.1f}K")
    if hasattr(model, "selection_summary"):
        sm = model.selection_summary()
        print(f"Scale streams: {sm['n_active']}/{len(sm['kernels'])} "
              f"kernels={sm['active_kernels']}  encoder ≈ {sm['macs_M']:.1f} MMACs "
              f"(all streams: {sm['macs_full_M']:.1f})")
        for t, g in zip(sm["tasks"], sm["gates"]):
            print(f"  {t:<8} " + " ".join(f"{k}:{v:.2f}" for k, v in zip(sm["kernels"], g)))
        if args.export_pruned and model.gates is not None:
            pruned = model.export_pruned()
            torch.save({"init_kwargs": pruned._init_kwargs,
                        "state_dict": pruned.state_dict()}, args.export_pruned)
            n_pr = sum(p.numel() for p in pruned.parameters())
            print(f"Pruned model ({n_pr/1e3:.1f}K params) → {args.export_pruned}")


if __name__ == "__main__":
    main()
