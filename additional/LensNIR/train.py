"""
Multi-task SMART-NIR training with 5-fold stratified cross-validation.

Tasks: sample classification (category) + multi-target regression + quality.
Quality is a second classification head for Rapeseed (N-/N+), and a
PASS/FAIL derived post-hoc from the regression outputs for Grainit / Mango.
Datasets: Rapeseed, Grainit, Mango (see dataset_benchmark.py).

Model selection uses an inner validation split carved from each training
fold (--val_frac); the outer test fold is used only for the final report.

Task-aware scale selection (--scale_select task|shared, --budget k):
  phase 1 (--epochs)       : gates + Lagrangian budget on the expected union
                             of task scale sets, target annealed S → k
  phase 2 (--ft_frac·epochs): gates frozen to their test-time values, the
                             pruned network is fine-tuned; the best epoch on
                             the validation split is kept

Usage:
  # All three datasets (paths resolved from --data_root / DATA_ROOT):
  python train.py

  # A subset:
  python train.py --dataset Rapeseed Grainit

  # Quick smoke-test:
  python train.py --dataset Grainit --epochs 2 --folds 1

  # Task-aware selection from 8 candidate scales, budget 3 streams:
  python train.py --kernels 5,7,9,13,17,25,33,49 \
      --scale_select task --budget 3
"""

import argparse
import csv
import io
import json
import math
import os
import random
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                              mean_absolute_error, mean_squared_error, r2_score)
from sklearn.utils import resample
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from dataset_benchmark import (DATASET_NAMES, DEFAULT_DATA_ROOT, derive_quality,
                               load_benchmark)
from models import (DEFAULT_KERNELS, LensNIR, FlatSMARTNIRMultiTask,
                              SpectraformerMultiTask, ResNet1DMultiTask,
                              CNN1DMultiTask, DeepSpectraMultiTask,
                              MultiOutputCNNMultiTask, AssadzadehMTCNN)

EXTERNAL_MODELS = {
    "spectraformer": SpectraformerMultiTask,
    "resnet1d":      ResNet1DMultiTask,
    "cnn1d":         CNN1DMultiTask,
    "deepspectra":   DeepSpectraMultiTask,
    "mocnn":         MultiOutputCNNMultiTask,
    "assadzadeh":    AssadzadehMTCNN,
}
from losses import BudgetConstraint, MultiTaskLoss

SEED = 43

# Per-dataset default batch sizes (override with --batch_size).
DATASET_BATCH = {"Rapeseed": 32, "Grainit": 32, "Mango": 256,
                  "DanangLeafy6_FLAMENIR": 64, "DanangLeafy6_OCEANFX": 64}

# Per-dataset default epochs (override with --epochs). Chosen by optimizer
# steps rather than epochs: after class balancing one epoch is ~97 steps on
# Rapeseed, ~31 on Grainit and ~692 on Mango.
DATASET_EPOCHS = {"Rapeseed": 100, "Grainit": 100, "Mango": 30,
                   "DanangLeafy6_FLAMENIR": 100, "DanangLeafy6_OCEANFX": 100}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─── LR schedule ─────────────────────────────────────────────────────────────

def cosine_warmup_lr(optimizer, epoch, n_epochs, warmup=None, lr_min=1e-6, lr_base=1e-3):
    if warmup is None:
        warmup = max(1, n_epochs // 10)   # 10 % of training
    if epoch < warmup:
        lr = lr_base * (epoch + 1) / warmup
    else:
        t = (epoch - warmup) / max(1, n_epochs - warmup)
        lr = lr_min + 0.5 * (lr_base - lr_min) * (1 + math.cos(math.pi * t))
    for pg in optimizer.param_groups:
        pg["lr"] = lr * pg.get("lr_scale", 1.0)
    return lr


# ─── Class balancing ─────────────────────────────────────────────────────────

def balance_food_classes(X, food_y, reg_y, aux_y, seed):
    classes, counts = np.unique(food_y, return_counts=True)
    n_max = counts.max()
    Xs, Fy, Ry, Ay = [], [], [], []
    for cls in classes:
        idx = np.where(food_y == cls)[0]
        idx_r = resample(idx, n_samples=n_max, replace=True, random_state=seed)
        Xs.append(X[idx_r]); Fy.append(food_y[idx_r])
        Ry.append(reg_y[idx_r]); Ay.append(aux_y[idx_r])
    return np.vstack(Xs), np.concatenate(Fy), np.vstack(Ry), np.concatenate(Ay)


def make_splitter(groups, n_splits=5, seed=SEED):
    """Grouped CV when the dataset has repeated scans of one sample (Mango)."""
    if groups is not None:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def inner_split(y_food, groups, val_frac, seed):
    """
    (fit_idx, val_idx) inside a training fold: one stratified (and, for
    Mango, fruit-grouped) split holding out ≈ val_frac of the samples.
    """
    n_splits = max(2, int(round(1.0 / val_frac)))
    sp = make_splitter(groups, n_splits=n_splits, seed=seed)
    return next(sp.split(np.zeros(len(y_food)), y_food, groups))


# ─── Single-fold evaluation ───────────────────────────────────────────────────

def metrics_from_predictions(food_true, food_pred, Y_true, Y_pred, info: Dict,
                             aux_true=None, aux_pred=None) -> Dict:
    """
    Classification, per-target regression (original units) and quality metrics
    from predictions. Shared by the deep models (evaluate_fold), the classical
    baselines (baselines_classical.py) and the post-hoc N×ST combination
    (combine_st.py), so every row of the comparison is scored identically.

    Quality: the learned head's predictions when given (Rapeseed), otherwise
    PASS/FAIL derived from the predicted class + predicted targets via
    info["quality_rules"], scored against the same rule on the true class +
    measured targets (samples with a required target missing are skipped).
    """
    target_names = info["target_names"]
    food_true, food_pred = np.asarray(food_true), np.asarray(food_pred)
    Y_true, Y_pred = np.asarray(Y_true, dtype=float), np.asarray(Y_pred, dtype=float)
    m = {
        "food_acc": accuracy_score(food_true, food_pred) * 100,
        "food_f1":  f1_score(food_true, food_pred, average="macro", zero_division=0),
    }
    for j, t in enumerate(target_names):
        obs = ~np.isnan(Y_true[:, j])
        yt, yp = Y_true[obs, j], Y_pred[obs, j]
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        m[f"r2_{t}"]   = float(r2_score(yt, yp)) if len(yt) > 1 else float("nan")
        m[f"rmse_{t}"] = rmse
        m[f"mae_{t}"]  = float(mean_absolute_error(yt, yp))
        m[f"rpd_{t}"]  = float(yt.std(ddof=1) / rmse) if rmse > 0 else float("nan")
    m["reg_r2"] = float(np.nanmean([m[f"r2_{t}"] for t in target_names]))

    if aux_pred is not None and len(aux_pred):
        q_true, q_pred = np.asarray(aux_true), np.asarray(aux_pred)
    elif info.get("quality_rules"):
        q_true = derive_quality(Y_true, food_true, info["food_classes"],
                                target_names, info["quality_rules"])
        q_pred = derive_quality(Y_pred, food_pred, info["food_classes"],
                                target_names, info["quality_rules"])
        keep = ~np.isnan(q_true)
        q_true, q_pred = q_true[keep].astype(int), q_pred[keep].astype(int)
    else:
        q_true = q_pred = None
    if q_true is not None:
        m["qual_acc"]  = accuracy_score(q_true, q_pred) * 100
        m["qual_f1"]   = f1_score(q_true, q_pred, average="macro", zero_division=0)
        m["qual_bacc"] = balanced_accuracy_score(q_true, q_pred) * 100
    return m


@torch.no_grad()
def predict_loader(model, loader, device, use_amp=False):
    """Predictions in original units: (food_true, food_pred, Y_true, Y_pred, aux_true, aux_pred)."""
    model.eval()
    food_true, food_pred, reg_true, reg_pred = [], [], [], []
    aux_true, aux_pred = [], []
    for X_b, food_b, reg_b, aux_b in loader:
        X_b = X_b.to(device)
        with torch.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            fl, rz, al = model(X_b)
        # Defensive: a transient fp16/AMP overflow can occasionally leave a
        # NaN/Inf in the outputs. Sanitizing keeps this epoch's metrics
        # computable instead of crashing the whole run.
        if not (torch.isfinite(fl).all() and torch.isfinite(rz).all()):
            tqdm.write("  [WARN] evaluate: non-finite values in model output — sanitizing for metrics")
            fl = torch.nan_to_num(fl, nan=0.0, posinf=0.0, neginf=0.0)
            rz = torch.nan_to_num(rz, nan=0.0, posinf=0.0, neginf=0.0)
        food_pred.append(fl.argmax(1).cpu().numpy())
        food_true.append(food_b.numpy())
        reg_pred.append(model.denormalize(rz).cpu().numpy())
        reg_true.append(reg_b.numpy())
        if al is not None:
            aux_pred.append(al.argmax(1).cpu().numpy())
            aux_true.append(aux_b.numpy())
    cat = lambda xs: np.concatenate(xs) if xs else None
    return (cat(food_true), cat(food_pred), np.vstack(reg_true), np.vstack(reg_pred),
            cat(aux_true), cat(aux_pred))


def evaluate_fold(model, loader, device, info: Dict, use_amp: bool = False) -> Dict:
    ft, fp, yt, yp, at, ap = predict_loader(model, loader, device, use_amp)
    return metrics_from_predictions(ft, fp, yt, yp, info, at, ap)


def has_quality(info: Dict) -> bool:
    return info.get("y_aux") is not None or bool(info.get("quality_rules"))


def metric_keys(info: Dict) -> List[str]:
    keys = ["food_acc", "food_f1", "reg_r2"]
    for t in info["target_names"]:
        keys += [f"r2_{t}", f"rmse_{t}", f"mae_{t}", f"rpd_{t}"]
    if has_quality(info):
        keys += ["qual_acc", "qual_f1", "qual_bacc"]
    return keys


# ─── Single-fold training ─────────────────────────────────────────────────────

def _composite(m: Dict, st_task: str, use_aux: bool) -> float:
    """Checkpoint-selection score. Derived quality (Grainit/Mango) is a
    function of food + reg, so it is not added; a learned quality head is."""
    r2 = m["reg_r2"] if not math.isnan(m["reg_r2"]) else -1.0
    if st_task == "food":
        return m["food_acc"] / 100.0
    if st_task == "reg":
        return r2
    if st_task == "aux":
        return m["qual_acc"] / 100.0
    return m["food_acc"] / 100.0 + r2 + (m["qual_acc"] / 100.0 if use_aux else 0.0)


def _make_loader(X, y_food, y_reg, y_aux, batch_size, shuffle):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y_food, dtype=torch.long),
                       torch.tensor(y_reg, dtype=torch.float32),
                       torch.tensor(y_aux, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_fold(train, val, test, args, device, fold, info) -> Dict:
    """
    train / val / test: tuples (X, y_food, y_reg, y_aux). y_aux are
    quality-head labels (all zeros, unused, when the dataset has none).
    `val` may be None (--val_frac 0): the last epoch is then kept.

    Returns the test metrics of the selected checkpoint, with the matching
    validation metrics under the "val_" prefix and, for MFSE models, the
    selected scales ("n_active", "macs_M").
    """
    X_tr, y_food_tr, y_reg_tr, y_aux_tr = train

    # Target z-score statistics from the (unbalanced) fitting split only.
    reg_mean = np.nanmean(y_reg_tr, axis=0)
    reg_std  = np.maximum(np.nanstd(y_reg_tr, axis=0), 1e-8)

    if args.balance:
        X_tr, y_food_tr, y_reg_tr, y_aux_tr = balance_food_classes(
            X_tr, y_food_tr, y_reg_tr, y_aux_tr, seed=SEED + fold
        )

    # Training targets are z-scored; val/test targets stay in original units.
    tr_loader = _make_loader(X_tr, y_food_tr, (y_reg_tr - reg_mean) / reg_std, y_aux_tr,
                             args.batch_size, shuffle=True)
    va_loader = _make_loader(*val, args.batch_size, shuffle=False) if val is not None else None
    te_loader = _make_loader(*test, args.batch_size, shuffle=False)

    model = build_model(args).to(device)
    model.set_target_stats(reg_mean, reg_std)
    if args.input_norm != "none":
        # statistics of the fitting split only (before class balancing)
        x_mean = train[0].mean(axis=0)
        x_std = (train[0].std(axis=0) + 1e-6 if args.input_norm == "zscore"
                 else np.ones(train[0].shape[1], dtype=np.float32))
        model.set_input_stats(x_mean, x_std)

    _st_task = getattr(args, "st_task", "all")
    _is_st   = _st_task != "all"
    _use_aux = args.n_aux > 0

    # ── scale selection set-up ───────────────────────────────────────────────
    gated = getattr(model, "gates", None) is not None
    n_ep_prune = args.epochs
    n_ep_dense = int(round(args.dense_frac * n_ep_prune)) if gated else 0
    n_ep_ft = int(math.ceil(args.ft_frac * args.epochs)) if gated else 0
    n_ep_total = n_ep_prune + n_ep_ft
    steps_per_epoch = len(tr_loader)

    criterion = MultiTaskLoss(
        weighting="fixed" if _is_st else args.loss_weighting,
        use_aux=_use_aux,
    ).to(device)

    gate_params = list(model.gates.parameters()) if gated else []
    gate_ids = {id(p) for p in gate_params}
    weight_params = [p for p in model.parameters() if id(p) not in gate_ids]
    if not _is_st:
        weight_params += list(criterion.parameters())
    param_groups = [{"params": weight_params, "weight_decay": args.weight_decay}]
    if gated:
        param_groups.append({"params": gate_params, "weight_decay": 0.0,
                             "lr_scale": args.gate_lr_scale})
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)

    budget = lam_opt = None
    if gated:
        S = model.encoder.n_scales
        k = args.budget if args.budget is not None else S
        with torch.no_grad():
            c0 = float(model.expected_cost(args.cost))
        budget = BudgetConstraint(
            c0, k,
            warmup_steps=int(args.prune_warmup * (n_ep_prune - n_ep_dense) * steps_per_epoch)
        ).to(device)
        # λ ascends (maximize=True): min over θ, max over λ of the Lagrangian.
        lam_opt = torch.optim.Adam(budget.parameters(), lr=args.lambda_lr, maximize=True)

    best_composite, best_state, best_val = -math.inf, None, None
    start_epoch = 0
    # EMA of the weights (--ema d): evaluated and checkpointed instead of the
    # raw weights. With gates it starts with the fine-tuning phase (fixed
    # architecture), so gate sampling never enters the average.
    ema, ema_n = None, 0

    use_amp = getattr(args, "amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    resume_path = Path(args.save_dir) / f"fold{fold+1}_resume.pt" if args.save_dir else None
    final_path  = Path(args.save_dir) / f"fold{fold+1}.pt" if args.save_dir else None
    _skipped_via_final = False
    if getattr(args, "resume", True) and final_path and final_path.exists():
        # This fold already finished in a previous run — load the final
        # weights and skip straight to evaluation.
        try:
            model.load_state_dict(torch.load(final_path, map_location=device, weights_only=True))
            start_epoch = n_ep_total
            _skipped_via_final = True
            tqdm.write(f"  [SKIP] fold{fold+1}: final checkpoint already exists at {final_path}")
        except Exception as e:
            tqdm.write(f"  [WARN] fold{fold+1}: final checkpoint unreadable "
                       f"({type(e).__name__}: {e}) — falling back to resume/retrain.")
    if not _skipped_via_final and getattr(args, "resume", True) and resume_path and resume_path.exists():
        try:
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            criterion.load_state_dict(ckpt["criterion"])
            if budget is not None and ckpt.get("budget") is not None:
                budget.load_state_dict(ckpt["budget"])
                lam_opt.load_state_dict(ckpt["lam_opt"])
            if use_amp and ckpt.get("scaler") is not None:
                scaler.load_state_dict(ckpt["scaler"])
            start_epoch = ckpt["epoch"]
            best_composite = ckpt["best_composite"]
            best_state = ckpt["best_state"]
            best_val = ckpt.get("best_val")
            ema, ema_n = ckpt.get("ema"), ckpt.get("ema_n", 0)
            tqdm.write(f"  [RESUME] fold{fold+1}: continuing from epoch {start_epoch+1}/{n_ep_total} "
                       f"(best_composite so far = {best_composite:.4f})")
        except Exception as e:
            # A corrupted/truncated resume file must not crash the whole run —
            # fall back to training this one fold from epoch 0 instead.
            tqdm.write(f"  [RESUME] fold{fold+1}: resume checkpoint unreadable "
                       f"({type(e).__name__}: {e}) — starting this fold from epoch 0.")
            start_epoch = 0
            best_composite, best_state, best_val = -math.inf, None, None

    target_names = info["target_names"]
    hist_cols = ["food_acc", "food_f1", "reg_r2"] + [
        f"{k}_{t}" for t in target_names for k in ("r2", "rmse")
    ] + (["qual_acc", "qual_f1"] if has_quality(info) else [])
    sel_loader = va_loader if va_loader is not None else te_loader
    _hist_f = None
    if start_epoch < n_ep_total:
        if args.save_dir:
            _hist_dir = Path(args.save_dir)
            _hist_dir.mkdir(parents=True, exist_ok=True)
            _hist_f = open(_hist_dir / f"fold{fold+1}_history.csv",
                           "a" if start_epoch > 0 else "w")
            if start_epoch == 0:
                _hist_f.write("epoch,phase,lr,loss,loss_food,loss_reg,loss_aux,cost,target,"
                              + ",".join(hist_cols) + ",n_active,composite\n")
                _hist_f.flush()

        epoch_bar = tqdm(range(start_epoch, n_ep_total), desc=f"Fold {fold+1}", unit="ep",
                         dynamic_ncols=True, file=sys.stderr,
                         initial=start_epoch, total=n_ep_total)
        for epoch in epoch_bar:
            finetune = gated and epoch >= n_ep_prune
            dense = gated and epoch < n_ep_dense
            if gated:
                model.gates_on = not dense
            if finetune:
                model.freeze_gates()          # idempotent; also right after a resume
                lr = cosine_warmup_lr(optimizer, epoch - n_ep_prune, n_ep_ft, warmup=1,
                                      lr_base=args.lr * args.ft_lr_scale)
            else:
                lr = cosine_warmup_lr(optimizer, epoch, n_ep_prune, lr_base=args.lr)
            model.train()
            total_loss = l1_acc = l2_acc = l3_acc = cost_acc = 0.0
            tgt = float("nan")

            batch_bar = tqdm(tr_loader, desc="  train", leave=False,
                             unit="b", dynamic_ncols=True, file=sys.stderr)
            for b_idx, (X_b, food_b, reg_b, aux_b) in enumerate(batch_bar):
                X_b, food_b = X_b.to(device), food_b.to(device)
                reg_b, aux_b = reg_b.to(device), aux_b.to(device)
                step = (epoch - n_ep_dense) * steps_per_epoch + b_idx   # gated-phase step

                optimizer.zero_grad()
                if lam_opt is not None:
                    lam_opt.zero_grad()
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    fl, rz, al = model(X_b)
                    loss, l1, l2, l3 = criterion(fl, rz, food_b, reg_b, al, aux_b)
                    eff_loss = {"food": l1, "reg": l2, "aux": l3}.get(_st_task, loss)
                if gated and not finetune and not dense:
                    cost = model.expected_cost(args.cost)
                    tgt = budget.target(step)
                    eff_loss = eff_loss + budget(cost, step)
                    if args.task_sparsity > 0 and model.gates.task is not None:
                        # Pushes each task to read few scales; acts on the
                        # task gates (∂/∂π(u_tk) = μ·π(s_k)/n_tasks, no product).
                        eff_loss = eff_loss + args.task_sparsity * model.expected_cost("sum")
                    cost_acc += cost.item()

                if not torch.isfinite(eff_loss):
                    # A single numerically unstable batch must not be allowed to
                    # poison the model weights via optimizer.step — skip it.
                    tqdm.write(f"  [WARN] fold{fold+1} epoch{epoch+1}: non-finite loss "
                               f"— skipping this batch")
                    del fl, rz, al, loss, l1, l2, l3, eff_loss
                    optimizer.zero_grad(set_to_none=True)
                    continue

                scaler.scale(eff_loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(weight_params, max_norm=1.0)
                scaler.step(optimizer)
                if gated and not finetune and not dense:
                    scaler.unscale_(lam_opt)
                    scaler.step(lam_opt)
                scaler.update()
                if args.ema > 0 and ((not gated) or finetune):
                    with torch.no_grad():
                        sd = model.state_dict()
                        if ema is None:
                            ema = {k_: v.detach().clone() for k_, v in sd.items()}
                        else:
                            dcy = min(args.ema, (1 + ema_n) / (10 + ema_n))
                            for k_, v in sd.items():
                                if v.dtype.is_floating_point:
                                    ema[k_].mul_(dcy).add_(v.detach(), alpha=1 - dcy)
                                else:
                                    ema[k_].copy_(v)
                        ema_n += 1

                if hasattr(criterion, "uncertainty"):
                    # Kendall uncertainty weighting has no natural floor/ceiling on
                    # log_sigma; clamping keeps exp(-2*log_sigma) within
                    # [exp(-6), exp(6)], far inside fp16 range under AMP.
                    with torch.no_grad():
                        criterion.uncertainty.log_sigma.clamp_(-3.0, 3.0)

                total_loss += loss.item()
                l1_acc += l1.item(); l2_acc += l2.item(); l3_acc += l3.item()
                batch_bar.set_postfix(loss=f"{loss.item():.4f}")

            n = len(tr_loader)
            epoch_bar.set_postfix(L=f"{total_loss/n:.4f}",
                                  food=f"{l1_acc/n:.3f}",
                                  reg=f"{l2_acc/n:.3f}",
                                  aux=f"{l3_acc/n:.3f}")

            w = criterion.task_weights.cpu().numpy()
            raw_sd = None
            if ema is not None:                       # evaluate the averaged weights
                raw_sd = {k_: v.clone() for k_, v in model.state_dict().items()}
                model.load_state_dict(ema)
            m = evaluate_fold(model, sel_loader, device, info, use_amp=use_amp)
            composite = _composite(m, _st_task, _use_aux)
            n_active = (model.selection_summary()["n_active"]
                        if hasattr(model, "selection_summary") else float("nan"))
            # With gates, only the fine-tuning phase (fixed architecture) is
            # eligible for checkpoint selection.
            eligible = (not gated) or finetune
            if eligible and (composite > best_composite or va_loader is None):
                best_composite, best_val = composite, m
                best_state = {k_: v.cpu().clone() for k_, v in model.state_dict().items()}
            if raw_sd is not None:
                model.load_state_dict(raw_sd)
            tqdm.write(
                f"  Fold {fold+1} [{epoch+1:3d}/{n_ep_total}]"
                f"{' ft' if finetune else (' dense' if dense else '')} lr={lr:.1e} "
                f"L={total_loss/n:.4f} "
                f"(food={l1_acc/n:.3f} reg={l2_acc/n:.3f} aux={l3_acc/n:.3f}) "
                f"w=[{','.join(f'{x:.2f}' for x in w)}]  "
                + (f"C={cost_acc/n:.2f}→{tgt:.2f} S_act={n_active} " if gated and not finetune and not dense
                   else (f"S_act={n_active} " if gated else ""))
                + f"{'val' if va_loader is not None else 'test'}: "
                f"food={m['food_acc']:.2f}% R²={m['reg_r2']:.3f} "
                + (f"qual={m['qual_acc']:.2f}% " if "qual_acc" in m else "")
                + f"comp={composite:.4f}"
            )
            if _hist_f:
                _hist_f.write(
                    f"{epoch+1},{'ft' if finetune else ('dense' if dense else 'gated')},{lr:.6e},{total_loss/n:.6f},"
                    f"{l1_acc/n:.6f},{l2_acc/n:.6f},{l3_acc/n:.6f},"
                    f"{cost_acc/n:.6f},{tgt:.6f},"
                    + ",".join(f"{m[c]:.6f}" for c in hist_cols)
                    + f",{n_active},{composite:.6f}\n"
                )
                _hist_f.flush()

            # Checkpoint progress every epoch so a killed/restarted process can
            # resume here. Written atomically (tmp file + rename) so a process
            # killed mid-write can never corrupt the previous good checkpoint.
            if resume_path:
                tmp_path = resume_path.with_suffix(".pt.tmp")
                torch.save({
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "criterion": criterion.state_dict(),
                    "budget": budget.state_dict() if budget is not None else None,
                    "lam_opt": lam_opt.state_dict() if lam_opt is not None else None,
                    "scaler": scaler.state_dict() if use_amp else None,
                    "best_composite": best_composite,
                    "best_state": best_state,
                    "best_val": best_val,
                    "ema": ema,
                    "ema_n": ema_n,
                }, tmp_path)
                tmp_path.replace(resume_path)  # atomic on POSIX

        if _hist_f:
            _hist_f.close()
    else:
        tqdm.write(f"  [SKIP] fold{fold+1}: resume checkpoint already at "
                   f"{start_epoch}/{n_ep_total} epochs")

    if best_state:
        model.load_state_dict({k_: v.to(device) for k_, v in best_state.items()})

    preds = predict_loader(model, te_loader, device, use_amp)
    m_final = metrics_from_predictions(*preds[:4], info, preds[4], preds[5])
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        np.savez(Path(args.save_dir) / f"fold{fold+1}_test_pred.npz",
                 food_true=preds[0], food_pred=preds[1], reg_true=preds[2],
                 reg_pred=preds[3],
                 aux_true=preds[4] if preds[4] is not None else np.array([]),
                 aux_pred=preds[5] if preds[5] is not None else np.array([]))
    if va_loader is not None:
        m_val = evaluate_fold(model, va_loader, device, info, use_amp=use_amp)
        m_final.update({f"val_{k_}": v for k_, v in m_val.items()})
    summary = None
    if hasattr(model, "selection_summary"):
        summary = model.selection_summary()
        m_final["n_active"] = summary["n_active"]
        m_final["macs_M"] = summary["macs_M"]

    if args.save_dir and final_path:
        final_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_final = final_path.with_suffix(".pt.tmp")
        torch.save(model.state_dict(), tmp_final)
        tmp_final.replace(final_path)  # atomic — never leaves a corrupt fold{N}.pt
        with open(final_path.with_name(f"fold{fold+1}_metrics.json"), "w") as f:
            json.dump({"metrics": m_final, "selection": summary,
                       "config": {k_: (list(v) if isinstance(v, tuple) else v)
                                  for k_, v in vars(args).items()
                                  if isinstance(v, (int, float, str, bool, list, tuple,
                                                    type(None)))}},
                      f, indent=1, default=float)
        # Training finished for this fold — the resume checkpoint is no longer needed.
        if resume_path and resume_path.exists():
            resume_path.unlink()

    return m_final


def parse_kernels(spec) -> tuple:
    if spec is None or spec == "":
        return DEFAULT_KERNELS
    if isinstance(spec, (list, tuple)):
        return tuple(int(k) for k in spec)
    return tuple(int(k) for k in str(spec).split(","))


def build_model(args) -> nn.Module:
    _model_type = getattr(args, "model_type", "lensnir")  # "csse": legacy name in saved configs
    if _model_type in EXTERNAL_MODELS:
        # External baselines — own hyperparameters, not the MFSE family's
        # c_out/n_layers/n_heads/h_hidden.
        return EXTERNAL_MODELS[_model_type](
            n_food_classes=args.n_food,
            n_targets=args.n_targets,
            n_aux_classes=args.n_aux,
            seq_len=args.seq_len,
            dropout=args.dropout,
        )
    if _model_type == "flat":
        return FlatSMARTNIRMultiTask(
            n_food_classes=args.n_food, n_targets=args.n_targets,
            n_aux_classes=args.n_aux, c_out=args.c_out, n_layers=args.n_layers,
            n_heads=args.n_heads, h_hidden=args.h_hidden, seq_len=args.seq_len,
            use_kan=args.use_kan, dropout=args.dropout,
        )
    return LensNIR(
        kernels=parse_kernels(getattr(args, "kernels", None)),
        scale_select=getattr(args, "scale_select", "none"),
        gate_init_prob=getattr(args, "gate_init_prob", 0.5),
        gate_struct=getattr(args, "gate_struct", "hier"),
        stem=getattr(args, "stem", "linear"),
        sim_mode=getattr(args, "sim", "summary"),
        token_pool=getattr(args, "token_pool", 1),
        readout=getattr(args, "readout", "mean"),
        readout_rank=getattr(args, "readout_rank", 8),
        task_linear=getattr(args, "task_linear", False),
        cond_reg=getattr(args, "cond_reg", "none"),
        fusion=getattr(args, "fusion", "task"),
        n_food_classes=args.n_food,
        n_targets=args.n_targets,
        n_aux_classes=args.n_aux,
        c_out=args.c_out,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        h_hidden=args.h_hidden,
        seq_len=args.seq_len,
        use_kan=args.use_kan,
        dropout=args.dropout,
        ffn_type=getattr(args, "ffn_type", "local_gated"),
        d_model=getattr(args, "d_model", None),
    )


# ─── Cross-validation loop ────────────────────────────────────────────────────

def print_summary(tag: str, results: List[Dict], info: Dict):
    print(f"\n{'─'*55}")
    print(f"  [{tag}]  {len(results)}-Fold Average")
    print(f"{'─'*55}")
    for k in metric_keys(info):
        vals = [r[k] for r in results]
        print(f"  {k:<24} {np.mean(vals):.4f} ± {np.std(vals, ddof=1) if len(vals) > 1 else 0.0:.4f}")


def save_cv_results(save_dir: str, folds: List[int], results: List[Dict], info: Dict,
                    keys: List[str] = None):
    """Per-fold metrics + mean/std rows → {save_dir}/cv_results.csv."""
    if keys is None:
        keys = [k for k in metric_keys(info) if k in results[0]]
        keys += [f"val_{k}" for k in metric_keys(info) if f"val_{k}" in results[0]]
        keys += [k for k in ("n_active", "macs_M") if k in results[0]]
    path = Path(save_dir) / "cv_results.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold"] + keys)
        for fold, r in zip(folds, results):
            w.writerow([fold] + [f"{r[k]:.6f}" for k in keys])
        vals = np.array([[r[k] for k in keys] for r in results])
        w.writerow(["mean"] + [f"{v:.6f}" for v in vals.mean(axis=0)])
        if len(results) > 1:
            w.writerow(["std"] + [f"{v:.6f}" for v in vals.std(axis=0, ddof=1)])
    print(f"  → {path}")


def run(args, device):
    set_seed(SEED)
    folds_to_run = {int(f) for f in args.folds.split(",")}
    save_root = args.save_dir

    for name in args.dataset:
        torch.cuda.empty_cache()
        d = load_benchmark(name, args.data_root)
        X, y_food, y_reg, groups = d["X"], d["y_food"], d["y_reg"], d["groups"]
        y_aux = d["y_aux"] if d["y_aux"] is not None else np.zeros(len(X), dtype=np.int64)
        target_names = d["target_names"]

        args.n_food     = len(d["food_classes"])
        args.n_targets  = len(target_names)
        args.n_aux      = len(d["aux_classes"])
        if args.st_task == "aux" and not args.n_aux:
            print(f"  [SKIP] {name}: --st_task aux needs a quality head "
                  f"(quality here is derived from regression)")
            continue
        args.seq_len    = d["seq_len"]
        args.batch_size = args.batch_size_override or DATASET_BATCH[name]
        args.epochs     = args.epochs_override or DATASET_EPOCHS[name]
        args.save_dir   = str(Path(save_root) / name) if save_root else None

        print(f"\n{'#'*62}")
        print(f"  DATASET: {name}  ({len(X):,} samples)  seq_len={args.seq_len}  "
              f"batch={args.batch_size}  epochs={args.epochs}")
        print(f"  classes: {d['food_classes']}")
        print(f"  targets: {target_names}")
        if args.n_aux:
            print(f"  quality head: {d['aux_classes']}")
        elif d["quality_rules"]:
            print(f"  quality (derived PASS/FAIL): {d['quality_rules']}")
        print(f"  folds this process: {sorted(folds_to_run)}  device: {device}")
        print(f"{'#'*62}")

        splitter = make_splitter(groups)
        results, done_folds = [], []
        for fold, (tr_idx, te_idx) in enumerate(splitter.split(X, y_food, groups)):
            if (fold + 1) not in folds_to_run:
                continue
            print(f"\n{'═'*62}")
            print(f"  FOLD {fold+1}/5  [{name}]  "
                  f"train={len(tr_idx):,}  test={len(te_idx):,}")
            print(f"{'═'*62}")

            if args.val_frac > 0:
                fit_rel, val_rel = inner_split(
                    y_food[tr_idx], None if groups is None else groups[tr_idx],
                    args.val_frac, seed=SEED + fold)
                fit_idx, val_idx = tr_idx[fit_rel], tr_idx[val_rel]
                print(f"  inner split: fit={len(fit_idx):,}  val={len(val_idx):,}")
            else:
                fit_idx, val_idx = tr_idx, None
            part = lambda idx: (X[idx], y_food[idx], y_reg[idx], y_aux[idx])
            try:
                mc = train_fold(
                    part(fit_idx), part(val_idx) if val_idx is not None else None,
                    part(te_idx), args, device, fold, d,
                )
            except Exception as e:
                # A fatal error in one fold must not take the other folds down
                # with it — the per-epoch resume checkpoint is untouched.
                print(f"  [FOLD FAILED] {name} fold{fold+1}: {type(e).__name__}: {e}")
                print(f"  Skipping to next fold; {name} fold{fold+1} can be resumed later.")
                torch.cuda.empty_cache()
                continue
            torch.cuda.empty_cache()
            results.append(mc)
            done_folds.append(fold + 1)
            print(f"  food={mc['food_acc']:.2f}%  F1={mc['food_f1']:.3f}  "
                  + "  ".join(f"R²[{t}]={mc[f'r2_{t}']:.3f}" for t in target_names)
                  + (f"  qual={mc['qual_acc']:.2f}%" if "qual_acc" in mc else "")
                  + (f"  scales={mc['n_active']} ({mc['macs_M']:.0f} MMACs)"
                     if "n_active" in mc else ""))

        if results:
            print_summary(name, results, d)
            if args.save_dir:
                save_cv_results(args.save_dir, done_folds, results, d)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(description="Train multi-task SMART-NIR")
    # Data
    p.add_argument("--dataset",         type=str,   nargs="+",
                   default=DATASET_NAMES, choices=DATASET_NAMES,
                   help="Dataset(s) to run, each with its own independent 5-fold CV "
                        "(default: all)")
    p.add_argument("--data_root",       type=str,   default=str(DEFAULT_DATA_ROOT),
                   help="Folder containing the dataset directories (or set DATA_ROOT)")
    p.add_argument("--balance",         dest="balance", action="store_true", default=True,
                   help="Upsample minority classes in the training fold (default: on)")
    p.add_argument("--no_balance",      dest="balance", action="store_false")
    # Architecture
    p.add_argument("--c_out",           type=int,   default=64)
    p.add_argument("--n_layers",        type=int,   default=6)
    p.add_argument("--n_heads",         type=int,   default=6)
    p.add_argument("--h_hidden",        type=int,   default=128)
    p.add_argument("--use_kan",         action="store_true", default=True)
    p.add_argument("--no_kan",          dest="use_kan", action="store_false")
    p.add_argument("--dropout",         type=float, default=0.1)
    p.add_argument("--weight_decay",    type=float, default=1e-4,
                   help="AdamW weight decay of the network weights (gates: 0)")
    p.add_argument("--stem",            type=str,   default="linear", choices=["linear", "deep"],
                   help="MFSE stem: linear = one strided conv per scale; deep = "
                        "conv(k)+BN+GELU+conv(3, stride 4)")
    p.add_argument("--token_pool",      type=int,   default=1,
                   help="MFSE: average this many neighbouring tokens after the stem "
                        "(T = ceil(L/(4·p)); 1 = no pooling)")
    p.add_argument("--ema",             type=float, default=0.0,
                   help="EMA decay of the weights used for evaluation / checkpoint "
                        "selection (e.g. 0.999; 0 = off)")
    p.add_argument("--readout",         type=str,   default="mean", choices=["mean", "flat"],
                   help="MFSE stream readout: mean of tokens or a "
                        "position-specific low-rank flatten readout")
    p.add_argument("--readout_rank",    type=int,   default=8,
                   help="--readout flat: dims kept per token before flattening")
    p.add_argument("--task_linear",     type=str,   default="none", nargs="?", const="all",
                   choices=["none", "reg", "all"],
                   help="--readout flat: per-task linear readout of the fused token maps, "
                        "added to the property outputs (reg) or to every output (all)")
    p.add_argument("--cond_reg",        type=str,   default="none",
                   choices=["none", "soft", "detach"],
                   help="condition every property head on the predicted probabilities "
                        "of every classification label (soft, end-to-end local calibration)")
    p.add_argument("--sim",             type=str,   default="summary",
                   choices=["summary", "none"],
                   help="Scale Interaction Module: mix mean-pooled stream summaries "
                        "(summary); none = no cross-scale exchange (ablation)")
    p.add_argument("--fusion",          type=str,   default="task", choices=["task", "shared"],
                   help="task: one scale fusion per task (default); shared: one fusion "
                        "for all tasks (ablation, fixed scales only)")
    p.add_argument("--ffn_type",        type=str,   default="local_gated",
                   choices=["local_gated", "local_gated_iso", "dual_mlp"],
                   help="FFN variant: local_gated (default) | local_gated_iso | dual_mlp")
    p.add_argument("--kernels",         type=str,   default=",".join(map(str, DEFAULT_KERNELS)),
                   help="Odd stem kernels of the MFSE streams, comma-separated "
                        "(default 5,9,17,33; candidate set for selection: "
                        "5,7,9,13,17,25,33,49)")
    p.add_argument("--d_model",         type=int,   default=None,
                   help="Token width D (default ceil(4*c_out/n_heads)*n_heads = 258)")
    # Scale selection
    p.add_argument("--scale_select",    type=str,   default="none",
                   choices=["none", "task", "shared"],
                   help="none: fixed streams; task: one hard-concrete gate per "
                        "(task, scale); shared: one gate per scale for all tasks")
    p.add_argument("--cost",            type=str,   default="union", choices=["union", "sum"],
                   help="union: expected size of the union of task scale sets "
                        "(true encoder cost); sum: mean per-task count (ablation)")
    p.add_argument("--budget",          type=float, default=None,
                   help="Target number of computed streams k (default S = no pruning)")
    p.add_argument("--dense_frac",      type=float, default=0.3,
                   help="Fraction of phase-1 epochs trained densely (all gates = 1) "
                        "before gates and budget are switched on")
    p.add_argument("--prune_warmup",    type=float, default=0.5,
                   help="Fraction of the gated epochs over which the target goes "
                        "from the initial cost to k")
    p.add_argument("--ft_frac",         type=float, default=0.3,
                   help="Fine-tuning epochs with frozen gates, as a fraction of --epochs")
    p.add_argument("--ft_lr_scale",     type=float, default=0.1,
                   help="Peak LR of the fine-tuning phase relative to --lr")
    p.add_argument("--gate_lr_scale",   type=float, default=50.0,
                   help="LR multiplier of the gate log-alpha parameters")
    p.add_argument("--gate_init_prob",  type=float, default=0.832,
                   help="Initial P(gate open) when gates are switched on; 0.832 ⇔ "
                        "log α = 0, the Louizos et al. default (keeps the union-cost "
                        "gradient Π(1-π) away from 0)")
    p.add_argument("--gate_struct",     type=str,   default="hier", choices=["hier", "flat"],
                   help="task selection gates: hier = scale gate × task gate "
                        "(z = s_k·u_tk); flat = task gates only (ablation; the "
                        "noisy-OR union cost then has vanishing gradients)")
    p.add_argument("--task_sparsity",   type=float, default=0.05,
                   help="μ: weight of the mean per-task expected number of read "
                        "scales, added to the budget term (task gates only)")
    p.add_argument("--lambda_lr",       type=float, default=0.01,
                   help="LR of the Lagrange multipliers (gradient ascent)")
    p.add_argument("--input_norm",      type=str,   default="zscore",
                   choices=["none", "center", "zscore"],
                   help="Per-wavelength input standardisation with training-fold "
                        "statistics (applied after the per-sample min-max scaling)")
    p.add_argument("--val_frac",        type=float, default=0.1,
                   help="Inner validation fraction of each training fold, used for "
                        "checkpoint and budget selection (0 = select on last epoch)")
    p.add_argument("--model_type",      type=str,   default="lensnir",
                   choices=["lensnir", "csse", "flat", "spectraformer", "resnet1d",
                            "cnn1d", "deepspectra", "mocnn", "assadzadeh"],
                   help="lensnir (default; legacy name csse): LensNIR; flat: flat SMART-NIR encoder with "
                        "the MT heads; spectraformer / resnet1d / mocnn: external "
                        "multi-output baselines (train with --st_task all); cnn1d / "
                        "deepspectra: single-task literature CNNs (--st_task "
                        "food|reg|aux, combined post-hoc by combine_st.py)")
    # Training
    p.add_argument("--epochs",          dest="epochs_override", type=int, default=None,
                   help=f"Epochs for every dataset (default per dataset: {DATASET_EPOCHS})")
    p.add_argument("--batch_size",      dest="batch_size_override", type=int, default=None,
                   help=f"Batch size for every dataset (default per dataset: {DATASET_BATCH})")
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--st_task",         type=str,   default="all",
                   choices=["all", "food", "reg", "aux"],
                   help="single-task mode: train only one head (all = multi-task); "
                        "aux = quality head, Rapeseed only")
    p.add_argument("--loss_weighting",  type=str,   default="uncertainty",
                   choices=["uncertainty", "fixed"])
    p.add_argument("--save_dir",        type=str,
                   default=os.getenv("SAVE_DIR", "results/main"),
                   help="Output root; each dataset writes to <save_dir>/<Dataset>/")
    p.add_argument("--device",          type=str,
                   default=os.getenv("DEVICE", "auto"),
                   help="auto | cpu | cuda | cuda:0 | cuda:1 ...")
    p.add_argument("--folds",           type=str,   default="1,2,3,4,5",
                   help="Comma-separated 1-based fold numbers to run in this "
                        "process, e.g. '1,2,3' on one GPU and '4,5' on another.")
    p.add_argument("--amp",             dest="amp", action="store_true", default=True,
                   help="Mixed-precision (autocast + GradScaler) training on CUDA. "
                        "On by default; ignored on CPU.")
    p.add_argument("--no_amp",          dest="amp", action="store_false",
                   help="Disable mixed precision, train in full FP32.")
    p.add_argument("--resume",          dest="resume", action="store_true", default=True,
                   help="Resume each fold from <save_dir>/fold{N}_resume.pt if present, "
                        "and skip folds whose fold{N}.pt already exists. On by default.")
    p.add_argument("--no_resume",       dest="resume", action="store_false",
                   help="Ignore any existing checkpoint and start the fold from epoch 0.")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    print(f"Device: {device}")
    run(args, device)
