"""Buoc 1 (presence/absence of each of the 19 pesticides), one file for the four baselines.

  METHOD=xgboost      XGBoost (was regression_stage1_engine.py)
  METHOD=lightgbm     LightGBM (was regression_stage1_lightgbm_engine.py)
  METHOD=smartnir     SMART-NIR + food one-hot (was regression_stage1_smartnir_engine.py)
  METHOD=guideddcnet  GuidedDCNet + food one-hot (was regression_stage1_guideddcnet_engine.py)

All follow the same protocol: 5 StratifiedKFold splits (random_state=42), food one-hot as an input,
Platt calibration and a Recall>=0.9 decision threshold, same metric names in history. Run e.g.
    MACHINE=FLAMENIR METHOD=lightgbm python3 stage1_detection.py
    MACHINE=FLAMENIR METHOD=smartnir SUBSTANCES=Thiamethoxam MAX_EPOCHS=50 POS_FRAC=0.25 \
        CAL_FRAC=0.15 TAG=stage1_smartnir_os2 python3 stage1_detection.py
Each method's own environment variables are documented in its main_<method> function.
"""


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score, brier_score_loss, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import os
import joblib
from dotenv import load_dotenv
from dataset.preprocessing import preprocess_spectra
import time
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from model.classification_model import KANClassifier, SMARTNIRClassifier, SmartNIRClassificationConfig
from model.guideddcnet_model import ConditionalUNet, GuidedDCNet, GuidedDCNetConfig


# ======================================================================
# xgboost  (from regression_stage1_engine.py)
# ======================================================================

load_dotenv()


def pick_xgb_hyperparams(num_pos: int):
    """Scale tree complexity to how much positive-class data a substance
    actually has, so substances with only a handful of detections (which
    XGBoost would otherwise overfit with deep/many trees) get a simpler
    model, while well-represented substances keep the fuller default.
    Returns (n_estimators, learning_rate, max_depth).
    """
    if num_pos < 50:
        return 100, 0.05, 3
    elif num_pos < 500:
        return 150, 0.08, 4
    else:
        return 200, 0.1, 6


def pick_threshold_for_recall(y_true, y_prob, target_recall=0.9):
    """Highest-precision decision threshold that still achieves at least
    `target_recall`. Swept on the caller's own data (train fold, not val),
    so the threshold choice doesn't leak information from the evaluation
    set -- same rationale as fitting calibration on train (Bước 1).
    Falls back to the max-recall threshold if the target is unreachable.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    precisions, recalls = precisions[:-1], recalls[:-1]  # drop the threshold=inf sentinel
    eligible = np.where(recalls >= target_recall)[0]
    if len(eligible) == 0:
        return float(thresholds[np.argmax(recalls)])
    best = eligible[np.argmax(precisions[eligible])]
    return float(thresholds[best])


def train_for_substance_xgboost(X, y, substance_name, n_splits=5, n_estimators=100, learning_rate=0.1, max_depth=6, patience=10,
                        target_recall=0.9,
                        save_history_dir="history", save_fig_dir="history", save_best_model_dir="checkpoint", save_scaler_dir="scalers"):

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_histories = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"Fold {fold + 1}/{n_splits} for {substance_name}")

        # Split data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Normalize: fit scaler on train
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Save scaler for inference
        scaler_path = os.path.join(save_scaler_dir, f"{substance_name}_fold_{fold + 1}_scaler.pkl")
        os.makedirs(save_scaler_dir, exist_ok=True)
        joblib.dump(scaler, scaler_path)

        # Prepare DMatrix
        dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
        dval = xgb.DMatrix(X_val_scaled, label=y_val)

        # Full (unbalanced) data now goes in directly, so tell XGBoost the
        # true train-fold class ratio instead of throwing away negatives.
        num_pos_train = int((y_train == 1).sum())
        num_neg_train = int((y_train == 0).sum())
        scale_pos_weight = num_neg_train / max(num_pos_train, 1)

        # Params
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'eta': learning_rate,
            'max_depth': max_depth,
            'tree_method': 'hist',
            'scale_pos_weight': scale_pos_weight,
        }

        # Train with early stopping
        evals = [(dtrain, 'train'), (dval, 'val')]
        model = xgb.train(params, dtrain, num_boost_round=n_estimators, evals=evals, early_stopping_rounds=patience, verbose_eval=False)

        # Calibrate: XGBoost trained with scale_pos_weight skews raw scores
        # away from true probabilities. Fit Platt scaling (1D logistic
        # regression) on the TRAIN fold's own predictions so evaluation on
        # val doesn't double-dip, then apply it to val.
        y_pred_prob_train = model.predict(dtrain)
        calibrator = LogisticRegression()
        calibrator.fit(y_pred_prob_train.reshape(-1, 1), y_train)

        # Decision threshold: swept on the calibrated TRAIN probabilities
        # (not val) for the same no-leakage reason as calibration itself --
        # pick the highest-precision cut that still hits `target_recall`,
        # since missing a real pesticide detection is costlier than a false
        # alarm. Applied as-is to val for evaluation.
        y_pred_prob_train_cal = calibrator.predict_proba(y_pred_prob_train.reshape(-1, 1))[:, 1]
        threshold = pick_threshold_for_recall(y_train, y_pred_prob_train_cal, target_recall=target_recall)

        y_pred_prob_raw = model.predict(dval)
        y_pred_prob = calibrator.predict_proba(y_pred_prob_raw.reshape(-1, 1))[:, 1]
        y_pred = (y_pred_prob > threshold).astype(int)

        calibrator_path = os.path.join(save_scaler_dir, f"{substance_name}_fold_{fold + 1}_calibrator.pkl")
        joblib.dump(calibrator, calibrator_path)
        threshold_path = os.path.join(save_scaler_dir, f"{substance_name}_fold_{fold + 1}_threshold.json")
        with open(threshold_path, "w") as f:
            json.dump({"threshold": threshold, "target_recall": target_recall}, f, indent=4)

        # Metrics
        val_acc = accuracy_score(y_val, y_pred)
        val_precision = precision_score(y_val, y_pred, average='macro', zero_division=0)
        val_recall = recall_score(y_val, y_pred, average='macro', zero_division=0)
        val_f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)
        val_pr_auc = average_precision_score(y_val, y_pred_prob)
        brier_before = brier_score_loss(y_val, y_pred_prob_raw)
        brier_after = brier_score_loss(y_val, y_pred_prob)

        # History (simplified, since no epochs)
        history = {
            "threshold": [threshold],
            "val_acc": [val_acc],
            "val_precision": [val_precision],
            "val_recall": [val_recall],
            "val_f1": [val_f1],
            "val_pr_auc": [val_pr_auc],
            "brier_before_calibration": [brier_before],
            "brier_after_calibration": [brier_after],
        }

        # Save history
        save_history_path = os.path.join(save_history_dir, f"{substance_name}_fold_{fold + 1}.json")
        os.makedirs(save_history_dir, exist_ok=True)
        with open(save_history_path, "w") as f:
            json.dump(history, f, indent=4)

        # Save model
        save_best_model_path = os.path.join(save_best_model_dir, f"{substance_name}_fold_{fold + 1}.json")
        os.makedirs(save_best_model_dir, exist_ok=True)
        model.save_model(save_best_model_path)

        # Plot (simple, no epochs)
        metrics = ['acc', 'precision', 'recall', 'f1', 'pr_auc']
        values = [val_acc, val_precision, val_recall, val_f1, val_pr_auc]
        plt.figure(figsize=(8, 6))
        plt.bar(metrics, values)
        plt.title(f"Validation Metrics for {substance_name} Fold {fold + 1}")
        plt.ylim(0, 1)
        save_fig_path = os.path.join(save_fig_dir, f"{substance_name}_fold_{fold + 1}.png")
        os.makedirs(save_fig_dir, exist_ok=True)
        plt.savefig(save_fig_path)
        plt.close()

        fold_histories.append(history)

    avg_val_acc = np.mean([h["val_acc"][0] for h in fold_histories])
    avg_pr_auc = np.mean([h["val_pr_auc"][0] for h in fold_histories])
    print(f"Average validation accuracy across folds for {substance_name}: {avg_val_acc:.4f} "
          f"(PR-AUC: {avg_pr_auc:.4f})")


# ======================================================================
# lightgbm  (from regression_stage1_lightgbm_engine.py)
# ======================================================================

ALL_SUBSTANCES = [
    'Thiamethoxam', 'Permethrin', 'Metalaxyl', 'Azoxystrobin',
    'Imidaclopird', 'Difenoconazole', 'Cypermethrin', 'Cyhalothrin',
    'Chlorantraniliprol', 'Chlopyrifos Methyl', 'Emamectin benzoate',
    'Chlorothalonil', 'Triadimefon', 'Cyantraniliprole', 'Flutolanil',
    'Indoxacarb', 'Abamectin', 'Propamocarb.HCL', 'Chlothianidin',
]


def run_substance_lightgbm(substance, X, y, machine, folds, n_jobs, patience=20):
    tag = "stage1_lightgbm"
    hist_dir = f"history/substance_regression/{tag}/{machine}/{substance}"
    ckpt_dir = f"checkpoint/substance_regression/{tag}/{machine}/{substance}"
    data_dir = f"data/substance_regression/{tag}/{machine}/{substance}"
    for d in (hist_dir, ckpt_dir, data_dir):
        os.makedirs(d, exist_ok=True)

    num_pos = int(y.sum())
    n_estimators, learning_rate, max_depth = pick_xgb_hyperparams(num_pos)
    print(f"{substance}: pos={num_pos}, neg={len(y) - num_pos} -> n_estimators={n_estimators}, "
          f"lr={learning_rate}, max_depth={max_depth}", flush=True)

    for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y), start=1):
        if folds and fold not in folds:
            continue
        print(f"Fold {fold}/5 for {substance}", flush=True)
        t0 = time.time()
        X_tr, X_va, y_tr, y_va = X[tr], X[va], y[tr], y[va]

        params = dict(
            objective="binary", learning_rate=learning_rate, max_depth=max_depth,
            num_leaves=min(2 ** max_depth - 1, 63), n_estimators=n_estimators,
            scale_pos_weight=(y_tr == 0).sum() / max((y_tr == 1).sum(), 1),
            n_jobs=n_jobs, random_state=42, verbose=-1, force_col_wise=True,
        )
        model = lgb.LGBMClassifier(**params)
        # Early stopping on validation average precision, not logloss: the model is
        # trained with scale_pos_weight, so unweighted validation logloss is lowest
        # at iteration 1-2 (LightGBM starts from the calibrated prior) and stopping
        # on it froze rare-substance models at a single tree.
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="average_precision",
                  callbacks=[lgb.early_stopping(patience, first_metric_only=True, verbose=False)])

        raw_tr = model.predict_proba(X_tr)[:, 1]
        calib = LogisticRegression().fit(raw_tr.reshape(-1, 1), y_tr)
        thr = pick_threshold_for_recall(y_tr, calib.predict_proba(raw_tr.reshape(-1, 1))[:, 1], 0.9)

        p_raw = model.predict_proba(X_va)[:, 1]
        p_va = calib.predict_proba(p_raw.reshape(-1, 1))[:, 1]
        pred = (p_va > thr).astype(int)
        accs = [accuracy_score(y_va, p_va > t) for t in np.linspace(0.05, 0.95, 91)]
        hist = {
            "threshold": [thr],
            "val_acc": [accuracy_score(y_va, pred)],
            "val_precision": [precision_score(y_va, pred, average="macro", zero_division=0)],
            "val_recall": [recall_score(y_va, pred, average="macro", zero_division=0)],
            "val_f1": [f1_score(y_va, pred, average="macro", zero_division=0)],
            "val_pr_auc": [average_precision_score(y_va, p_va)],
            "brier_before_calibration": [brier_score_loss(y_va, p_raw)],
            "brier_after_calibration": [brier_score_loss(y_va, p_va)],
            "val_pos_recall": [recall_score(y_va, pred, zero_division=0)],
            "val_pos_precision": [precision_score(y_va, pred, zero_division=0)],
            "val_acc_best_threshold": [float(max(accs))],
            "best_iteration": [int(model.best_iteration_ or n_estimators)],
            "seconds": [time.time() - t0],
        }
        with open(f"{hist_dir}/{substance}_fold_{fold}.json", "w") as f:
            json.dump(hist, f, indent=4)
        model.booster_.save_model(f"{ckpt_dir}/{substance}_fold_{fold}.txt")
        joblib.dump(calib, f"{data_dir}/{substance}_fold_{fold}_calibrator.pkl")
        with open(f"{data_dir}/{substance}_fold_{fold}_threshold.json", "w") as f:
            json.dump({"threshold": thr, "target_recall": 0.9}, f, indent=4)
        print(f"  -> acc={hist['val_acc'][0]:.4f} acc@bestthr={hist['val_acc_best_threshold'][0]:.4f} "
              f"PR-AUC={hist['val_pr_auc'][0]:.4f} pos_recall={hist['val_pos_recall'][0]:.3f} "
              f"pos_prec={hist['val_pos_precision'][0]:.3f} ({hist['seconds'][0]:.0f}s)", flush=True)


# ======================================================================
# smartnir  (from regression_stage1_smartnir_engine.py)
# ======================================================================

TRIAL_SUBSTANCES = ['Thiamethoxam', 'Azoxystrobin', 'Propamocarb.HCL']


class SmartNIRWithFood(nn.Module):
    """SMART-NIR backbone; the food one-hot is concatenated to the CLS
    embedding before the KAN head (the stock model only sees the spectrum)."""

    def __init__(self, signal_len: int, n_food: int, d_model=128, depth=3, n_heads=4):
        super().__init__()
        cfg = SmartNIRClassificationConfig(
            signal_len=signal_len, out_ch_per_branch=64, d_model=d_model,
            depth=depth, n_heads=n_heads, classifier="kan", num_classes=2,
        )
        base = SMARTNIRClassifier(cfg)
        self.mk, self.proj, self.encoder = base.mk, base.proj, base.encoder
        self.head = KANClassifier(d_model + n_food, 2, n_basis=cfg.kan_basis)

    def forward(self, spec, food):
        z = self.encoder(self.proj(self.mk(spec.unsqueeze(1))))
        return self.head(torch.cat([z[:, 0, :], food.to(z.dtype)], dim=1))


@torch.no_grad()
def predict_logit_diff(model, Xs, Fs, idx, bs):
    """Returns logit(class1) - logit(class0) for rows `idx`, float32 on CPU."""
    model.eval()
    out = []
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(Xs[b], Fs[b])
        lg = lg.float()
        out.append((lg[:, 1] - lg[:, 0]).cpu())
    return torch.cat(out).numpy()


def train_one_fold_smartnir(Xs, Fs, ys, tr_idx, va_idx, device, max_epochs, bs, patience=8, min_epochs=10, lr=2e-3, pos_frac=0.0):
    model = SmartNIRWithFood(Xs.shape[1], Fs.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps_per_epoch = (len(tr_idx) + bs - 1) // bs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=max_epochs * steps_per_epoch, pct_start=0.1)
    n_pos = int(ys[tr_idx].sum().item())
    # Softened class weight (sqrt of the neg/pos ratio): the full ratio (~35x for
    # the rarest substances) destabilised training; Platt scaling below repairs
    # the remaining probability skew anyway.
    class_w = torch.tensor([1.0, ((len(tr_idx) - n_pos) / max(n_pos, 1)) ** 0.5], device=device)
    tr_idx_t = torch.as_tensor(tr_idx, device=device)
    if pos_frac > 0:
        # Fixed positive share in every batch (positives drawn with replacement),
        # so the rare class gives gradient signal at every step instead of the
        # model idling near the all-negative solution. Plain CE then, since the
        # sampling already re-balances the classes; Platt scaling fixes calibration.
        pos_pool = tr_idx_t[ys[tr_idx_t] == 1]
        neg_pool = tr_idx_t[ys[tr_idx_t] == 0]
        n_pos_b = max(1, int(bs * pos_frac))
        class_w = None
    va_idx_t = torch.as_tensor(va_idx, device=device)

    best, best_state, bad = -1.0, None, 0
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        model.train()
        perm = tr_idx_t[torch.randperm(len(tr_idx_t), device=device)]
        run = 0.0
        for i in range(0, len(perm), bs):
            if pos_frac > 0:
                b = torch.cat([
                    pos_pool[torch.randint(len(pos_pool), (n_pos_b,), device=device)],
                    neg_pool[torch.randint(len(neg_pool), (bs - n_pos_b,), device=device)]])
            else:
                b = perm[i:i + bs]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(Xs[b], Fs[b])
            loss = F.cross_entropy(lg.float(), ys[b], weight=class_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += loss.item() * len(b)
        d = predict_logit_diff(model, Xs, Fs, va_idx_t, bs * 2)
        p = 1.0 / (1.0 + np.exp(-d))
        yv = ys[va_idx_t].cpu().numpy()
        val_loss = float(-np.mean(yv * np.log(p + 1e-7) + (1 - yv) * np.log(1 - p + 1e-7)))
        print(f"  epoch {epoch:2d} train_loss={run / len(perm):.4f} val_logloss={val_loss:.4f} "
              f"val_PR-AUC={average_precision_score(yv, p):.4f} ({time.time() - t0:.0f}s)", flush=True)
        # Early stopping on val PR-AUC (the headline metric), and not before
        # `min_epochs` so the OneCycle warm-up/peak-LR phase can't trigger it.
        val_ap = average_precision_score(yv, p)
        if val_ap > best + 1e-4:
            best, bad = val_ap, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch > min_epochs:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


def run_substance_smartnir(substance, Xnp, Fnp, y, machine, device, max_epochs, bs, pos_frac=0.0, folds=None, tag="stage1_smartnir",
                  cal_frac=0.0):
    hist_dir = f"history/substance_regression/{tag}/{machine}/{substance}"
    ckpt_dir = f"checkpoint/substance_regression/{tag}/{machine}/{substance}"
    data_dir = f"data/substance_regression/{tag}/{machine}/{substance}"
    for d in (hist_dir, ckpt_dir, data_dir):
        os.makedirs(d, exist_ok=True)

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    Fs = torch.tensor(Fnp, device=device)
    ys = torch.tensor(y, dtype=torch.long, device=device)
    for fold, (tr, va) in enumerate(kf.split(Xnp, y), start=1):
        if folds and fold not in folds:
            continue
        print(f"{substance} fold {fold}/5", flush=True)
        t0 = time.time()
        mean = Xnp[tr].mean(axis=0, keepdims=True)
        std = Xnp[tr].std(axis=0, keepdims=True) + 1e-8
        Xs = torch.tensor((Xnp - mean) / std, dtype=torch.float32, device=device)
        np.savez(f"{data_dir}/fold_{fold}_norm.npz", mean=mean, std=std)

        # Calibration/threshold split: a stratified slice of the train fold that
        # never updates the weights. The network can memorise its own training
        # rows (train loss -> ~0), which made a Recall>=0.9 threshold picked on
        # the fitted rows far too strict on unseen rows.
        if cal_frac > 0:
            tr_fit, tr_cal = train_test_split(tr, test_size=cal_frac, stratify=y[tr], random_state=fold)
        else:
            tr_fit, tr_cal = tr, tr
        model = train_one_fold_smartnir(Xs, Fs, ys, tr_fit, va, device, max_epochs, bs, pos_frac=pos_frac)

        # calibration + threshold on the held-back train slice (CAL_FRAC>0) or the
        # whole train fold (CAL_FRAC=0, same rule as XGBoost)
        d_tr = predict_logit_diff(model, Xs, Fs, torch.as_tensor(tr_cal, device=device), bs * 2)
        calib = LogisticRegression().fit(d_tr.reshape(-1, 1), y[tr_cal])
        p_tr = calib.predict_proba(d_tr.reshape(-1, 1))[:, 1]
        thr = pick_threshold_for_recall(y[tr_cal], p_tr, target_recall=0.9)

        d_va = predict_logit_diff(model, Xs, Fs, torch.as_tensor(va, device=device), bs * 2)
        p_raw = 1.0 / (1.0 + np.exp(-d_va))
        p_va = calib.predict_proba(d_va.reshape(-1, 1))[:, 1]
        yv = y[va]
        pred = (p_va > thr).astype(int)
        grid = np.linspace(0.05, 0.95, 91)
        accs = [accuracy_score(yv, p_va > t) for t in grid]
        hist = {
            "threshold": [thr],
            "val_acc": [accuracy_score(yv, pred)],
            "val_precision": [precision_score(yv, pred, average="macro", zero_division=0)],
            "val_recall": [recall_score(yv, pred, average="macro", zero_division=0)],
            "val_f1": [f1_score(yv, pred, average="macro", zero_division=0)],
            "val_pr_auc": [average_precision_score(yv, p_va)],
            "brier_before_calibration": [brier_score_loss(yv, p_raw)],
            "brier_after_calibration": [brier_score_loss(yv, p_va)],
            # extras beyond the XGBoost history, for the accuracy-vs-recall discussion
            "val_pos_recall": [recall_score(yv, pred, zero_division=0)],
            "val_pos_precision": [precision_score(yv, pred, zero_division=0)],
            "val_acc_best_threshold": [float(max(accs))],
            "seconds": [time.time() - t0],
        }
        with open(f"{hist_dir}/{substance}_fold_{fold}.json", "w") as f:
            json.dump(hist, f, indent=4)
        torch.save(model.state_dict(), f"{ckpt_dir}/{substance}_fold_{fold}.pth")
        joblib.dump(calib, f"{data_dir}/{substance}_fold_{fold}_calibrator.pkl")
        with open(f"{data_dir}/{substance}_fold_{fold}_threshold.json", "w") as f:
            json.dump({"threshold": thr, "target_recall": 0.9}, f, indent=4)
        print(f"  -> acc={hist['val_acc'][0]:.4f} acc@bestthr={hist['val_acc_best_threshold'][0]:.4f} "
              f"PR-AUC={hist['val_pr_auc'][0]:.4f} pos_recall={hist['val_pos_recall'][0]:.3f} "
              f"pos_prec={hist['val_pos_precision'][0]:.3f} ({hist['seconds'][0]:.0f}s)", flush=True)
        del Xs
        torch.cuda.empty_cache()


# ======================================================================
# guideddcnet  (from regression_stage1_guideddcnet_engine.py)
# ======================================================================

class GuidedDCNetFood(GuidedDCNet):
    """GuidedDCNet whose conditioning embedding pi(x) also carries the food
    one-hot. Set `self._food` (B, n_food) before every forward call."""

    def __init__(self, n_food: int):
        cfg = GuidedDCNetConfig(num_classes=2)
        super().__init__(cfg)
        self.unet = ConditionalUNet(cfg.num_classes, embed_dim=cfg.stage_channels[-1] + n_food,
                                    d_hidden=cfg.d_hidden, K=cfg.K)
        self._food = None

    def encode(self, x):
        pi_x, y_g_hat, y_l_hat = super().encode(x)
        return torch.cat([pi_x, self._food], dim=1), y_g_hat, y_l_hat


def draw_batch(pos_pool, neg_pool, perm, i, bs, pos_frac):
    if pos_frac > 0:
        n_pos_b = max(1, int(bs * pos_frac))
        dev = pos_pool.device
        return torch.cat([pos_pool[torch.randint(len(pos_pool), (n_pos_b,), device=dev)],
                          neg_pool[torch.randint(len(neg_pool), (bs - n_pos_b,), device=dev)]])
    return perm[i:i + bs]


@torch.no_grad()
def score_rows(model, Xs, Fs, idx, bs):
    """Diffusion score y0_hat[:,1]-y0_hat[:,0] via full reverse sampling."""
    model.eval()
    out = []
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        model._food = Fs[b]
        y0 = model.reverse_sample(Xs[b])
        out.append((y0[:, 1] - y0[:, 0]).cpu())
    return torch.cat(out).numpy()


def train_one_fold_guideddcnet(Xs, Fs, ys, tr_idx, va_idx, device, pre_epochs, max_epochs, bs, eval_every, pos_frac, lam=0.5,
                   lr_scale=1.0, wd=0.0, use_last=False):
    model = GuidedDCNetFood(Fs.shape[1]).to(device)
    data_head = nn.Linear(model.data_encoder.embed_dim + Fs.shape[1], 2).to(device)
    tr_idx_t = torch.as_tensor(tr_idx, device=device)
    va_idx_t = torch.as_tensor(va_idx, device=device)
    pos_pool = tr_idx_t[ys[tr_idx_t] == 1]
    neg_pool = tr_idx_t[ys[tr_idx_t] == 0]
    n_pos = len(pos_pool)
    class_w = None if pos_frac > 0 else torch.tensor(
        [1.0, ((len(tr_idx) - n_pos) / max(n_pos, 1)) ** 0.5], device=device)
    yv = ys[va_idx_t].cpu().numpy()

    # ---- Stage 1: CE pretraining of MCGM + data encoder ----
    params = (list(model.mcgm.parameters()) + list(model.data_encoder.parameters())
              + list(data_head.parameters()))
    opt = torch.optim.AdamW(params, lr=5e-4 * lr_scale, weight_decay=wd)
    for epoch in range(1, pre_epochs + 1):
        t0 = time.time()
        model.train(); data_head.train()
        perm = tr_idx_t[torch.randperm(len(tr_idx_t), device=device)]
        run = 0.0
        for i in range(0, len(perm), bs):
            b = draw_batch(pos_pool, neg_pool, perm, i, bs, pos_frac)
            model._food = Fs[b]
            loss = F.cross_entropy(model.pretrain_logits(Xs[b], data_head), ys[b], weight=class_w)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            run += loss.item() * len(b)
        print(f"  [pretrain] epoch {epoch:2d} train_loss={run / len(perm):.4f} ({time.time() - t0:.0f}s)", flush=True)

    # ---- Stage 2: diffusion training ----
    opt = torch.optim.AdamW([
        {"params": list(model.mcgm.parameters()) + list(model.data_encoder.parameters()), "lr": 5e-4 * lr_scale},
        {"params": model.unet.parameters(), "lr": 1e-3 * lr_scale}], weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    best, best_state = -1.0, None
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        model.train()
        perm = tr_idx_t[torch.randperm(len(tr_idx_t), device=device)]
        run = 0.0
        for i in range(0, len(perm), bs):
            b = draw_batch(pos_pool, neg_pool, perm, i, bs, pos_frac)
            model._food = Fs[b]
            y0 = F.one_hot(ys[b], 2).float()
            l_eps, l_g, l_l = model.training_losses(Xs[b], y0)
            loss = l_eps + lam * (l_g + l_l)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            run += loss.item() * len(b)
        sched.step()
        msg = f"  epoch {epoch:2d} train_loss={run / len(perm):.4f}"
        if epoch % eval_every == 0 or epoch == max_epochs:
            ap = average_precision_score(yv, score_rows(model, Xs, Fs, va_idx_t, bs * 2))
            msg += f" val_PR-AUC={ap:.4f}"
            if ap > best:
                best = ap
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(msg + f" ({time.time() - t0:.0f}s)", flush=True)
    if not use_last:
        model.load_state_dict(best_state)
    return model


def run_substance_guideddcnet(substance, Xnp, Fnp, y, machine, device, pre_epochs, max_epochs, bs, eval_every,
                  pos_frac, cal_frac, folds, tag, lr_scale=1.0, wd=0.0, use_last=False):
    hist_dir = f"history/substance_regression/{tag}/{machine}/{substance}"
    ckpt_dir = f"checkpoint/substance_regression/{tag}/{machine}/{substance}"
    data_dir = f"data/substance_regression/{tag}/{machine}/{substance}"
    for d in (hist_dir, ckpt_dir, data_dir):
        os.makedirs(d, exist_ok=True)
    Fs = torch.tensor(Fnp, device=device)
    ys = torch.tensor(y, dtype=torch.long, device=device)

    for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(Xnp, y), start=1):
        if folds and fold not in folds:
            continue
        print(f"{substance} fold {fold}/5", flush=True)
        t0 = time.time()
        mean = Xnp[tr].mean(axis=0, keepdims=True)
        std = Xnp[tr].std(axis=0, keepdims=True) + 1e-8
        Xs = torch.tensor((Xnp - mean) / std, dtype=torch.float32, device=device)
        np.savez(f"{data_dir}/fold_{fold}_norm.npz", mean=mean, std=std)

        if cal_frac > 0:
            tr_fit, tr_cal = train_test_split(tr, test_size=cal_frac, stratify=y[tr], random_state=fold)
        else:
            tr_fit, tr_cal = tr, tr
        model = train_one_fold_guideddcnet(Xs, Fs, ys, tr_fit, va, device, pre_epochs, max_epochs, bs, eval_every, pos_frac,
                               lr_scale=lr_scale, wd=wd, use_last=use_last)

        d_cal = score_rows(model, Xs, Fs, torch.as_tensor(tr_cal, device=device), bs * 2)
        calib = LogisticRegression().fit(d_cal.reshape(-1, 1), y[tr_cal])
        thr = pick_threshold_for_recall(y[tr_cal], calib.predict_proba(d_cal.reshape(-1, 1))[:, 1], 0.9)

        d_va = score_rows(model, Xs, Fs, torch.as_tensor(va, device=device), bs * 2)
        p_raw = 1.0 / (1.0 + np.exp(-d_va))
        p_va = calib.predict_proba(d_va.reshape(-1, 1))[:, 1]
        yv = y[va]
        pred = (p_va > thr).astype(int)
        accs = [accuracy_score(yv, p_va > t) for t in np.linspace(0.05, 0.95, 91)]
        hist = {
            "threshold": [thr],
            "val_acc": [accuracy_score(yv, pred)],
            "val_precision": [precision_score(yv, pred, average="macro", zero_division=0)],
            "val_recall": [recall_score(yv, pred, average="macro", zero_division=0)],
            "val_f1": [f1_score(yv, pred, average="macro", zero_division=0)],
            "val_pr_auc": [average_precision_score(yv, p_va)],
            "brier_before_calibration": [brier_score_loss(yv, p_raw)],
            "brier_after_calibration": [brier_score_loss(yv, p_va)],
            "val_pos_recall": [recall_score(yv, pred, zero_division=0)],
            "val_pos_precision": [precision_score(yv, pred, zero_division=0)],
            "val_acc_best_threshold": [float(max(accs))],
            "seconds": [time.time() - t0],
        }
        with open(f"{hist_dir}/{substance}_fold_{fold}.json", "w") as f:
            json.dump(hist, f, indent=4)
        torch.save(model.state_dict(), f"{ckpt_dir}/{substance}_fold_{fold}.pth")
        joblib.dump(calib, f"{data_dir}/{substance}_fold_{fold}_calibrator.pkl")
        with open(f"{data_dir}/{substance}_fold_{fold}_threshold.json", "w") as f:
            json.dump({"threshold": thr, "target_recall": 0.9}, f, indent=4)
        print(f"  -> acc={hist['val_acc'][0]:.4f} acc@bestthr={hist['val_acc_best_threshold'][0]:.4f} "
              f"PR-AUC={hist['val_pr_auc'][0]:.4f} pos_recall={hist['val_pos_recall'][0]:.3f} "
              f"pos_prec={hist['val_pos_precision'][0]:.3f} ({hist['seconds'][0]:.0f}s)", flush=True)
        del Xs
        torch.cuda.empty_cache()


# ======================================================================
# entry point: METHOD=xgboost
# ======================================================================
def main_xgboost():
    machine = f"{os.environ['MACHINE']}"
    task = "substance_regression"

    full_path = f"{os.environ['DATASET_ROOT']}/{machine}/ALL.csv"
    df = pd.read_csv(full_path)

    # Assume columns: w_1, w_2, ..., w_n for wavelengths, and substance columns
    wavelength_cols = [col for col in df.columns if col.startswith('w_')]  # Adjust if needed

    X_raw = df[wavelength_cols].values.astype(np.float32)
    X_raw, keep_mask = preprocess_spectra(X_raw)
    n_dropped = (~keep_mask).sum()
    if n_dropped:
        print(f"Dropped {n_dropped} outlier spectra out of {len(keep_mask)}")
    df = df[keep_mask].reset_index(drop=True)
    X = pd.DataFrame(X_raw, columns=wavelength_cols)

    # Food category as an explicit input feature (one-hot over FOOD_ID),
    # not just something the model has to infer implicitly from the
    # spectrum -- the detection threshold ultimately traced back to a
    # (food, pesticide) MRL table (../all-dataset/Danang-NIR/thresholds.json),
    # so giving the model the food identity directly is strictly easier than
    # making it re-derive food type from spectral shape on its own. Verified
    # empirically that concatenating this before the shared StandardScaler
    # (rather than scaling it separately) doesn't hurt XGBoost: it only
    # makes monotonic per-column affine changes, which tree splits are
    # invariant to.
    with open(f"{os.environ['DATASET_ROOT']}/food_ids.json") as f:
        food_ids = json.load(f)
    food_name_to_id = {v["name"]: k for k, v in food_ids.items()}
    unmapped_cats = set(df["category"].unique()) - set(food_name_to_id)
    if unmapped_cats:
        raise ValueError(f"category values with no FOOD_ID mapping: {unmapped_cats}")
    food_id_per_row = df["category"].map(food_name_to_id)
    cat_onehot = pd.get_dummies(food_id_per_row, prefix="cat").reindex(
        columns=[f"cat_{fid}" for fid in food_ids], fill_value=0
    ).astype(np.float32)
    X = pd.concat([X, cat_onehot.reset_index(drop=True)], axis=1)

    substances = [
        'Thiamethoxam', 'Permethrin', 'Metalaxyl', 'Azoxystrobin',
        'Imidaclopird', 'Difenoconazole', 'Cypermethrin', 'Cyhalothrin',
        'Chlorantraniliprol', 'Chlopyrifos Methyl', 'Emamectin benzoate',
        'Chlorothalonil', 'Triadimefon', 'Cyantraniliprole', 'Flutolanil',
        'Indoxacarb', 'Abamectin', 'Propamocarb.HCL', 'Chlothianidin'
    ]

    for substance in substances:
        print(f"Training for {substance}")
        y_raw = df[substance]
        # Convert to binary: 0 if -1 (no), 1 if >0 (yes)
        y = (y_raw > 0).astype(int)

        save_history_dir = f"history/{task}/stage1/{machine}/{substance}"
        save_fig_dir = f"history/{task}/stage1/{machine}/{substance}"
        save_best_model_dir = f"checkpoint/{task}/stage1/{machine}/{substance}"
        save_scaler_dir = f"data/{task}/stage1/{machine}/{substance}"

        if np.all(y_raw == -1):
            print(f"No positive samples for {substance}, creating folders and skipping training.")
            os.makedirs(save_history_dir, exist_ok=True)
            os.makedirs(save_fig_dir, exist_ok=True)
            os.makedirs(save_best_model_dir, exist_ok=True)
            os.makedirs(save_scaler_dir, exist_ok=True)
            continue

        n_splits = 5
        num_pos = int((y == 1).sum())
        num_neg = int((y == 0).sum())
        if num_pos < n_splits or num_neg < n_splits:
            print(f"Not enough samples for {substance} (pos={num_pos}, neg={num_neg}), "
                  f"need >= {n_splits} of each for {n_splits}-fold CV. Skipping.")
            os.makedirs(save_history_dir, exist_ok=True)
            os.makedirs(save_fig_dir, exist_ok=True)
            os.makedirs(save_best_model_dir, exist_ok=True)
            os.makedirs(save_scaler_dir, exist_ok=True)
            continue

        n_estimators, learning_rate, max_depth = pick_xgb_hyperparams(num_pos)
        print(f"{substance}: pos={num_pos}, neg={num_neg} -> "
              f"n_estimators={n_estimators}, lr={learning_rate}, max_depth={max_depth}")

        train_for_substance_xgboost(X, y, substance, n_splits=n_splits, n_estimators=n_estimators,
                            learning_rate=learning_rate, max_depth=max_depth, patience=10,
                            save_history_dir=save_history_dir, save_fig_dir=save_fig_dir,
                            save_best_model_dir=save_best_model_dir, save_scaler_dir=save_scaler_dir)


# ======================================================================
# entry point: METHOD=lightgbm
# ======================================================================
def main_lightgbm():
    """Buoc 1 (presence/absence of each pesticide) with LightGBM.
    
    Mirrors stage1_detection.py (XGBoost) one to one so the two tree
    baselines are directly comparable: same 5 StratifiedKFold splits
    (random_state=42), food one-hot appended to the spectrum, same tree-complexity
    schedule by number of positives, scale_pos_weight from the train-fold class
    ratio, early stopping on validation average precision, Platt calibration and the
    Recall>=0.9 decision threshold fitted on the train fold, same metric names in
    history (plus positive-class recall/precision and the best-threshold accuracy).
    
    Env: MACHINE (default FLAMENIR), SUBSTANCES (comma list; default all 19),
    FOLDS, N_JOBS.
    """
    machine = os.environ.get("MACHINE", "FLAMENIR")
    subs = os.environ.get("SUBSTANCES")
    substances = [s.strip() for s in subs.split(",")] if subs else ALL_SUBSTANCES
    folds = [int(f) for f in os.environ["FOLDS"].split(",")] if os.environ.get("FOLDS") else None
    n_jobs = int(os.environ.get("N_JOBS", 32))

    root = os.environ["DATASET_ROOT"]
    df = pd.read_csv(f"{root}/{machine}/ALL.csv")
    w_cols = [c for c in df.columns if c.startswith("w_")]
    X, keep = preprocess_spectra(df[w_cols].values.astype(np.float32))
    df = df[keep].reset_index(drop=True)
    with open(f"{root}/food_ids.json") as f:
        food_ids = json.load(f)
    name2id = {v["name"]: k for k, v in food_ids.items()}
    onehot = pd.get_dummies(df["category"].map(name2id)).reindex(
        columns=list(food_ids), fill_value=0).astype(np.float32).values
    X = np.hstack([X, onehot])
    print(f"{machine}: {len(df)} rows, {X.shape[1]} features, substances={substances}", flush=True)
    for s in substances:
        run_substance_lightgbm(s, X, (df[s] > 0).astype(int).values, machine, folds, n_jobs)


# ======================================================================
# entry point: METHOD=smartnir
# ======================================================================
def main_smartnir():
    """Buoc 1 (presence/absence of each pesticide) with the SMART-NIR backbone.
    
    Same protocol as stage1_detection.py (XGBoost) so the two are directly
    comparable: the same 5 StratifiedKFold splits (random_state=42), food one-hot
    as an extra input, a calibrator (Platt scaling) and a decision threshold
    (Recall >= 0.9 on the train fold) fitted on the TRAIN fold, and the same
    metric names written to history.
    
    Speed: the whole spectra matrix lives on the GPU, batches are index slices
    (no DataLoader), bf16 autocast, large batch.
    
    Env: MACHINE (default FLAMENIR), SUBSTANCES (comma list; default = a small
    trial subset), MAX_EPOCHS, BATCH_SIZE, POS_FRAC (fixed positive share per
    batch; 0 = off), FOLDS (comma list of folds to run), TAG (output subfolder), BIN (wavelength binning factor), CAL_FRAC (share of the train fold held back
    for calibration/threshold; 0 = use the whole train fold).
    """
    machine = os.environ.get("MACHINE", "FLAMENIR")
    max_epochs = int(os.environ.get("MAX_EPOCHS", 30))
    bs = int(os.environ.get("BATCH_SIZE", 2048))
    subs = os.environ.get("SUBSTANCES")
    substances = [s.strip() for s in subs.split(",")] if subs else TRIAL_SUBSTANCES
    device = "cuda"
    pos_frac = float(os.environ.get("POS_FRAC", 0))
    folds = [int(f) for f in os.environ["FOLDS"].split(",")] if os.environ.get("FOLDS") else None
    tag = os.environ.get("TAG", "stage1_smartnir")
    cal_frac = float(os.environ.get("CAL_FRAC", 0))

    root = os.environ["DATASET_ROOT"]
    df = pd.read_csv(f"{root}/{machine}/ALL.csv")
    w_cols = [c for c in df.columns if c.startswith("w_")]
    X, keep = preprocess_spectra(df[w_cols].values.astype(np.float32))
    df = df[keep].reset_index(drop=True)
    # Optional wavelength binning (average pooling) for the high-resolution
    # machine: attention over the 2136-point OCEANFX spectrum costs ~110 s/epoch,
    # so BIN=8 averages groups of 8 neighbouring points (and trims to a length
    # the multi-kernel stem accepts, a multiple of 8) -> 264 points, ~13 s/epoch.
    bin_factor = int(os.environ.get("BIN", 1))
    if bin_factor > 1:
        n_out = (X.shape[1] // bin_factor) // 8 * 8
        X = X[:, :n_out * bin_factor].reshape(len(X), n_out, bin_factor).mean(axis=2).astype(np.float32)
        print(f"binned spectra by {bin_factor}: {len(w_cols)} -> {X.shape[1]} points", flush=True)
    with open(f"{root}/food_ids.json") as f:
        food_ids = json.load(f)
    name2id = {v["name"]: k for k, v in food_ids.items()}
    Fnp = pd.get_dummies(df["category"].map(name2id)).reindex(
        columns=list(food_ids), fill_value=0).astype(np.float32).values
    print(f"{machine}: {len(df)} rows, {X.shape[1]} wavelengths, substances={substances}", flush=True)

    for s in substances:
        run_substance_smartnir(s, X, Fnp, (df[s] > 0).astype(int).values, machine, device, max_epochs, bs,
                      pos_frac=pos_frac, folds=folds, tag=tag, cal_frac=cal_frac)


# ======================================================================
# entry point: METHOD=guideddcnet
# ======================================================================
def main_guideddcnet():
    """Buoc 1 (presence/absence of each pesticide) with the GuidedDCNet baseline.
    
    Same protocol as stage1_detection.py so the deep baselines are
    comparable: same 5 StratifiedKFold splits (random_state=42), food one-hot as an
    extra input, fixed positive share per batch (POS_FRAC), a slice of the train
    fold held back for Platt calibration + the Recall>=0.9 threshold (CAL_FRAC),
    best epoch chosen by validation PR-AUC, identical metric names in history.
    
    GuidedDCNet is a 2-stage model: (1) cross-entropy pretraining of the MCGM and
    the data encoder, (2) conditional-diffusion training. The prediction score is
    y0_hat[:,1] - y0_hat[:,0] from the full T-step reverse-sampling loop.
    
    The stock model only sees the spectrum, so the food one-hot is concatenated to
    the data-encoder embedding pi(x) that conditions the UNet.
    
    Env: MACHINE, SUBSTANCES, PRETRAIN_EPOCHS, MAX_EPOCHS, EVAL_EVERY, BATCH_SIZE,
    POS_FRAC, CAL_FRAC, FOLDS, TAG, LR_SCALE (multiplies both stage learning rates),
    WD (AdamW weight decay), BIN (wavelength binning factor; use 8 for OCEANFX,
    same rationale as main_smartnir), USE_LAST (1 = keep the final-epoch weights
    instead of the best-by-PR-AUC checkpoint; useful when EVAL_EVERY is large
    since few epochs get evaluated).
    """
    machine = os.environ.get("MACHINE", "FLAMENIR")
    subs = os.environ.get("SUBSTANCES")
    substances = [s.strip() for s in subs.split(",")] if subs else TRIAL_SUBSTANCES
    cfg = dict(
        pre_epochs=int(os.environ.get("PRETRAIN_EPOCHS", 5)),
        max_epochs=int(os.environ.get("MAX_EPOCHS", 20)),
        bs=int(os.environ.get("BATCH_SIZE", 1024)),
        eval_every=int(os.environ.get("EVAL_EVERY", 2)),
        pos_frac=float(os.environ.get("POS_FRAC", 0.25)),
        cal_frac=float(os.environ.get("CAL_FRAC", 0.15)),
        folds=[int(f) for f in os.environ["FOLDS"].split(",")] if os.environ.get("FOLDS") else None,
        tag=os.environ.get("TAG", "stage1_guideddcnet"),
        lr_scale=float(os.environ.get("LR_SCALE", 1.0)),
        wd=float(os.environ.get("WD", 0.0)),
        use_last=bool(int(os.environ.get("USE_LAST", 0))),
    )
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    root = os.environ["DATASET_ROOT"]
    df = pd.read_csv(f"{root}/{machine}/ALL.csv")
    w_cols = [c for c in df.columns if c.startswith("w_")]
    X, keep = preprocess_spectra(df[w_cols].values.astype(np.float32))
    df = df[keep].reset_index(drop=True)
    bin_factor = int(os.environ.get("BIN", 1))
    if bin_factor > 1:
        n_out = (X.shape[1] // bin_factor) // 8 * 8
        X = X[:, :n_out * bin_factor].reshape(len(X), n_out, bin_factor).mean(axis=2).astype(np.float32)
        print(f"binned spectra by {bin_factor}: {len(w_cols)} -> {X.shape[1]} points", flush=True)
    with open(f"{root}/food_ids.json") as f:
        food_ids = json.load(f)
    name2id = {v["name"]: k for k, v in food_ids.items()}
    Fnp = pd.get_dummies(df["category"].map(name2id)).reindex(
        columns=list(food_ids), fill_value=0).astype(np.float32).values
    print(f"{machine}: {len(df)} rows, {X.shape[1]} wavelengths, substances={substances}, {cfg}", flush=True)
    for s in substances:
        run_substance_guideddcnet(s, X, Fnp, (df[s] > 0).astype(int).values, machine, "cuda", **cfg)


if __name__ == "__main__":
    _method = os.environ.get("METHOD")
    _mains = {'xgboost': main_xgboost, 'lightgbm': main_lightgbm, 'smartnir': main_smartnir, 'guideddcnet': main_guideddcnet}
    if _method not in _mains:
        raise SystemExit("set METHOD to one of: " + ", ".join(_mains))
    _mains[_method]()

