"""Measures the metrics the pesticide-6 benchmark CSVs track (precision/
recall/f1 for food classification and pesticide detection, mse/rmse for
concentration regression) for the MT-SMARTNIR 3-head model
(multitask_mt_smartnir_engine.py) across both machines, and appends a
"MT-SMARTNIR" row to each -- the training loop only logged accuracy/f1 and
mae/r2, not precision/recall/mse/rmse, and never the mean+std-across-folds
the benchmark tables report (see additional_experiments/*_benchmark.py for
the same pattern on Mango/Grainit/Rapeseed).

Reconstructs each fold's exact train/val split (same StratifiedKFold seed,
same SAMPLE_N=100000 subsample, same BIN for OCEANFX) and re-fits
normalization on it (deterministic, no randomness) so the saved checkpoints
can be evaluated on the same held-out rows they were validated on during
training.
"""

import os

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                              mean_squared_error, precision_score, r2_score,
                              recall_score)
from sklearn.model_selection import StratifiedKFold

from additional_experiments.benchmark_utils import append_classification_row, append_regression_row
from dataset.pesticide_multitask_dataset import PesticideMultiTaskDataset
from model.mt_smartnir_model import MultiTaskSMARTNIRModel
from xspecmamba_joint_pesticide_engine import DEFAULT_SUBSTANCES

device = "cuda" if torch.cuda.is_available() else "cpu"
k_folds = 5
task = "multitask_mt_smartnir"
substances = DEFAULT_SUBSTANCES

MACHINE_CFG = {
    "FLAMENIR": {"bin_factor": 1},
    "OCEANFX": {"bin_factor": 8},
}

for machine, cfg in MACHINE_CFG.items():
    full_path = f"{os.environ['DATASET_ROOT']}/{machine}/ALL.csv"
    full_ds = PesticideMultiTaskDataset(full_path, substances, bin_factor=cfg["bin_factor"], sample_n=100000)
    splits = list(StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
                  .split(np.arange(len(full_ds.food_raw)), full_ds.food_raw))

    food_accs, food_precs, food_recs, food_f1s = [], [], [], []
    det_accs, det_precs, det_recs, det_f1s = [], [], [], []
    reg_mse = {s: [] for s in substances}
    reg_mae = {s: [] for s in substances}
    reg_rmse = {s: [] for s in substances}
    reg_r2 = {s: [] for s in substances}

    for fold, (train_idx, val_idx) in enumerate(splits):
        full_ds.fit_normalization_and_labels(train_idx, save_dir=None)

        model = MultiTaskSMARTNIRModel(
            n_food_classes=full_ds.n_classes, n_pesticides=len(substances),
            c_out=64, n_layers=6, n_heads=6, h_hidden=128, seq_len=full_ds.signal_length,
            use_kan=True, dropout=0.1,
        ).to(device)
        ckpt_path = f"checkpoint/{task}/{machine}/checkpoint_fold{fold + 1}.pth"
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        food_true, food_pred = [], []
        pest_true, pest_pred = [], []
        conc_true, conc_pred = [], []
        with torch.no_grad():
            for start in range(0, len(val_idx), 256):
                chunk = val_idx[start:start + 256]
                X_b = torch.tensor(full_ds.X[chunk], dtype=torch.float32, device=device)
                food_logits, pest_logits, pest_conc = model(X_b)
                food_pred.extend(food_logits.argmax(-1).cpu().numpy())
                food_true.extend(full_ds.food_y[chunk])
                pest_pred.append((torch.sigmoid(pest_logits) >= 0.5).cpu().numpy().astype(int))
                pest_true.append(full_ds.pres_raw[chunk])
                conc_pred.append(pest_conc.cpu().numpy())
                conc_true.append(full_ds.conc_log[chunk])

        food_true, food_pred = np.array(food_true), np.array(food_pred)
        food_accs.append(accuracy_score(food_true, food_pred) * 100.0)
        food_precs.append(precision_score(food_true, food_pred, average="macro", zero_division=0) * 100.0)
        food_recs.append(recall_score(food_true, food_pred, average="macro", zero_division=0) * 100.0)
        food_f1s.append(f1_score(food_true, food_pred, average="macro", zero_division=0) * 100.0)

        pest_true = np.concatenate(pest_true, axis=0)
        pest_pred = np.concatenate(pest_pred, axis=0)
        p_accs, p_precs, p_recs, p_f1s = [], [], [], []
        for j in range(len(substances)):
            p_accs.append(accuracy_score(pest_true[:, j], pest_pred[:, j]))
            p_precs.append(precision_score(pest_true[:, j], pest_pred[:, j], zero_division=0))
            p_recs.append(recall_score(pest_true[:, j], pest_pred[:, j], zero_division=0))
            p_f1s.append(f1_score(pest_true[:, j], pest_pred[:, j], zero_division=0))
        det_accs.append(float(np.mean(p_accs)) * 100.0)
        det_precs.append(float(np.mean(p_precs)) * 100.0)
        det_recs.append(float(np.mean(p_recs)) * 100.0)
        det_f1s.append(float(np.mean(p_f1s)) * 100.0)

        conc_true = np.concatenate(conc_true, axis=0)
        conc_pred = np.concatenate(conc_pred, axis=0)
        conc_true_mgkg = np.expm1(conc_true)
        conc_pred_mgkg = np.expm1(conc_pred)
        for j, name in enumerate(substances):
            m = pest_true[:, j] == 1
            if m.sum() >= 2:
                mse = mean_squared_error(conc_true_mgkg[m, j], conc_pred_mgkg[m, j])
                reg_mse[name].append(mse)
                reg_mae[name].append(mean_absolute_error(conc_true_mgkg[m, j], conc_pred_mgkg[m, j]))
                reg_rmse[name].append(np.sqrt(mse))
                reg_r2[name].append(r2_score(conc_true_mgkg[m, j], conc_pred_mgkg[m, j]) * 100.0)

        print(f"{machine} fold {fold + 1}/{k_folds}: food_acc={food_accs[-1]:.2f} "
              f"det_f1={det_f1s[-1]:.3f}")

    append_classification_row(
        f"benchmark/multitask_pesticide6/classification/{machine}.csv", "MT-SMARTNIR",
        np.mean(food_accs), np.std(food_accs), np.mean(food_precs), np.std(food_precs),
        np.mean(food_recs), np.std(food_recs), np.mean(food_f1s), np.std(food_f1s),
    )
    append_classification_row(
        f"benchmark/multitask_pesticide6/detection/{machine}.csv", "MT-SMARTNIR",
        np.mean(det_accs), np.std(det_accs), np.mean(det_precs), np.std(det_precs),
        np.mean(det_recs), np.std(det_recs), np.mean(det_f1s), np.std(det_f1s),
    )
    for name in substances:
        if reg_mse[name]:
            append_regression_row(
                f"benchmark/multitask_pesticide6/regression/{machine}.csv", name, "MT-SMARTNIR",
                np.mean(reg_mse[name]), np.std(reg_mse[name]), np.mean(reg_mae[name]), np.std(reg_mae[name]),
                np.mean(reg_rmse[name]), np.std(reg_rmse[name]), np.mean(reg_r2[name]), np.std(reg_r2[name]),
            )
    print(f"{machine}: benchmark CSVs updated.")
