import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, brier_score_loss, precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import os
import joblib
from dotenv import load_dotenv

from dataset.preprocessing import preprocess_spectra

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


def train_for_substance(X, y, substance_name, n_splits=5, n_estimators=100, learning_rate=0.1, max_depth=6, patience=10,
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

# ---------------- Run ----------------
if __name__ == "__main__":
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

        train_for_substance(X, y, substance, n_splits=n_splits, n_estimators=n_estimators,
                            learning_rate=learning_rate, max_depth=max_depth, patience=10,
                            save_history_dir=save_history_dir, save_fig_dir=save_fig_dir,
                            save_best_model_dir=save_best_model_dir, save_scaler_dir=save_scaler_dir)