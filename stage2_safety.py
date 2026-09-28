"""Bài toán (2), Bước 2 -- Phân loại mức độ (METHOD=xgboost | lightgbm | smartnir |
guideddcnet). So nồng độ thực đo với MRL
(Maximum Residue Limit) của đúng cặp (loại thực phẩm, thuốc trừ sâu) của
từng mẫu (../all-dataset/Danang-NIR/{thresholds,food_ids,pesticide_ids}.json):

    0 - An toan      : khong phat hien (y == -1), HOAC phat hien nhung
                        nong do <= mrl(food, pesticide)
    1 - Vuot nguong  : phat hien VA nong do > mrl(food, pesticide)

Khác với thiết kế cascade thuần túy (Bước 2 chỉ thấy mẫu Bước 1 đã xác nhận
có phát hiện), model ở đây huấn luyện trên TOÀN BỘ mẫu -- mẫu không phát
hiện được gộp thẳng vào An toàn thay vì bị loại khỏi tập huấn luyện của
Bước 2. Bước 1 (stage1_detection.py) không đổi, vẫn chỉ huấn luyện
trên nhãn phát hiện có/không của riêng nó; ở suy luận, cascade hai bước vẫn
giữ nguyên (Bước 2 chỉ thực sự được hỏi khi Bước 1 báo có phát hiện) -- thay
đổi ở đây chỉ là tập dữ liệu huấn luyện của Bước 2 được mở rộng ra toàn bộ
mẫu, giúp mô hình mạnh hơn ngay cả khi được hỏi độc lập, không chỉ khi được
Bước 1 lọc sẵn.

Chỉ dùng MRL -- không còn safe_limit. Vì MRL khác nhau theo từng loại thực
phẩm, nhãn của mỗi mẫu được gán theo đúng food category của mẫu đó (tra qua
food_ids.json), nhưng vẫn giữ quy ước 1 mô hình/thuốc huấn luyện gộp chung
cả 9 loại thực phẩm (MRL chỉ dùng để gán nhãn, không phải input của model).
Loại thực phẩm cũng được đưa vào làm đặc trưng one-hot tường minh (giống
Bước 1) vì ngưỡng MRL phụ thuộc vào nó.

Tái sử dụng gần như nguyên vẹn cấu trúc calibration + threshold-cho-recall
của stage1_detection.py (Bước 1): bỏ sót một mẫu thực sự vượt
ngưỡng cũng nguy hiểm như bỏ sót một mẫu có thuốc, nên áp dụng cùng triết
lý ưu tiên recall.

Sau hiệu chỉnh Platt scaling, xác suất còn được trộn thêm với một prior
thực nghiệm theo riêng loại thực phẩm (xem food_prior_shrinkage) -- vì phần
lớn các cặp (thực phẩm, thuốc trừ sâu) gần như tất định trong dữ liệu hiện
có (không có mẫu nào gần ranh giới MRL để học, xem Hình 7 trong báo cáo),
một mô hình phổ thuần túy dễ "học" một ranh giới ảo ở những cặp này. Trọng
số trộn giữa prior và xác suất mô hình được suy ra trực tiếp từ chính độ
"tất định" của prior đó (phương sai Bernoulli), không phải một
siêu tham số cần dò, nên không cần dữ liệu mới để cải thiện độ tổng quát ở
những cặp (thực phẩm, thuốc trừ sâu) hiện chưa được khảo sát đầy đủ.

SMART-NIR và GuidedDCNet dùng lại nguyên vẹn các lớp mô hình và vòng lặp
huấn luyện của Bước 1 (import từ stage1_detection.py) -- điểm khác
biệt duy nhất so với Bước 1 nằm ở lớp hiệu chỉnh: sau Platt scaling, xác
suất còn được trộn với food_prior_shrinkage (xem ở trên) trước khi dùng để
chọn ngưỡng Recall>=0.9 và đánh giá.
"""
import json
import os
import time

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    f1_score, precision_recall_curve, precision_score, recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from dataset.preprocessing import preprocess_spectra
from stage1_detection import (
    predict_logit_diff, score_rows, train_one_fold_guideddcnet, train_one_fold_smartnir,
)

load_dotenv()


def pick_xgb_hyperparams(num_pos: int):
    """Same rationale as stage1_detection.py."""
    if num_pos < 50:
        return 100, 0.05, 3
    elif num_pos < 500:
        return 150, 0.08, 4
    else:
        return 200, 0.1, 6


def pick_threshold_for_recall(y_true, y_prob, target_recall=0.9):
    """Same as stage1_detection.py -- swept on the caller's own
    (train fold) data to avoid leakage; highest-precision cut that still
    hits target_recall on "Vuot nguong" (missing a real over-limit sample
    is costlier than a false alarm)."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    precisions, recalls = precisions[:-1], recalls[:-1]
    eligible = np.where(recalls >= target_recall)[0]
    if len(eligible) == 0:
        return float(thresholds[np.argmax(recalls)])
    best = eligible[np.argmax(precisions[eligible])]
    return float(thresholds[best])


def food_prior_shrinkage(y_prob, food_idx, prior_by_food):
    """Blend a calibrated model probability with an empirical per-food prior
    P(Vuot nguong | food), weighted by how deterministic that prior itself
    is (Bernoulli-variance based confidence): alpha = 1 - 4*p*(1-p), which
    is 1 (trust the prior completely) when the food's historical outcome is
    near-certain (p near 0 or 1) and 0 (trust the spectral model completely)
    when the food is genuinely 50/50. Rationale: for many (food, pesticide)
    pairs the training data never observes samples near the MRL boundary at
    all (Hình 7 -- roughly two-thirds of cells are exactly 0% or 100% Vuot
    nguong), so a purely spectral model has no real signal to learn a
    boundary from there and can only overfit noise; falling back to the
    food's own empirical rate is the most defensible, best-generalizing
    answer given the data actually available, without needing new samples.
    For the remaining, genuinely mixed cells, this leaves the spectral model
    in charge, since that's exactly where the data supports learning one.
    `food_idx` is an integer array (row -> food index); `prior_by_food` is
    an array indexed by that same food index, holding each food's Beta(1,1)
    -smoothed empirical P(Vuot nguong) computed on the TRAIN fold only.
    """
    p_food = prior_by_food[food_idx]
    alpha = 1.0 - 4.0 * p_food * (1.0 - p_food)
    alpha = np.clip(alpha, 0.0, 1.0)
    return alpha * p_food + (1.0 - alpha) * y_prob


def train_for_substance(X, y, substance_name, n_splits=5, n_estimators=100, learning_rate=0.1, max_depth=6,
                         patience=10, target_recall=0.9,
                         save_history_dir="history", save_fig_dir="history", save_best_model_dir="checkpoint",
                         save_scaler_dir="scalers"):
    cat_cols = [c for c in X.columns if c.startswith("cat_")]
    # one-hot -> a single integer food index per row (argmax over the cat_ block)
    food_idx_all = X[cat_cols].values.argmax(axis=1) if cat_cols else None
    n_foods = len(cat_cols)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_histories = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"Fold {fold + 1}/{n_splits} for {substance_name}")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Per-food prior P(Vuot nguong), Beta(1,1)-smoothed, estimated on the
        # TRAIN fold only (never val, to avoid leakage) -- used below to
        # shrink the spectral model's prediction back towards the food's own
        # empirical rate wherever that rate is itself near-deterministic.
        prior_by_food = np.full(max(n_foods, 1), 0.5, dtype=np.float64)
        if cat_cols:
            food_idx_train = food_idx_all[train_idx]
            food_idx_val = food_idx_all[val_idx]
            for fidx in range(n_foods):
                mask = food_idx_train == fidx
                n_food = int(mask.sum())
                n_pos_food = int(y_train.values[mask].sum()) if n_food else 0
                prior_by_food[fidx] = (n_pos_food + 1) / (n_food + 2)  # Beta(1,1)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        scaler_path = os.path.join(save_scaler_dir, f"{substance_name}_fold_{fold + 1}_scaler.pkl")
        os.makedirs(save_scaler_dir, exist_ok=True)
        joblib.dump(scaler, scaler_path)

        dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
        dval = xgb.DMatrix(X_val_scaled, label=y_val)

        num_pos_train = int((y_train == 1).sum())
        num_neg_train = int((y_train == 0).sum())
        scale_pos_weight = num_neg_train / max(num_pos_train, 1)

        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'eta': learning_rate,
            'max_depth': max_depth,
            'tree_method': 'hist',
            'scale_pos_weight': scale_pos_weight,
        }

        evals = [(dtrain, 'train'), (dval, 'val')]
        model = xgb.train(params, dtrain, num_boost_round=n_estimators, evals=evals,
                           early_stopping_rounds=patience, verbose_eval=False)

        y_pred_prob_train = model.predict(dtrain)
        calibrator = LogisticRegression()
        calibrator.fit(y_pred_prob_train.reshape(-1, 1), y_train)

        y_pred_prob_train_cal = calibrator.predict_proba(y_pred_prob_train.reshape(-1, 1))[:, 1]
        if cat_cols:
            y_pred_prob_train_final = food_prior_shrinkage(y_pred_prob_train_cal, food_idx_train, prior_by_food)
        else:
            y_pred_prob_train_final = y_pred_prob_train_cal
        threshold = pick_threshold_for_recall(y_train, y_pred_prob_train_final, target_recall=target_recall)

        y_pred_prob_raw = model.predict(dval)
        y_pred_prob_cal = calibrator.predict_proba(y_pred_prob_raw.reshape(-1, 1))[:, 1]
        if cat_cols:
            y_pred_prob = food_prior_shrinkage(y_pred_prob_cal, food_idx_val, prior_by_food)
        else:
            y_pred_prob = y_pred_prob_cal
        y_pred = (y_pred_prob > threshold).astype(int)

        calibrator_path = os.path.join(save_scaler_dir, f"{substance_name}_fold_{fold + 1}_calibrator.pkl")
        joblib.dump(calibrator, calibrator_path)
        threshold_path = os.path.join(save_scaler_dir, f"{substance_name}_fold_{fold + 1}_threshold.json")
        with open(threshold_path, "w") as f:
            json.dump({"threshold": threshold, "target_recall": target_recall}, f, indent=4)
        prior_path = os.path.join(save_scaler_dir, f"{substance_name}_fold_{fold + 1}_food_prior.json")
        with open(prior_path, "w") as f:
            json.dump({"prior_by_food": prior_by_food.tolist(), "cat_cols": cat_cols}, f, indent=4)

        val_acc = accuracy_score(y_val, y_pred)
        val_precision = precision_score(y_val, y_pred, average='macro', zero_division=0)
        val_recall = recall_score(y_val, y_pred, average='macro', zero_division=0)
        val_f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)
        val_pr_auc = average_precision_score(y_val, y_pred_prob)
        brier_before = brier_score_loss(y_val, y_pred_prob_raw)
        brier_after = brier_score_loss(y_val, y_pred_prob_cal)
        brier_after_shrinkage = brier_score_loss(y_val, y_pred_prob)

        history = {
            "threshold": [threshold],
            "val_acc": [val_acc],
            "val_precision": [val_precision],
            "val_recall": [val_recall],
            "val_f1": [val_f1],
            "val_pr_auc": [val_pr_auc],
            "brier_before_calibration": [brier_before],
            "brier_after_calibration": [brier_after],
            "brier_after_shrinkage": [brier_after_shrinkage],
        }

        save_history_path = os.path.join(save_history_dir, f"{substance_name}_fold_{fold + 1}.json")
        os.makedirs(save_history_dir, exist_ok=True)
        with open(save_history_path, "w") as f:
            json.dump(history, f, indent=4)

        save_best_model_path = os.path.join(save_best_model_dir, f"{substance_name}_fold_{fold + 1}.json")
        os.makedirs(save_best_model_dir, exist_ok=True)
        model.save_model(save_best_model_path)

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


def train_for_substance_lightgbm(X, y, substance_name, n_splits=5, n_estimators=100, learning_rate=0.1,
                                  max_depth=6, patience=20, target_recall=0.9, n_jobs=32,
                                  save_history_dir="history", save_best_model_dir="checkpoint",
                                  save_scaler_dir="scalers"):
    """Same protocol as train_for_substance (XGBoost) above, including the
    food-prior shrinkage step -- only the underlying tree library differs.
    Mirrors stage1_detection.py's run_substance_lightgbm: early stopping on
    validation average precision rather than logloss (see that function's
    docstring for why: scale_pos_weight otherwise makes unweighted logloss
    lowest at iteration 1-2, freezing rare substances at a single tree), no
    separate StandardScaler step (trees are scale-invariant).
    """
    cat_cols = [c for c in X.columns if c.startswith("cat_")]
    food_idx_all = X[cat_cols].values.argmax(axis=1) if cat_cols else None
    n_foods = len(cat_cols)
    Xv = X.values
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_histories = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(Xv, y)):
        print(f"Fold {fold + 1}/{n_splits} for {substance_name}")
        X_train, X_val = Xv[train_idx], Xv[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        prior_by_food = np.full(max(n_foods, 1), 0.5, dtype=np.float64)
        if cat_cols:
            food_idx_train = food_idx_all[train_idx]
            food_idx_val = food_idx_all[val_idx]
            for fidx in range(n_foods):
                mask = food_idx_train == fidx
                n_food = int(mask.sum())
                n_pos_food = int(y_train.values[mask].sum()) if n_food else 0
                prior_by_food[fidx] = (n_pos_food + 1) / (n_food + 2)

        params = dict(
            objective="binary", learning_rate=learning_rate, max_depth=max_depth,
            num_leaves=min(2 ** max_depth - 1, 63), n_estimators=n_estimators,
            scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
            n_jobs=n_jobs, random_state=42, verbose=-1, force_col_wise=True,
        )
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="average_precision",
                  callbacks=[lgb.early_stopping(patience, first_metric_only=True, verbose=False)])

        y_pred_prob_train = model.predict_proba(X_train)[:, 1]
        calibrator = LogisticRegression()
        calibrator.fit(y_pred_prob_train.reshape(-1, 1), y_train)

        y_pred_prob_train_cal = calibrator.predict_proba(y_pred_prob_train.reshape(-1, 1))[:, 1]
        y_pred_prob_train_final = (food_prior_shrinkage(y_pred_prob_train_cal, food_idx_train, prior_by_food)
                                   if cat_cols else y_pred_prob_train_cal)
        threshold = pick_threshold_for_recall(y_train, y_pred_prob_train_final, target_recall=target_recall)

        y_pred_prob_raw = model.predict_proba(X_val)[:, 1]
        y_pred_prob_cal = calibrator.predict_proba(y_pred_prob_raw.reshape(-1, 1))[:, 1]
        y_pred_prob = (food_prior_shrinkage(y_pred_prob_cal, food_idx_val, prior_by_food)
                      if cat_cols else y_pred_prob_cal)
        y_pred = (y_pred_prob > threshold).astype(int)

        os.makedirs(save_scaler_dir, exist_ok=True)
        calibrator_path = os.path.join(save_scaler_dir, f"{substance_name}_fold_{fold + 1}_calibrator.pkl")
        joblib.dump(calibrator, calibrator_path)
        with open(os.path.join(save_scaler_dir, f"{substance_name}_fold_{fold + 1}_threshold.json"), "w") as f:
            json.dump({"threshold": threshold, "target_recall": target_recall}, f, indent=4)
        with open(os.path.join(save_scaler_dir, f"{substance_name}_fold_{fold + 1}_food_prior.json"), "w") as f:
            json.dump({"prior_by_food": prior_by_food.tolist(), "cat_cols": cat_cols}, f, indent=4)

        val_acc = accuracy_score(y_val, y_pred)
        val_precision = precision_score(y_val, y_pred, average='macro', zero_division=0)
        val_recall = recall_score(y_val, y_pred, average='macro', zero_division=0)
        val_f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)
        val_pr_auc = average_precision_score(y_val, y_pred_prob)
        history = {
            "threshold": [threshold],
            "val_acc": [val_acc],
            "val_precision": [val_precision],
            "val_recall": [val_recall],
            "val_f1": [val_f1],
            "val_pr_auc": [val_pr_auc],
            "brier_before_calibration": [brier_score_loss(y_val, y_pred_prob_raw)],
            "brier_after_calibration": [brier_score_loss(y_val, y_pred_prob_cal)],
            "brier_after_shrinkage": [brier_score_loss(y_val, y_pred_prob)],
        }
        os.makedirs(save_history_dir, exist_ok=True)
        with open(os.path.join(save_history_dir, f"{substance_name}_fold_{fold + 1}.json"), "w") as f:
            json.dump(history, f, indent=4)
        os.makedirs(save_best_model_dir, exist_ok=True)
        model.booster_.save_model(os.path.join(save_best_model_dir, f"{substance_name}_fold_{fold + 1}.txt"))
        fold_histories.append(history)

    avg_val_acc = np.mean([h["val_acc"][0] for h in fold_histories])
    avg_pr_auc = np.mean([h["val_pr_auc"][0] for h in fold_histories])
    print(f"Average validation accuracy across folds for {substance_name}: {avg_val_acc:.4f} "
          f"(PR-AUC: {avg_pr_auc:.4f})")


# ======================================================================
# entry point: METHOD=lightgbm
# ======================================================================
def main_lightgbm():
    """Buoc 2 (an toan / vuot nguong) voi LightGBM, cung giao thuc nhu XGBoost
    o main_xgboost ben duoi (bao gom food_prior_shrinkage), chi khac thu vien
    cay va tieu chi dung som (average precision, giong stage1_detection.py
    METHOD=lightgbm).

    Env: MACHINE, SUBSTANCES (comma list; default all 19), N_JOBS.
    """
    machine = os.environ.get("MACHINE", "FLAMENIR")
    task = "substance_severity_lightgbm"
    n_splits = 5
    n_jobs = int(os.environ.get("N_JOBS", 32))

    dataset_root = os.environ['DATASET_ROOT']
    full_path = f"{dataset_root}/{machine}/ALL.csv"
    with open(f"{dataset_root}/food_ids.json") as f:
        food_ids = json.load(f)
    with open(f"{dataset_root}/pesticide_ids.json") as f:
        pesticide_ids = json.load(f)
    with open(f"{dataset_root}/thresholds.json") as f:
        thresholds = json.load(f)
    food_name_to_id = {v["name"]: k for k, v in food_ids.items()}
    pesticide_name_to_id = {v["name"]: k for k, v in pesticide_ids.items()}

    df = pd.read_csv(full_path)
    wavelength_cols = [col for col in df.columns if col.startswith('w_')]
    X_raw, keep_mask = preprocess_spectra(df[wavelength_cols].values.astype(np.float32))
    n_dropped = (~keep_mask).sum()
    if n_dropped:
        print(f"Dropped {n_dropped} outlier spectra out of {len(keep_mask)}")
    df = df[keep_mask].reset_index(drop=True)
    X = pd.DataFrame(X_raw, columns=wavelength_cols)

    food_id_per_row = df["category"].map(food_name_to_id)
    cat_onehot = pd.get_dummies(food_id_per_row, prefix="cat").reindex(
        columns=[f"cat_{fid}" for fid in food_ids], fill_value=0
    ).astype(np.float32)
    X = pd.concat([X, cat_onehot.reset_index(drop=True)], axis=1)

    subs = os.environ.get("SUBSTANCES")
    substances = [s.strip() for s in subs.split(",")] if subs else list(pesticide_name_to_id.keys())

    for substance in substances:
        print(f"Training for {substance}")
        pid = pesticide_name_to_id[substance]
        mrl_per_row = food_id_per_row.map(lambda fid: thresholds[fid][pid]["mrl"])
        y_full = ((df[substance] != -1) & (df[substance] > mrl_per_row)).astype(int)

        num_pos = int((y_full == 1).sum())
        num_neg = int((y_full == 0).sum())
        save_history_dir = f"history/{task}/{machine}/{substance}"
        save_best_model_dir = f"checkpoint/{task}/{machine}/{substance}"
        save_scaler_dir = f"data/{task}/{machine}/{substance}"
        if num_pos < n_splits or num_neg < n_splits:
            print(f"Not enough samples for {substance} (An toan={num_neg}, Vuot nguong={num_pos}), skipping.")
            for d in (save_history_dir, save_best_model_dir, save_scaler_dir):
                os.makedirs(d, exist_ok=True)
            continue

        n_estimators, learning_rate, max_depth = pick_xgb_hyperparams(num_pos)
        print(f"{substance}: An toan={num_neg}, Vuot nguong={num_pos} -> "
              f"n_estimators={n_estimators}, lr={learning_rate}, max_depth={max_depth}")
        train_for_substance_lightgbm(X, y_full, substance, n_splits=n_splits, n_estimators=n_estimators,
                                      learning_rate=learning_rate, max_depth=max_depth, n_jobs=n_jobs,
                                      save_history_dir=save_history_dir, save_best_model_dir=save_best_model_dir,
                                      save_scaler_dir=save_scaler_dir)


# ======================================================================
# smartnir / guideddcnet -- shared calibration layer
# ======================================================================
# Both reuse the Buoc 1 model classes and training loops verbatim (imported
# above); the food-index bookkeeping and calibration below is what's specific
# to Buoc 2 (Platt scaling -> food_prior_shrinkage -> Recall>=0.9 threshold),
# mirroring train_for_substance (XGBoost) above.

def _food_prior(y_tr, food_idx_tr, n_foods):
    prior_by_food = np.full(max(n_foods, 1), 0.5, dtype=np.float64)
    for fidx in range(n_foods):
        mask = food_idx_tr == fidx
        n_food = int(mask.sum())
        n_pos_food = int(y_tr[mask].sum()) if n_food else 0
        prior_by_food[fidx] = (n_pos_food + 1) / (n_food + 2)  # Beta(1,1)
    return prior_by_food


def run_substance_smartnir(substance, Xnp, Fnp, y, machine, device, max_epochs, bs,
                            pos_frac=0.0, folds=None, cal_frac=0.15):
    task = "substance_severity_smartnir"
    hist_dir = f"history/{task}/{machine}/{substance}"
    ckpt_dir = f"checkpoint/{task}/{machine}/{substance}"
    data_dir = f"data/{task}/{machine}/{substance}"
    for d in (hist_dir, ckpt_dir, data_dir):
        os.makedirs(d, exist_ok=True)

    food_idx_all = Fnp.argmax(axis=1)
    n_foods = Fnp.shape[1]
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

        # Same held-back calibration slice used for both Platt scaling and the
        # per-food prior below (never seen by the optimizer), for the same
        # no-leakage reason as stage1_detection.py.
        if cal_frac > 0:
            tr_fit, tr_cal = train_test_split(tr, test_size=cal_frac, stratify=y[tr], random_state=fold)
        else:
            tr_fit, tr_cal = tr, tr
        model = train_one_fold_smartnir(Xs, Fs, ys, tr_fit, va, device, max_epochs, bs, pos_frac=pos_frac)

        prior_by_food = _food_prior(y[tr_cal], food_idx_all[tr_cal], n_foods)

        d_tr = predict_logit_diff(model, Xs, Fs, torch.as_tensor(tr_cal, device=device), bs * 2)
        calib = LogisticRegression().fit(d_tr.reshape(-1, 1), y[tr_cal])
        p_tr = food_prior_shrinkage(calib.predict_proba(d_tr.reshape(-1, 1))[:, 1], food_idx_all[tr_cal], prior_by_food)
        thr = pick_threshold_for_recall(y[tr_cal], p_tr, target_recall=0.9)

        d_va = predict_logit_diff(model, Xs, Fs, torch.as_tensor(va, device=device), bs * 2)
        p_raw = 1.0 / (1.0 + np.exp(-d_va))
        p_va_cal = calib.predict_proba(d_va.reshape(-1, 1))[:, 1]
        p_va = food_prior_shrinkage(p_va_cal, food_idx_all[va], prior_by_food)
        yv = y[va]
        pred = (p_va > thr).astype(int)
        hist = {
            "threshold": [thr],
            "val_acc": [accuracy_score(yv, pred)],
            "val_precision": [precision_score(yv, pred, average="macro", zero_division=0)],
            "val_recall": [recall_score(yv, pred, average="macro", zero_division=0)],
            "val_f1": [f1_score(yv, pred, average="macro", zero_division=0)],
            "val_pr_auc": [average_precision_score(yv, p_va)],
            "brier_before_calibration": [brier_score_loss(yv, p_raw)],
            "brier_after_calibration": [brier_score_loss(yv, p_va_cal)],
            "brier_after_shrinkage": [brier_score_loss(yv, p_va)],
            "val_pos_recall": [recall_score(yv, pred, zero_division=0)],
            "val_pos_precision": [precision_score(yv, pred, zero_division=0)],
            "seconds": [time.time() - t0],
        }
        with open(f"{hist_dir}/{substance}_fold_{fold}.json", "w") as f:
            json.dump(hist, f, indent=4)
        torch.save(model.state_dict(), f"{ckpt_dir}/{substance}_fold_{fold}.pth")
        joblib.dump(calib, f"{data_dir}/{substance}_fold_{fold}_calibrator.pkl")
        with open(f"{data_dir}/{substance}_fold_{fold}_threshold.json", "w") as f:
            json.dump({"threshold": thr, "target_recall": 0.9}, f, indent=4)
        with open(f"{data_dir}/{substance}_fold_{fold}_food_prior.json", "w") as f:
            json.dump({"prior_by_food": prior_by_food.tolist()}, f, indent=4)
        print(f"  -> acc={hist['val_acc'][0]:.4f} PR-AUC={hist['val_pr_auc'][0]:.4f} "
              f"pos_recall={hist['val_pos_recall'][0]:.3f} pos_prec={hist['val_pos_precision'][0]:.3f} "
              f"brier_shrink={hist['brier_after_shrinkage'][0]:.4f} ({hist['seconds'][0]:.0f}s)", flush=True)
        del Xs
        torch.cuda.empty_cache()


def run_substance_guideddcnet(substance, Xnp, Fnp, y, machine, device, pre_epochs, max_epochs, bs, eval_every,
                               pos_frac, cal_frac, folds, lr_scale=1.0, wd=0.0, use_last=False):
    task = "substance_severity_guideddcnet"
    hist_dir = f"history/{task}/{machine}/{substance}"
    ckpt_dir = f"checkpoint/{task}/{machine}/{substance}"
    data_dir = f"data/{task}/{machine}/{substance}"
    for d in (hist_dir, ckpt_dir, data_dir):
        os.makedirs(d, exist_ok=True)

    food_idx_all = Fnp.argmax(axis=1)
    n_foods = Fnp.shape[1]
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
        model = train_one_fold_guideddcnet(Xs, Fs, ys, tr_fit, va, device, pre_epochs, max_epochs, bs, eval_every,
                                           pos_frac, lr_scale=lr_scale, wd=wd, use_last=use_last)

        prior_by_food = _food_prior(y[tr_cal], food_idx_all[tr_cal], n_foods)

        d_tr = score_rows(model, Xs, Fs, torch.as_tensor(tr_cal, device=device), bs * 2)
        calib = LogisticRegression().fit(d_tr.reshape(-1, 1), y[tr_cal])
        p_tr = food_prior_shrinkage(calib.predict_proba(d_tr.reshape(-1, 1))[:, 1], food_idx_all[tr_cal], prior_by_food)
        thr = pick_threshold_for_recall(y[tr_cal], p_tr, target_recall=0.9)

        d_va = score_rows(model, Xs, Fs, torch.as_tensor(va, device=device), bs * 2)
        p_raw = 1.0 / (1.0 + np.exp(-d_va))
        p_va_cal = calib.predict_proba(d_va.reshape(-1, 1))[:, 1]
        p_va = food_prior_shrinkage(p_va_cal, food_idx_all[va], prior_by_food)
        yv = y[va]
        pred = (p_va > thr).astype(int)
        hist = {
            "threshold": [thr],
            "val_acc": [accuracy_score(yv, pred)],
            "val_precision": [precision_score(yv, pred, average="macro", zero_division=0)],
            "val_recall": [recall_score(yv, pred, average="macro", zero_division=0)],
            "val_f1": [f1_score(yv, pred, average="macro", zero_division=0)],
            "val_pr_auc": [average_precision_score(yv, p_va)],
            "brier_before_calibration": [brier_score_loss(yv, p_raw)],
            "brier_after_calibration": [brier_score_loss(yv, p_va_cal)],
            "brier_after_shrinkage": [brier_score_loss(yv, p_va)],
            "val_pos_recall": [recall_score(yv, pred, zero_division=0)],
            "val_pos_precision": [precision_score(yv, pred, zero_division=0)],
            "seconds": [time.time() - t0],
        }
        with open(f"{hist_dir}/{substance}_fold_{fold}.json", "w") as f:
            json.dump(hist, f, indent=4)
        torch.save(model.state_dict(), f"{ckpt_dir}/{substance}_fold_{fold}.pth")
        joblib.dump(calib, f"{data_dir}/{substance}_fold_{fold}_calibrator.pkl")
        with open(f"{data_dir}/{substance}_fold_{fold}_threshold.json", "w") as f:
            json.dump({"threshold": thr, "target_recall": 0.9}, f, indent=4)
        with open(f"{data_dir}/{substance}_fold_{fold}_food_prior.json", "w") as f:
            json.dump({"prior_by_food": prior_by_food.tolist()}, f, indent=4)
        print(f"  -> acc={hist['val_acc'][0]:.4f} PR-AUC={hist['val_pr_auc'][0]:.4f} "
              f"pos_recall={hist['val_pos_recall'][0]:.3f} pos_prec={hist['val_pos_precision'][0]:.3f} "
              f"brier_shrink={hist['brier_after_shrinkage'][0]:.4f} ({hist['seconds'][0]:.0f}s)", flush=True)
        del Xs
        torch.cuda.empty_cache()


def _load_severity_data(machine, bin_factor=1):
    """Shared data loading for main_smartnir/main_guideddcnet: same spectra
    preprocessing, food one-hot and per-row MRL severity label as
    main_xgboost below, factored out since both deep methods need it as
    plain numpy (not the pandas DataFrame the XGBoost path uses).

    bin_factor > 1 average-pools neighbouring wavelengths (same rationale as
    stage1_detection.py's BIN option): OCEANFX's 2136 raw points is what
    triggered a CUDA OOM the first time SMART-NIR ran Buoc 2 here at full
    resolution (this option was missing then) -- BIN=8 was the config that
    worked for stage1_detection.py/food_classification.py on OCEANFX.
    """
    dataset_root = os.environ['DATASET_ROOT']
    with open(f"{dataset_root}/food_ids.json") as f:
        food_ids = json.load(f)
    with open(f"{dataset_root}/pesticide_ids.json") as f:
        pesticide_ids = json.load(f)
    with open(f"{dataset_root}/thresholds.json") as f:
        thresholds = json.load(f)
    food_name_to_id = {v["name"]: k for k, v in food_ids.items()}
    pesticide_name_to_id = {v["name"]: k for k, v in pesticide_ids.items()}

    df = pd.read_csv(f"{dataset_root}/{machine}/ALL.csv")
    w_cols = [c for c in df.columns if c.startswith('w_')]
    X, keep = preprocess_spectra(df[w_cols].values.astype(np.float32))
    if bin_factor > 1:
        n_out = (X.shape[1] // bin_factor) // 8 * 8
        X = X[:, :n_out * bin_factor].reshape(len(X), n_out, bin_factor).mean(axis=2).astype(np.float32)
        print(f"binned spectra by {bin_factor}: {len(w_cols)} -> {X.shape[1]} points", flush=True)
    n_dropped = (~keep).sum()
    if n_dropped:
        print(f"Dropped {n_dropped} outlier spectra out of {len(keep)}")
    df = df[keep].reset_index(drop=True)

    unmapped_cats = set(df["category"].unique()) - set(food_name_to_id)
    if unmapped_cats:
        raise ValueError(f"category values with no FOOD_ID mapping: {unmapped_cats}")
    food_id_per_row = df["category"].map(food_name_to_id)
    Fnp = pd.get_dummies(food_id_per_row).reindex(columns=list(food_ids), fill_value=0).astype(np.float32).values

    return X, Fnp, df, food_id_per_row, pesticide_name_to_id, thresholds


def _severity_label(df, substance, food_id_per_row, pid, thresholds):
    mrl_per_row = food_id_per_row.map(lambda fid: thresholds[fid][pid]["mrl"])
    y_raw = df[substance]
    return ((y_raw != -1) & (y_raw > mrl_per_row)).astype(int).values


# ======================================================================
# entry point: METHOD=smartnir
# ======================================================================
def main_smartnir():
    """Buoc 2 (an toan / vuot nguong) voi SMART-NIR + food one-hot, cung cau
    hinh da kiem chung o Buoc 1 (stage1_detection.py METHOD=smartnir,
    TAG=stage1_smartnir_os2): POS_FRAC=0.25, CAL_FRAC=0.15, MAX_EPOCHS=50.

    Env: MACHINE, SUBSTANCES (comma list; default all 19), MAX_EPOCHS,
    BATCH_SIZE, POS_FRAC, CAL_FRAC, FOLDS, BIN (wavelength binning factor;
    use 8 for OCEANFX -- unbinned 2136 points OOM'd at bs=2048).
    """
    machine = os.environ.get("MACHINE", "FLAMENIR")
    max_epochs = int(os.environ.get("MAX_EPOCHS", 50))
    bs = int(os.environ.get("BATCH_SIZE", 2048))
    pos_frac = float(os.environ.get("POS_FRAC", 0.25))
    cal_frac = float(os.environ.get("CAL_FRAC", 0.15))
    folds = [int(f) for f in os.environ["FOLDS"].split(",")] if os.environ.get("FOLDS") else None
    device = "cuda"

    X, Fnp, df, food_id_per_row, pesticide_name_to_id, thresholds = _load_severity_data(
        machine, bin_factor=int(os.environ.get("BIN", 1)))
    subs = os.environ.get("SUBSTANCES")
    substances = [s.strip() for s in subs.split(",")] if subs else list(pesticide_name_to_id.keys())
    print(f"{machine}: {len(df)} rows, {X.shape[1]} wavelengths, substances={substances}", flush=True)

    for s in substances:
        y = _severity_label(df, s, food_id_per_row, pesticide_name_to_id[s], thresholds)
        num_pos, num_neg = int(y.sum()), int((1 - y).sum())
        if num_pos < 5 or num_neg < 5:
            print(f"Not enough samples for {s} (An toan={num_neg}, Vuot nguong={num_pos}), skipping.")
            continue
        run_substance_smartnir(s, X, Fnp, y, machine, device, max_epochs, bs,
                               pos_frac=pos_frac, folds=folds, cal_frac=cal_frac)


# ======================================================================
# entry point: METHOD=guideddcnet
# ======================================================================
def main_guideddcnet():
    """Buoc 2 (an toan / vuot nguong) voi GuidedDCNet + food one-hot, cung
    cau hinh da dung o Buoc 1 (stage1_detection.py METHOD=guideddcnet):
    pretrain 8 epoch + khuech tan 30 epoch, POS_FRAC=0.25, CAL_FRAC=0.15 --
    ngan hon cau hinh day du (25+200, dung cho phan loai thuc pham, 1 tac
    vu 9-lop) vi o day co 19 tac vu nhi phan rieng biet (19 x 5 fold).

    BATCH_SIZE=3072 (khong phai 1024 nhu main_guideddcnet cua Buoc 1): do
    rieng cho ham nay (khong DataLoader, giong benchmark da lam cho
    food_classification.py) cho thay epoch khuech tan tai bs=1024 mat ~60s
    (gom ca vong lap reverse-sample T=100 buoc tren tap validation) trong khi
    bs=3072 chi ~17-22s.

    Env: MACHINE, SUBSTANCES, PRETRAIN_EPOCHS, MAX_EPOCHS, EVAL_EVERY,
    BATCH_SIZE, POS_FRAC, CAL_FRAC, FOLDS, LR_SCALE, WD, BIN (wavelength
    binning factor; use 8 for OCEANFX, same rationale as main_smartnir),
    USE_LAST (1 = keep the final-epoch weights instead of the best-by-PR-AUC
    checkpoint; useful when EVAL_EVERY is large since few epochs get evaluated).
    """
    machine = os.environ.get("MACHINE", "FLAMENIR")
    cfg = dict(
        pre_epochs=int(os.environ.get("PRETRAIN_EPOCHS", 8)),
        max_epochs=int(os.environ.get("MAX_EPOCHS", 30)),
        bs=int(os.environ.get("BATCH_SIZE", 3072)),
        eval_every=int(os.environ.get("EVAL_EVERY", 2)),
        pos_frac=float(os.environ.get("POS_FRAC", 0.25)),
        cal_frac=float(os.environ.get("CAL_FRAC", 0.15)),
        folds=[int(f) for f in os.environ["FOLDS"].split(",")] if os.environ.get("FOLDS") else None,
        lr_scale=float(os.environ.get("LR_SCALE", 1.0)),
        wd=float(os.environ.get("WD", 0.0)),
        use_last=bool(int(os.environ.get("USE_LAST", 0))),
    )
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda"

    X, Fnp, df, food_id_per_row, pesticide_name_to_id, thresholds = _load_severity_data(
        machine, bin_factor=int(os.environ.get("BIN", 1)))
    subs = os.environ.get("SUBSTANCES")
    substances = [s.strip() for s in subs.split(",")] if subs else list(pesticide_name_to_id.keys())
    print(f"{machine}: {len(df)} rows, {X.shape[1]} wavelengths, substances={substances}, {cfg}", flush=True)

    for s in substances:
        y = _severity_label(df, s, food_id_per_row, pesticide_name_to_id[s], thresholds)
        num_pos, num_neg = int(y.sum()), int((1 - y).sum())
        if num_pos < 5 or num_neg < 5:
            print(f"Not enough samples for {s} (An toan={num_neg}, Vuot nguong={num_pos}), skipping.")
            continue
        run_substance_guideddcnet(s, X, Fnp, y, machine, device, **cfg)


# ======================================================================
# entry point: METHOD=xgboost
# ======================================================================
def main_xgboost():
    machine = f"{os.environ['MACHINE']}"
    task = "substance_severity"
    n_splits = 5

    dataset_root = os.environ['DATASET_ROOT']
    full_path = f"{dataset_root}/{machine}/ALL.csv"
    with open(f"{dataset_root}/food_ids.json") as f:
        food_ids = json.load(f)
    with open(f"{dataset_root}/pesticide_ids.json") as f:
        pesticide_ids = json.load(f)
    with open(f"{dataset_root}/thresholds.json") as f:
        thresholds = json.load(f)

    food_name_to_id = {v["name"]: k for k, v in food_ids.items()}
    pesticide_name_to_id = {v["name"]: k for k, v in pesticide_ids.items()}

    df = pd.read_csv(full_path)
    wavelength_cols = [col for col in df.columns if col.startswith('w_')]

    X_raw = df[wavelength_cols].values.astype(np.float32)
    X_raw, keep_mask = preprocess_spectra(X_raw)
    n_dropped = (~keep_mask).sum()
    if n_dropped:
        print(f"Dropped {n_dropped} outlier spectra out of {len(keep_mask)}")
    df = df[keep_mask].reset_index(drop=True)
    X = pd.DataFrame(X_raw, columns=wavelength_cols)

    unmapped_cats = set(df["category"].unique()) - set(food_name_to_id)
    if unmapped_cats:
        raise ValueError(f"category values with no FOOD_ID mapping: {unmapped_cats}")
    food_id_per_row = df["category"].map(food_name_to_id)

    # Food category as an explicit input feature (one-hot over FOOD_ID), same
    # rationale/empirical check as stage1_detection.py -- MRL itself
    # depends on (food, pesticide), so giving the model food identity
    # directly is strictly easier than making it re-derive food type from
    # spectral shape on its own.
    cat_onehot = pd.get_dummies(food_id_per_row, prefix="cat").reindex(
        columns=[f"cat_{fid}" for fid in food_ids], fill_value=0
    ).astype(np.float32)
    X = pd.concat([X, cat_onehot.reset_index(drop=True)], axis=1)

    substances = list(pesticide_name_to_id.keys())

    for substance in substances:
        print(f"Training for {substance}")
        pid = pesticide_name_to_id[substance]
        # Per-row MRL: same pesticide, but the limit depends on which food
        # this particular sample is (thresholds.json[FOOD_ID][PESTICIDE_ID]).
        mrl_per_row = food_id_per_row.map(lambda fid: thresholds[fid][pid]["mrl"])

        y_raw = df[substance]

        save_history_dir = f"history/{task}/{machine}/{substance}"
        save_fig_dir = save_history_dir
        save_best_model_dir = f"checkpoint/{task}/{machine}/{substance}"
        save_scaler_dir = f"data/{task}/{machine}/{substance}"

        # Train on the FULL dataset, not just detected samples: samples with
        # no detection (y_raw == -1) are folded directly into An toan (0),
        # same as samples that were detected but stayed under MRL, rather
        # than being excluded from this step's training set entirely. Bước 1
        # (stage1_detection.py) is unchanged and still only trains
        # on its own presence/absence label; this only changes what Bước 2
        # itself trains on, giving it the full sample pool (and making it
        # robust standalone, not just as a filter fed exclusively by Bước 1).
        y_full = ((y_raw != -1) & (y_raw > mrl_per_row)).astype(int)

        num_pos = int((y_full == 1).sum())  # Vuot nguong
        num_neg = int((y_full == 0).sum())  # An toan (not detected, or detected <= MRL)
        if num_pos < n_splits or num_neg < n_splits:
            print(f"Not enough samples for {substance} (An toan={num_neg}, Vuot nguong={num_pos}), "
                  f"need >= {n_splits} of each for {n_splits}-fold CV. Skipping.")
            os.makedirs(save_history_dir, exist_ok=True)
            os.makedirs(save_best_model_dir, exist_ok=True)
            os.makedirs(save_scaler_dir, exist_ok=True)
            continue

        n_estimators, learning_rate, max_depth = pick_xgb_hyperparams(num_pos)
        print(f"{substance}: An toan={num_neg}, Vuot nguong={num_pos} -> "
              f"n_estimators={n_estimators}, lr={learning_rate}, max_depth={max_depth}")

        train_for_substance(X, y_full, substance, n_splits=n_splits, n_estimators=n_estimators,
                             learning_rate=learning_rate, max_depth=max_depth, patience=10,
                             save_history_dir=save_history_dir, save_fig_dir=save_fig_dir,
                             save_best_model_dir=save_best_model_dir, save_scaler_dir=save_scaler_dir)


if __name__ == "__main__":
    _method = os.environ.get("METHOD", "xgboost")
    _mains = {"xgboost": main_xgboost, "lightgbm": main_lightgbm, "smartnir": main_smartnir,
              "guideddcnet": main_guideddcnet}
    if _method not in _mains:
        raise SystemExit("set METHOD to one of: " + ", ".join(_mains))
    _mains[_method]()
