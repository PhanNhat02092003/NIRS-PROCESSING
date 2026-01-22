import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import os
import joblib

def train_for_substance(X, y, substance_name, n_splits=5, n_estimators=100, learning_rate=0.1, max_depth=6, patience=10, 
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

        # Params
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'eta': learning_rate,
            'max_depth': max_depth,
            'tree_method': 'hist',
        }

        # Train with early stopping
        evals = [(dtrain, 'train'), (dval, 'val')]
        model = xgb.train(params, dtrain, num_boost_round=n_estimators, evals=evals, early_stopping_rounds=patience, verbose_eval=False)

        # Predict on val
        y_pred_prob = model.predict(dval)
        y_pred = (y_pred_prob > 0.5).astype(int)

        # Metrics
        val_acc = accuracy_score(y_val, y_pred)
        val_precision = precision_score(y_val, y_pred, average='macro', zero_division=0)
        val_recall = recall_score(y_val, y_pred, average='macro', zero_division=0)
        val_f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)

        # History (simplified, since no epochs)
        history = {
            "val_acc": [val_acc],
            "val_precision": [val_precision],
            "val_recall": [val_recall],
            "val_f1": [val_f1]
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
        metrics = ['acc', 'precision', 'recall', 'f1']
        values = [val_acc, val_precision, val_recall, val_f1]
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
    print(f"Average validation accuracy across folds for {substance_name}: {avg_val_acc:.4f}")

# ---------------- Run ----------------
if __name__ == "__main__":
    machine = "FLAMENIR"
    task = "substance_regression"

    full_path = f"data/merge/{machine}/ALL.csv"  # Example path, adjust accordingly
    df = pd.read_csv(full_path)

    # Assume columns: w_1, w_2, ..., w_n for wavelengths, and substance columns
    wavelength_cols = [col for col in df.columns if col.startswith('w_')]  # Adjust if needed
    X = df[wavelength_cols]

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

        # Balance the dataset by undersampling negative samples
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        num_pos = len(pos_idx)
        if num_pos == 0:
            print(f"No positive samples for {substance} after check, skipping.")
            continue

        sampled_neg_idx = np.random.choice(neg_idx, size=min(num_pos, len(neg_idx)), replace=False)
        balanced_idx = np.concatenate([pos_idx, sampled_neg_idx])
        np.random.shuffle(balanced_idx)  # Shuffle to mix classes

        X_bal = X.iloc[balanced_idx]
        y_bal = y.iloc[balanced_idx]

        train_for_substance(X_bal, y_bal, substance, n_splits=5, n_estimators=200, learning_rate=0.1, max_depth=6, patience=10,
                            save_history_dir=save_history_dir, save_fig_dir=save_fig_dir,
                            save_best_model_dir=save_best_model_dir, save_scaler_dir=save_scaler_dir)