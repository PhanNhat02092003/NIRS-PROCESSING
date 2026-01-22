import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import json
import os
from model.regression_model import SMARTNIRRegressor, SmartNIRRegressionConfig
from dataset.regression_dataset import RegressionNIRSDataset

def train(model, train_loader, val_loader, device, epochs, criterion, optimizer, scheduler=None, patience=10,
          save_history_path="history/smart_nir_regression.json", save_fig_path="history/plot.png",
          save_best_model_path="checkpoint/checkpoint.pth"):
    best_loss = float('inf')
    best_model_wts = None
    early_stop_counter = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_mae": [],
        "val_rmse": [],
        "val_r2": []
    }

    for epoch in tqdm(range(1, epochs + 1)):
        # ----- Training -----
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)

            loss = criterion(outputs.squeeze(-1), y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # ----- Validation -----
        model.eval()
        val_running_loss = 0.0
        y_true, y_pred = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                outputs = model(X_batch)

                loss = criterion(outputs.squeeze(-1), y_batch)
                val_running_loss += loss.item() * X_batch.size(0)

                y_true.extend(y_batch.cpu().numpy())
                y_pred.extend(outputs.squeeze(-1).cpu().numpy())

        val_loss = val_running_loss / len(val_loader.dataset)
        val_mae = mean_absolute_error(y_true, y_pred)
        val_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        val_r2 = r2_score(y_true, y_pred)

        # Lưu vào history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)
        history["val_rmse"].append(val_rmse)
        history["val_r2"].append(val_r2)

        # Scheduler (nếu có)
        if scheduler is not None:
            scheduler.step(val_loss)

        # Cập nhật best model và kiểm tra early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = model.state_dict()
            torch.save(best_model_wts, save_best_model_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"MAE: {val_mae:.4f} | "
            f"RMSE: {val_rmse:.4f} | "
            f"R²: {val_r2:.4f}"
        )

    # load best model
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    with open(save_history_path, "w") as f:
        json.dump(history, f, indent=4)

    epochs_range = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(16, 12))

    # Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, history["train_loss"], label="Train Loss")
    plt.plot(epochs_range, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()

    # MAE
    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, history["val_mae"], label="MAE", color="g")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("Validation MAE")
    plt.legend()

    # RMSE
    plt.subplot(2, 2, 3)
    plt.plot(epochs_range, history["val_rmse"], label="RMSE", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("Validation RMSE")
    plt.legend()

    # R²
    plt.subplot(2, 2, 4)
    plt.plot(epochs_range, history["val_r2"], label="R²", color="purple")
    plt.xlabel("Epoch")
    plt.ylabel("R²")
    plt.title("Validation R²")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_fig_path)
    plt.close()

    return model, history


# ---------------- Run ----------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_epochs = 500
    patience = 50
    k_folds = 5
    machine = "OCEANFX"
    task = "substance_regression"

    full_path = f"data/merge/{machine}/ALL.csv"
    df = pd.read_csv(full_path)

    wavelength_cols = [col for col in df.columns if col.startswith('w_')]

    substances = [
        'Thiamethoxam', 'Permethrin', 'Metalaxyl', 'Azoxystrobin',
        'Imidaclopird', 'Difenoconazole', 'Cypermethrin', 'Cyhalothrin',
        'Chlorantraniliprol', 'Chlopyrifos Methyl', 'Emamectin benzoate',
        'Chlorothalonil', 'Triadimefon', 'Cyantraniliprole', 'Flutolanil',
        'Indoxacarb', 'Abamectin', 'Propamocarb.HCL', 'Chlothianidin'
    ]

    for substance in substances:
        print(f"Training for {substance}")
        if substance not in df.columns:
            print(f"Skipping {substance}: column not found")
            continue

        y_raw = df[substance]
        non_neg = y_raw[y_raw != -1]

        if len(non_neg) == 0 or len(non_neg.unique()) <= 1:
            print(f"Skipping {substance}: no valid targets or only one unique value")
            save_history_dir = f"history/{task}/stage2/{machine}/{substance}"
            save_fig_dir = f"history/{task}/stage2/{machine}/{substance}"
            save_best_model_dir = f"checkpoint/{task}/stage2/{machine}/{substance}"
            os.makedirs(save_history_dir, exist_ok=True)
            os.makedirs(save_fig_dir, exist_ok=True)
            os.makedirs(save_best_model_dir, exist_ok=True)
            continue

        try:
            full_ds = RegressionNIRSDataset(full_path, substance)
        except ValueError as e:
            print(f"Skipping {substance}: {e}")
            save_history_dir = f"history/{task}/stage2/{machine}/{substance}"
            save_fig_dir = f"history/{task}/stage2/{machine}/{substance}"
            save_best_model_dir = f"checkpoint/{task}/stage2/{machine}/{substance}"
            os.makedirs(save_history_dir, exist_ok=True)
            os.makedirs(save_fig_dir, exist_ok=True)
            os.makedirs(save_best_model_dir, exist_ok=True)
            continue

        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        fold_histories = []

        save_history_dir = f"history/{task}/stage2/{machine}/{substance}"
        save_fig_dir = f"history/{task}/stage2/{machine}/{substance}"
        save_best_model_dir = f"checkpoint/{task}/stage2/{machine}/{substance}"
        os.makedirs(save_history_dir, exist_ok=True)
        os.makedirs(save_fig_dir, exist_ok=True)
        os.makedirs(save_best_model_dir, exist_ok=True)

        for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(full_ds.y_raw)))):
            print(f"Fold {fold + 1}/{k_folds} for {substance}")

            save_fold_dir = f"data/{task}/stage2/{machine}/{substance}/fold_{fold + 1}"
            full_ds.fit_normalization(train_idx, save_dir=save_fold_dir)

            cfg = SmartNIRRegressionConfig(
                signal_len=full_ds.X_raw.shape[1],
                out_ch_per_branch=64,
                d_model=128,
                depth=3,
                n_heads=4,
                classifier="kan",
                num_targets=1,
                kan_basis=8
            )

            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)

            train_loader = DataLoader(full_ds, batch_size=512, sampler=train_sampler, num_workers=4)
            val_loader = DataLoader(full_ds, batch_size=512, sampler=val_sampler, num_workers=4)

            model = SMARTNIRRegressor(cfg).to(device)

            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5
            )

            save_history_path = f"{save_history_dir}/smart_nir_regression_fold{fold + 1}.json"
            save_fig_path = f"{save_fig_dir}/plot_fold{fold + 1}.png"
            save_best_model_path = f"{save_best_model_dir}/checkpoint_fold{fold + 1}.pth"

            model, history = train(
                model, train_loader, val_loader, device, max_epochs, criterion, optimizer,
                scheduler=scheduler, patience=patience,
                save_history_path=save_history_path, save_fig_path=save_fig_path,
                save_best_model_path=save_best_model_path
            )

            fold_histories.append(history)

        if fold_histories:
            avg_best_r2 = np.mean([max(h["val_r2"]) for h in fold_histories])
            print(f"Average best validation R² across folds for {substance}: {avg_best_r2:.4f}")