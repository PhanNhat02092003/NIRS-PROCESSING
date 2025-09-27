import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset.regression_dataset import RegressionNIRSDataset
from model.regression_model import *
import matplotlib.pyplot as plt
import json

cfg = SmartNIRRegressorConfig(
    signal_len=2136,
    out_ch_per_branch=128,
    d_model=256,
    depth=6,
    n_heads=8,
    classifier="kan",   # hoặc "mlp"
    num_targets=21
)

def train(model, train_loader, val_loader, device, epochs, criterion, optimizer, scheduler=None, 
          save_history_path="history/smart_nir_regression.json", 
          save_fig_path="history/plot_regression.png", 
          save_best_model_path="pretrained/smart_nir_regression_1.pth"):
    
    best_val_loss = float("inf")
    history = {
        "train_loss": [], 
        "val_loss": [], 
        "val_mse": [],
        "val_mae": [],
        "val_r2": []
    }

    for epoch in tqdm(range(1, epochs+1)):
        # ----- Training -----
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
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
                loss = criterion(outputs, y_batch)
                val_running_loss += loss.item() * X_batch.size(0)

                y_true.append(y_batch.cpu().numpy())
                y_pred.append(outputs.cpu().numpy())

        val_loss = val_running_loss / len(val_loader.dataset)

        # convert list to numpy
        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)

        # regression metrics
        val_mse = mean_squared_error(y_true, y_pred)
        val_mae = mean_absolute_error(y_true, y_pred)
        val_r2 = r2_score(y_true, y_pred, multioutput="uniform_average")

        # save history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mse"].append(val_mse)
        history["val_mae"].append(val_mae)
        history["val_r2"].append(val_r2)

        # Scheduler
        if scheduler is not None:
            scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = model.state_dict()
            torch.save(best_model_wts, save_best_model_path)

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"MSE: {val_mse:.4f} | "
            f"MAE: {val_mae:.4f} | "
            f"R²: {val_r2:.4f}"
        )

    # load best model
    model.load_state_dict(best_model_wts)

    with open(save_history_path, "w") as f:
        json.dump(history, f, indent=4)

    # ---- Plot ----
    epochs_range = range(1, epochs+1)

    plt.figure(figsize=(20, 12))

    # Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, history["train_loss"], label="Train Loss")
    plt.plot(epochs_range, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()

    # MSE
    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, history["val_mse"], label="MSE", color="g")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Validation MSE")
    plt.legend()

    # MAE
    plt.subplot(2, 2, 3)
    plt.plot(epochs_range, history["val_mae"], label="MAE", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("Validation MAE")
    plt.legend()

    # R²
    plt.subplot(2, 2, 4)
    plt.plot(epochs_range, history["val_r2"], label="R²", color="purple")
    plt.xlabel("Epoch")
    plt.ylabel("R²")
    plt.title("Validation R² Score")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_fig_path)
    plt.show()

    return model, history


# ---------------- Run ----------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 200
    train_ds = RegressionNIRSDataset("data/regression_data/dataset_1/train", mode="train")
    val_ds = RegressionNIRSDataset("data/regression_data/dataset_1/val", mode="val")

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=4)

    model = SMARTNIRRegressor(cfg).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, 
    )

    model, history = train(model, train_loader, val_loader, device, epochs, criterion, optimizer, scheduler)
