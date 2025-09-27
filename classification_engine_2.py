import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset.classification_dataset import ClassificationNIRSDataset
from model.classification_model import *
import matplotlib.pyplot as plt
import json

cfg = SmartNIRClassificationConfig(
    signal_len=128,
    out_ch_per_branch=64,
    d_model=256,
    depth=6,
    n_heads=8,
    classifier="kan",   # đổi "mlp" nếu muốn MLP
    num_classes=9
)

def train(model, train_loader, val_loader, device, epochs, criterion, optimizer, scheduler=None, save_history_path="history/smart_nir_classification_2.json", save_fig_path="history/plot_2.png", save_best_model_path="pretrained/smart_nir_classification_2.pth"):
    best_acc = 0.0
    history = {
        "train_loss": [], 
        "val_loss": [], 
        "val_acc": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": []
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

                preds = torch.argmax(outputs, dim=1)
                y_true.extend(y_batch.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = accuracy_score(y_true, y_pred)
        val_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        val_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        val_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        # Lưu vào history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_precision"].append(val_precision)
        history["val_recall"].append(val_recall)
        history["val_f1"].append(val_f1)

        # Scheduler (nếu có)
        if scheduler is not None:
            scheduler.step(val_loss)

        # Cập nhật best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict()
            torch.save(best_model_wts, save_best_model_path)

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Acc: {val_acc:.4f} | "
            f"Precision: {val_precision:.4f} | "
            f"Recall: {val_recall:.4f} | "
            f"F1: {val_f1:.4f}"
        )

    # load best model
    model.load_state_dict(best_model_wts)

    with open(save_history_path, "w") as f:
        json.dump(history, f, indent=4)

    epochs_range = range(1, epochs+1)

    plt.figure(figsize=(20, 16))

    # Loss
    plt.subplot(2, 3, 1)
    plt.plot(epochs_range, history["train_loss"], label="Train Loss")
    plt.plot(epochs_range, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()

    # Accuracy
    plt.subplot(2, 3, 2)
    plt.plot(epochs_range, history["val_acc"], label="Accuracy", color="g")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()

    # Precision
    plt.subplot(2, 3, 3)
    plt.plot(epochs_range, history["val_precision"], label="Precision", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.title("Validation Precision")
    plt.legend()

    # Recall
    plt.subplot(2, 3, 4)
    plt.plot(epochs_range, history["val_recall"], label="Recall", color="purple")
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.title("Validation Recall")
    plt.legend()

    # F1
    plt.subplot(2, 3, 5)
    plt.plot(epochs_range, history["val_f1"], label="F1-score", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.title("Validation F1-score")
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.savefig(save_fig_path)

    return model, history


# ---------------- Run ----------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 300
    train_ds = ClassificationNIRSDataset("data/classification_data/dataset_2/train", mode="train")
    val_ds = ClassificationNIRSDataset("data/classification_data/dataset_2/val", mode="val")

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=4)

    model = SMARTNIR(cfg).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, 
    )

    model, history = train(model, train_loader, val_loader, device, epochs, criterion, optimizer, scheduler=None) 