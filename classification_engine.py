import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import json
from model.classification_model import SMARTNIRClassifier, SmartNIRClassificationConfig
from dataset.classification_dataset import ClassificationNIRSDataset
import os

def train(model, train_loader, val_loader, device, epochs, criterion, optimizer, scheduler=None, patience=10, save_history_path="history/smart_nir_classification.json", save_fig_path="history/plot.png", save_best_model_path="checkpoint/checkpoint.pth"):
    best_acc = 0.0
    best_model_wts = None
    early_stop_counter = 0
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

        # Cập nhật best model và kiểm tra early stopping
        if val_acc > best_acc:
            best_acc = val_acc
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
            f"Acc: {val_acc:.4f} | "
            f"Precision: {val_precision:.4f} | "
            f"Recall: {val_recall:.4f} | "
            f"F1: {val_f1:.4f}"
        )

    # load best model
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    with open(save_history_path, "w") as f:
        json.dump(history, f, indent=4)

    epochs_range = range(1, len(history["train_loss"]) + 1)

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
    plt.savefig(save_fig_path)
    plt.close() 

    return model, history


# ---------------- Run ----------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_epochs = 200
    patience = 10
    k_folds = 5
    machine = "FLAMENIR"
    task = "category_classification"

    full_path = f"data/merge/{machine}/ALL.csv"
    full_ds = ClassificationNIRSDataset(full_path)

    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_histories = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(full_ds.y_raw)), full_ds.y_raw)):
        print(f"Fold {fold + 1}/{k_folds}")

        save_fold_dir = f"data/{task}/{machine}/fold_{fold + 1}"
        full_ds.fit_normalization_and_labels(train_idx, save_dir=save_fold_dir)

        cfg = SmartNIRClassificationConfig(
            signal_len=full_ds.signal_length,
            out_ch_per_branch=64,
            d_model=128,
            depth=3,
            n_heads=4,
            classifier="kan",   
            num_classes=full_ds.n_classes
        )

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(full_ds, batch_size=512, sampler=train_sampler, num_workers=4)
        val_loader = DataLoader(full_ds, batch_size=512, sampler=val_sampler, num_workers=4)

        model = SMARTNIRClassifier(cfg).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        save_history_path = f"history/{task}/{machine}/smart_nir_classification_fold{fold + 1}.json"
        save_fig_path = f"history/{task}/{machine}/plot_fold{fold + 1}.png"
        os.makedirs(f"history/{task}/{machine}", exist_ok=True)
        save_best_model_path = f"checkpoint/{task}/{machine}/checkpoint_fold{fold + 1}.pth"
        os.makedirs(f"checkpoint/{task}/{machine}", exist_ok=True)

        model, history = train(
            model, train_loader, val_loader, device, max_epochs, criterion, optimizer,
            scheduler=scheduler, patience=patience,
            save_history_path=save_history_path, save_fig_path=save_fig_path,
            save_best_model_path=save_best_model_path
        )

        fold_histories.append(history)

    avg_val_acc = np.mean([max(h["val_acc"]) for h in fold_histories])
    print(f"Average best validation accuracy across folds: {avg_val_acc:.4f}")