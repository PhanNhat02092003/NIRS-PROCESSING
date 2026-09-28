"""Trains XSpecMamba (model/xspecmamba_model.py) via the project's joint
multi-target training path (n_outputs=K, see
rapeseed_xspecmamba_joint_regression_engine.py) on the Mendeley Mango
DMC/NIR dataset (prepare_mango_dataset.py). Mango has only one target
(dry_matter), so n_outputs=1 here -- kept on the same joint code path as
Rapeseed/Grainit purely for a consistent official-benchmark methodology
(same plain KFold splitting, same train_joint() training loop) across all
three new datasets, per user request, even though with a single target
there is no actual multi-task sharing happening.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))  # repo root: model/, dataset/, shared engines
_ALL = os.path.join(_ROOT, "..", "all-dataset")
sys.path.insert(0, _ROOT)
os.chdir(_HERE)  # relative outputs (history/, checkpoint/, data/) land next to this script
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from tqdm import tqdm

from dataset.joint_regression_dataset import JointRegressionNIRSDataset
from model.xspecmamba_model import XSpecMamba, XSpecMambaConfig, spectrum_to_image


class JointSpectralImageDataset(Dataset):
    """Same precompute-once-per-fold strategy as SpectralImageDataset
    (regression.py), but wraps JointRegressionNIRSDataset
    (vector targets) instead of RegressionNIRSDataset (scalar target)."""

    def __init__(self, base_ds: JointRegressionNIRSDataset, img_size: int = 64):
        self.base_ds = base_ds
        self.img_size = img_size
        n = len(base_ds)
        self.images = np.empty((n, 3, img_size, img_size), dtype=np.float32)
        for i in range(n):
            spectrum, _ = base_ds[i]
            self.images[i] = spectrum_to_image(spectrum.numpy(), img_size=img_size)

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        _, target = self.base_ds[idx]
        return torch.from_numpy(self.images[idx]), target


def train_joint(model, train_loader, val_loader, device, epochs, criterion, optimizer,
                 target_names, inverse_transform_y, scheduler=None, patience=20, grad_clip_norm=1.0,
                 use_amp=False,
                 save_history_path="history/xspecmamba_joint.json", save_best_model_path="checkpoint/checkpoint.pth"):
    best_loss = float("inf")
    best_model_wts = None
    early_stop_counter = 0
    amp_enabled = use_amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history = {"train_loss": [], "val_loss": []}
    for name in target_names:
        history[f"val_mae_{name}"] = []
        history[f"val_rmse_{name}"] = []
        history[f"val_r2_{name}"] = []

    for epoch in tqdm(range(1, epochs + 1)):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                if amp_enabled:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * X_batch.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_running_loss = 0.0
        y_true, y_pred = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                val_running_loss += loss.item() * X_batch.size(0)
                y_true.append(y_batch.cpu().numpy())
                y_pred.append(outputs.cpu().numpy())
        val_loss = val_running_loss / len(val_loader.dataset)
        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)
        y_true_real = inverse_transform_y(y_true)
        y_pred_real = inverse_transform_y(y_pred)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        for i, name in enumerate(target_names):
            mae = float(mean_absolute_error(y_true_real[:, i], y_pred_real[:, i]))
            rmse = float(np.sqrt(mean_squared_error(y_true_real[:, i], y_pred_real[:, i])))
            r2 = float(r2_score(y_true_real[:, i], y_pred_real[:, i]))
            history[f"val_mae_{name}"].append(mae)
            history[f"val_rmse_{name}"].append(rmse)
            history[f"val_r2_{name}"].append(r2)

        if scheduler is not None:
            scheduler.step()

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

        r2_str = " | ".join(f"R2_{n}: {history[f'val_r2_{n}'][-1]:.4f}" for n in target_names)
        print(f"Epoch [{epoch}/{epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | {r2_str}")

    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    with open(save_history_path, "w") as f:
        json.dump(history, f, indent=4)

    return model, history


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_epochs = 200
    patience = 20
    k_folds = 5
    img_size = 64
    machine = "Mango"
    task = "substance_regression"
    target_names = ["dry_matter"]

    dataset_root = os.environ.get("MANGO_DATASET_ROOT", _ALL + "/Mendeley-Mango")
    full_path = f"{dataset_root}/{machine}/ALL.csv"

    save_history_dir = f"history/{task}/stage2_xspecmamba_joint/{machine}"
    save_best_model_dir = f"checkpoint/{task}/stage2_xspecmamba_joint/{machine}"
    save_data_dir = f"data/{task}/stage2_xspecmamba_joint/{machine}"
    os.makedirs(save_history_dir, exist_ok=True)
    os.makedirs(save_best_model_dir, exist_ok=True)

    # apply_savgol_snv=False: see rapeseed_xspecmamba_regression_engine.py --
    # the reference paper's NIR Gradient Enhancement module is designed to
    # learn SG/scatter-correction end-to-end, not receive it pre-applied.
    full_ds = JointRegressionNIRSDataset(full_path, target_names, apply_savgol_snv=False)
    print(f"n samples valid for both targets: {len(full_ds.y_raw)}")

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    folds = list(kf.split(np.arange(len(full_ds.y_raw))))

    fold_histories = []
    for fold, (train_idx, val_idx) in enumerate(folds):
        save_history_path = f"{save_history_dir}/xspecmamba_joint_fold{fold + 1}.json"
        if os.path.exists(save_history_path):
            print(f"Skipping fold {fold + 1}/{k_folds}: already completed")
            with open(save_history_path) as f:
                fold_histories.append(json.load(f))
            continue

        print(f"Fold {fold + 1}/{k_folds}")
        save_fold_dir = f"{save_data_dir}/fold_{fold + 1}"
        full_ds.fit_normalization(train_idx, save_dir=save_fold_dir, apply_zscore_X=False)
        img_ds = JointSpectralImageDataset(full_ds, img_size=img_size)

        cfg = XSpecMambaConfig(img_size=img_size, n_outputs=len(target_names), emb_dim=128, depth=1, n_dirs=4,
                                backbone="mamba")
        model = XSpecMamba(cfg).to(device)

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(img_ds, batch_size=64, sampler=train_sampler, num_workers=4, drop_last=True)
        val_loader = DataLoader(img_ds, batch_size=64, sampler=val_sampler, num_workers=4)

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=5e-4)
        warmup_epochs = max(1, int(0.05 * max_epochs))
        warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs - warmup_epochs)
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
        )

        save_best_model_path = f"{save_best_model_dir}/checkpoint_fold{fold + 1}.pth"
        model, history = train_joint(
            model, train_loader, val_loader, device, max_epochs, criterion, optimizer,
            target_names, full_ds.inverse_transform_y, scheduler=scheduler, patience=patience,
            grad_clip_norm=1.0, use_amp=True,
            save_history_path=save_history_path, save_best_model_path=save_best_model_path,
        )
        fold_histories.append(history)

    for name in target_names:
        best_r2s = [max(h[f"val_r2_{name}"]) for h in fold_histories]
        print(f"{name}: average best val_r2 across {len(fold_histories)} fold(s): {np.mean(best_r2s):.4f}")
