"""Trains XSpecMamba (model/xspecmamba_model.py) in its *joint* multi-target
mode (n_outputs=6, one shared backbone for all 6 pesticide substances) on
DanangLeafy6 -- intended to mirror Grainit/grainit_xspecmamba_joint_
regression_engine.py ("the official XSpecMamba benchmark number for this
project's multi-target datasets instead of the per-substance-independent
convention"), but with ONE necessary deviation:

Grainit/Rapeseed's joint engine requires a row to have EVERY target present
(dataset/joint_regression_dataset.py's JointRegressionNIRSDataset) -- fine
for co-measured properties like Moisture+Protein. DanangLeafy6's 6
substances are each present in only a minority of samples and are
essentially never all simultaneously present: 0/8000 rows have all 6
present, and at most 4 ever co-occur (in just 25 rows) -- see
dataset/masked_joint_regression_dataset.py's docstring. Requiring all 6
would leave 0 training rows, so this engine uses
MaskedJointRegressionNIRSDataset instead (keeps a row if ANY target is
present; missing targets are NaN) and a masked MSE loss / masked per-target
metrics (a missing target contributes 0 to the loss and is excluded from
that target's MAE/RMSE/R2 for that row) -- same missing-value convention
additional/LensNIR/dataset_benchmark.py uses for its own joint head.

Group-safe: folds split via plain GroupKFold on `_group_id` (not quantile-
stratified -- a 6-column multi-target doesn't have one scalar to stratify
on; this is a judgment call, flagged for review) so one scanned batch never
straddles train/val, unlike Grainit which has no such grouping.

Env: MACHINE (FLAMENIR | OCEANFX), BIN (OCEANFX: average wavelengths in
groups of 8, matching every other model on this machine).

Usage: MACHINE=FLAMENIR python3 danangleafy6_xspecmamba_joint_regression_engine.py
       MACHINE=OCEANFX BIN=8 python3 danangleafy6_xspecmamba_joint_regression_engine.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))  # repo root: model/, dataset/, shared engines
sys.path.insert(0, os.path.join(_ROOT, "main"))
sys.path.insert(0, os.path.dirname(_HERE))  # additional/: lets Python merge additional/dataset/ into the dataset namespace package
os.chdir(_HERE)  # relative outputs (history/, checkpoint/, data/) land next to this script
import json

import numpy as np
import torch
import torch.optim as optim
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from tqdm import tqdm

from dataset.masked_joint_regression_dataset import MaskedJointRegressionNIRSDataset
from model.xspecmamba_model import XSpecMamba, XSpecMambaConfig, spectrum_to_image

load_dotenv()

SUBSTANCES = ['Thiamethoxam', 'Permethrin', 'Azoxystrobin',
              'Difenoconazole', 'Cypermethrin', 'Chlothianidin']


def bin_wavelengths(X: np.ndarray, bin_factor: int) -> np.ndarray:
    if bin_factor <= 1:
        return X
    n_out = (X.shape[1] // bin_factor) // 8 * 8
    return X[:, :n_out * bin_factor].reshape(len(X), n_out, bin_factor).mean(axis=2).astype(np.float32)


class JointSpectralImageDataset(Dataset):
    """Same precompute-once-per-fold strategy as SpectralImageDataset
    (regression.py)/JointSpectralImageDataset (Grainit's joint engine), but
    also passes the per-target mask through unchanged."""

    def __init__(self, base_ds: MaskedJointRegressionNIRSDataset, img_size: int = 64):
        self.base_ds = base_ds
        self.img_size = img_size
        n = len(base_ds)
        self.images = np.empty((n, 3, img_size, img_size), dtype=np.float32)
        for i in range(n):
            spectrum, _, _ = base_ds[i]
            self.images[i] = spectrum_to_image(spectrum.numpy(), img_size=img_size)

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        _, target, mask = self.base_ds[idx]
        return torch.from_numpy(self.images[idx]), target, mask


def train_joint_masked(model, train_loader, val_loader, device, epochs, optimizer,
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
        # log1p-space counterparts: pesticide concentrations are heavy-tailed
        # (rare large outliers dominate SS_tot/SS_res), so raw-scale R2 can be
        # catastrophically negative even for a well-fit model -- these report
        # fit quality without being dominated by outliers (see regression.py's
        # train() docstring for the same rationale).
        history[f"val_mae_log_{name}"] = []
        history[f"val_rmse_log_{name}"] = []
        history[f"val_r2_log_{name}"] = []

    for epoch in tqdm(range(1, epochs + 1)):
        model.train()
        running_loss, running_mask = 0.0, 0.0
        for X_batch, y_batch, mask_batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            X_batch, y_batch, mask_batch = X_batch.to(device), y_batch.to(device), mask_batch.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                outputs = model(X_batch)
                sq_err = (outputs - y_batch) ** 2 * mask_batch
                loss = sq_err.sum() / mask_batch.sum().clamp(min=1.0)
            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                if amp_enabled:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * mask_batch.sum().item()
            running_mask += mask_batch.sum().item()
        train_loss = running_loss / max(running_mask, 1.0)

        model.eval()
        val_running_loss, val_running_mask = 0.0, 0.0
        y_true, y_pred, masks = [], [], []
        with torch.no_grad():
            for X_batch, y_batch, mask_batch in val_loader:
                X_batch, y_batch, mask_batch = X_batch.to(device), y_batch.to(device), mask_batch.to(device)
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    outputs = model(X_batch)
                    sq_err = (outputs - y_batch) ** 2 * mask_batch
                    loss = sq_err.sum() / mask_batch.sum().clamp(min=1.0)
                val_running_loss += loss.item() * mask_batch.sum().item()
                val_running_mask += mask_batch.sum().item()
                y_true.append(y_batch.cpu().numpy())
                y_pred.append(outputs.cpu().numpy())
                masks.append(mask_batch.cpu().numpy())
        val_loss = val_running_loss / max(val_running_mask, 1.0)
        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)
        masks = np.concatenate(masks, axis=0).astype(bool)
        y_true_real = inverse_transform_y(y_true)
        y_pred_real = inverse_transform_y(y_pred)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        for i, name in enumerate(target_names):
            m = masks[:, i]
            if m.sum() < 2:
                # too few present-target rows in this fold's val split to
                # score (can happen for the rarer substances) -- repeat the
                # previous epoch's value rather than inject a NaN/0 that
                # would corrupt argmax/argmin selection downstream.
                prev = history[f"val_r2_{name}"][-1] if history[f"val_r2_{name}"] else 0.0
                history[f"val_mae_{name}"].append(history[f"val_mae_{name}"][-1] if history[f"val_mae_{name}"] else 0.0)
                history[f"val_rmse_{name}"].append(history[f"val_rmse_{name}"][-1] if history[f"val_rmse_{name}"] else 0.0)
                history[f"val_r2_{name}"].append(prev)
                prev_log = history[f"val_r2_log_{name}"][-1] if history[f"val_r2_log_{name}"] else 0.0
                history[f"val_mae_log_{name}"].append(history[f"val_mae_log_{name}"][-1] if history[f"val_mae_log_{name}"] else 0.0)
                history[f"val_rmse_log_{name}"].append(history[f"val_rmse_log_{name}"][-1] if history[f"val_rmse_log_{name}"] else 0.0)
                history[f"val_r2_log_{name}"].append(prev_log)
                continue
            mae = float(mean_absolute_error(y_true_real[m, i], y_pred_real[m, i]))
            rmse = float(np.sqrt(mean_squared_error(y_true_real[m, i], y_pred_real[m, i])))
            r2 = float(r2_score(y_true_real[m, i], y_pred_real[m, i]))
            history[f"val_mae_{name}"].append(mae)
            history[f"val_rmse_{name}"].append(rmse)
            history[f"val_r2_{name}"].append(r2)

            y_true_log = np.log1p(y_true_real[m, i].astype(np.float64))
            y_pred_log = np.log1p(y_pred_real[m, i].astype(np.float64))
            history[f"val_mae_log_{name}"].append(float(mean_absolute_error(y_true_log, y_pred_log)))
            history[f"val_rmse_log_{name}"].append(float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))))
            history[f"val_r2_log_{name}"].append(float(r2_score(y_true_log, y_pred_log)))

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
    machine = os.environ["MACHINE"]
    bin_factor = int(os.environ.get("BIN", 1))
    max_epochs = 200
    patience = 20
    k_folds = 5
    img_size = 64
    task = "substance_regression"

    full_path = f"{_HERE}/data/{machine}/ALL_synthetic.csv"
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"{full_path} not found -- run prepare_danangleafy6_dataset.py first")

    save_history_dir = f"history/{task}/stage2_xspecmamba_joint/{machine}"
    save_best_model_dir = f"checkpoint/{task}/stage2_xspecmamba_joint/{machine}"
    save_data_dir = f"data/{task}/stage2_xspecmamba_joint/{machine}"
    os.makedirs(save_history_dir, exist_ok=True)
    os.makedirs(save_best_model_dir, exist_ok=True)

    # apply_savgol_snv=False: see rapeseed_xspecmamba_regression_engine.py --
    # the reference paper's NIR Gradient Enhancement module is designed to
    # learn SG/scatter-correction end-to-end, not receive it pre-applied.
    full_ds = MaskedJointRegressionNIRSDataset(full_path, SUBSTANCES, apply_savgol_snv=False, group_col="_group_id")
    full_ds.X_raw = bin_wavelengths(full_ds.X_raw, bin_factor)
    print(f"n samples valid for >=1 target: {len(full_ds.y_raw)}")

    # Plain GroupKFold (not quantile-stratified like make_group_folds): with
    # 6 sparse targets there's no single scalar to stratify on -- judgment
    # call, flagged for review.
    gkf = GroupKFold(n_splits=k_folds)
    folds = list(gkf.split(np.arange(len(full_ds.y_raw)), groups=full_ds.group_raw))

    fold_histories = []
    for fold, (train_idx, val_idx) in enumerate(folds):
        save_history_path = f"{save_history_dir}/xspecmamba_joint_fold{fold + 1}.json"
        if os.path.exists(save_history_path):
            print(f"Skipping fold {fold + 1}/{k_folds}: already completed")
            with open(save_history_path) as f:
                fold_histories.append(json.load(f))
            continue

        overlap = set(full_ds.group_raw[train_idx]) & set(full_ds.group_raw[val_idx])
        assert not overlap, f"fold {fold + 1}: {len(overlap)} groups leak across train/val"

        print(f"Fold {fold + 1}/{k_folds}")
        save_fold_dir = f"{save_data_dir}/fold_{fold + 1}"
        full_ds.fit_normalization(train_idx, save_dir=save_fold_dir, apply_zscore_X=False)
        img_ds = JointSpectralImageDataset(full_ds, img_size=img_size)

        cfg = XSpecMambaConfig(img_size=img_size, n_outputs=len(SUBSTANCES), emb_dim=128, depth=1, n_dirs=4,
                                backbone="mamba")
        model = XSpecMamba(cfg).to(device)

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(img_ds, batch_size=64, sampler=train_sampler, num_workers=4, drop_last=True)
        val_loader = DataLoader(img_ds, batch_size=64, sampler=val_sampler, num_workers=4)

        optimizer = optim.AdamW(model.parameters(), lr=5e-4)
        warmup_epochs = max(1, int(0.05 * max_epochs))
        warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs - warmup_epochs)
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
        )

        save_best_model_path = f"{save_best_model_dir}/checkpoint_fold{fold + 1}.pth"
        model, history = train_joint_masked(
            model, train_loader, val_loader, device, max_epochs, optimizer,
            SUBSTANCES, full_ds.inverse_transform_y, scheduler=scheduler, patience=patience,
            grad_clip_norm=1.0, use_amp=True,
            save_history_path=save_history_path, save_best_model_path=save_best_model_path,
        )
        fold_histories.append(history)

    for name in SUBSTANCES:
        best_r2s = [max(h[f"val_r2_{name}"]) for h in fold_histories]
        print(f"{name}: average best val_r2 across {len(fold_histories)} fold(s): {np.mean(best_r2s):.4f}")
