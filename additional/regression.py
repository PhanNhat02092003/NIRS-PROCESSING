"""Shared regression-baseline helpers (make_folds, train, train_for_substance,
SpectralImageDataset), reused by the EBAR/NirMACNet/XSpecMamba/SMART-NIR
Stage-2 regression engines in Grainit/Mango/Rapeseed. Restored from the
original repo-root regression.py (which also had its own main_stage2/
main_ebar/main_nirmacnet/main_xspecmamba entry points for the Danang
pesticide dataset, dropped along with the rest of that no-longer-deployed
regression task -- see main/food_classification.py's analogous trim) after
discovering additional/'s engines still imported from it.

Lives in additional/ (not main/) because every model it touches
(model.ebar_model, model.xspecmamba_model) is additional-only.
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from joblib import dump
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import Dataset
from tqdm import tqdm

from model.ebar_model import build_ebar_model
from model.xspecmamba_model import spectrum_to_image


def make_folds(y_raw, k_folds, seed=42):
    """Stratify folds on quantile bins of the target so each fold sees a
    similar concentration distribution, instead of plain KFold which can
    leave a fold skewed toward the low or high end of a right-skewed target.
    Falls back to plain KFold if there isn't enough data per bin to stratify.
    """
    try:
        bins = pd.qcut(y_raw, q=k_folds, labels=False, duplicates="drop")
        if len(np.unique(bins)) < 2:
            raise ValueError("not enough distinct quantile bins to stratify")
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
        return list(skf.split(np.zeros(len(y_raw)), bins))
    except ValueError:
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
        return list(kf.split(np.arange(len(y_raw))))


def train(model, train_loader, val_loader, device, epochs, criterion, optimizer, scheduler=None, patience=10,
          inverse_transform_y=None, accum_steps=1, grad_clip_norm=None, use_amp=False,
          save_history_path="history/smart_nir_regression.json", save_fig_path="history/plot.png",
          save_best_model_path="checkpoint/checkpoint.pth"):
    """accum_steps=1 (default) is a plain per-batch optimizer step, matching
    every existing caller's behavior unchanged. accum_steps>1 lets a caller
    use a smaller DataLoader batch_size (to fit GPU memory) while still
    reaching the same effective batch size for the optimizer step -- e.g.
    batch_size=128, accum_steps=4 behaves like batch_size=512 for gradients,
    at 1/4 the peak activation memory of a single forward/backward pass.
    BatchNorm layers still see the smaller per-microbatch statistics (not
    the effective batch), which is the standard, accepted tradeoff of
    gradient accumulation.

    grad_clip_norm=None (default) skips clipping entirely, matching every
    existing caller's behavior unchanged. A caller that wants max-norm
    gradient clipping (e.g. XSpecMamba, matching its reference paper's
    training recipe) passes a float max-norm value.

    use_amp=False (default) trains in plain fp32, matching every existing
    caller's behavior unchanged. A caller that wants mixed-precision
    training (e.g. XSpecMamba, matching its reference paper's recipe) can
    pass True; only meaningful with a CUDA device (silently behaves like
    False otherwise, since torch.amp.autocast has no effect on CPU tensors
    without a device-appropriate dtype).
    """
    best_loss = float('inf')
    best_model_wts = None
    early_stop_counter = 0
    amp_enabled = use_amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
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
        optimizer.zero_grad()
        n_batches = len(train_loader)
        for i, (X_batch, y_batch) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                outputs = model(X_batch)
                loss = criterion(outputs.squeeze(-1), y_batch)
            # scale down before backward so accumulated microbatch gradients
            # sum to the same magnitude as one effective-batch-size gradient
            scaler.scale(loss / accum_steps).backward()

            if (i + 1) % accum_steps == 0 or (i + 1) == n_batches:
                if grad_clip_norm is not None:
                    if amp_enabled:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * X_batch.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # ----- Validation -----
        model.eval()
        val_running_loss = 0.0
        y_true, y_pred = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    outputs = model(X_batch)
                    loss = criterion(outputs.squeeze(-1), y_batch)
                val_running_loss += loss.item() * X_batch.size(0)

                y_true.extend(y_batch.cpu().numpy())
                y_pred.extend(outputs.squeeze(-1).cpu().numpy())

        val_loss = val_running_loss / len(val_loader.dataset)

        # report MAE/RMSE/R2 in original concentration units, not the
        # z-scored log space the model is actually trained/selected on
        if inverse_transform_y is not None:
            y_true_report = inverse_transform_y(np.array(y_true))
            y_pred_report = inverse_transform_y(np.array(y_pred))
        else:
            y_true_report, y_pred_report = y_true, y_pred

        # cast to plain float: sklearn returns numpy.float32 for float32
        # input, which json.dump can't serialize (unlike numpy.float64)
        val_mae = float(mean_absolute_error(y_true_report, y_pred_report))
        val_rmse = float(np.sqrt(mean_squared_error(y_true_report, y_pred_report)))
        val_r2 = float(r2_score(y_true_report, y_pred_report))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)
        history["val_rmse"].append(val_rmse)
        history["val_r2"].append(val_r2)

        if scheduler is not None:
            # ReduceLROnPlateau needs the metric it's tracking; other
            # schedulers (CosineAnnealingLR, SequentialLR, ...) step on
            # their own internal epoch counter and take no argument --
            # calling .step(val_loss) on those silently misuses the
            # deprecated positional "epoch" arg with a loss value instead.
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
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

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"MAE: {val_mae:.4f} | "
            f"RMSE: {val_rmse:.4f} | "
            f"R²: {val_r2:.4f}"
        )

    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    with open(save_history_path, "w") as f:
        json.dump(history, f, indent=4)

    epochs_range = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(16, 12))

    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, history["train_loss"], label="Train Loss")
    plt.plot(epochs_range, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, history["val_mae"], label="MAE", color="g")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("Validation MAE")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(epochs_range, history["val_rmse"], label="RMSE", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("Validation RMSE")
    plt.legend()

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


# LassoCV/ElasticNetCV probe a range of regularization strengths internally;
# the weakest ones are near-unregularized least-squares on highly collinear
# NIR wavelengths, which coordinate descent converges to slowly/imperfectly.
# Benign (the CV-selected alpha is rarely one of these), just noisy.
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def fit_xy_scaler(X_train, y_train):
    """Z-score for X; log1p + z-score for y.

    The paper's own preprocessing is plain standardization (no log1p), but
    several substances can be far more right-skewed than the paper's
    dataset. log1p first (the same transform RegressionNIRSDataset/SMART-NIR/
    every other baseline here already uses) keeps this a fair apples-to-apples
    comparison instead of penalizing EBAR for a data-scaling artifact
    unrelated to the stacking architecture itself.
    """
    mean_X = X_train.mean(axis=0, keepdims=True)
    std_X = X_train.std(axis=0, keepdims=True) + 1e-8
    y_train_log = np.log1p(y_train)
    mean_y = y_train_log.mean()
    std_y = y_train_log.std() + 1e-8
    return mean_X, std_X, mean_y, std_y


def train_for_substance(X_raw, y_raw, substance_name, folds,
                         save_history_dir, save_checkpoint_dir):
    fold_histories = []

    for fold, (train_idx, val_idx) in enumerate(folds):
        save_history_path = os.path.join(save_history_dir, f"{substance_name}_fold_{fold + 1}.json")

        # Resume support: a completed fold's history file is only ever
        # written after that fold's fit+eval fully succeeds, so its
        # presence is a reliable "already done" marker.
        if os.path.exists(save_history_path):
            print(f"Skipping fold {fold + 1}/{len(folds)} for {substance_name}: already completed")
            with open(save_history_path) as f:
                fold_histories.append(json.load(f))
            continue

        print(f"Fold {fold + 1}/{len(folds)} for {substance_name}")

        X_train_raw, X_val_raw = X_raw[train_idx], X_raw[val_idx]
        y_train_raw, y_val_raw = y_raw[train_idx], y_raw[val_idx]

        mean_X, std_X, mean_y, std_y = fit_xy_scaler(X_train_raw, y_train_raw)
        X_train = (X_train_raw - mean_X) / std_X
        X_val = (X_val_raw - mean_X) / std_X
        y_train = (np.log1p(y_train_raw) - mean_y) / std_y

        model = build_ebar_model(n_samples=len(train_idx), n_features=X_raw.shape[1])
        model.fit(X_train, y_train)

        y_pred_scaled = model.predict(X_val)
        y_pred = np.expm1(y_pred_scaled * std_y + mean_y)  # undo z-score + log1p

        val_mae = float(mean_absolute_error(y_val_raw, y_pred))
        val_rmse = float(np.sqrt(mean_squared_error(y_val_raw, y_pred)))
        val_r2 = float(r2_score(y_val_raw, y_pred))

        history = {"val_mae": [val_mae], "val_rmse": [val_rmse], "val_r2": [val_r2]}

        os.makedirs(save_history_dir, exist_ok=True)
        with open(save_history_path, "w") as f:
            json.dump(history, f, indent=4)

        save_model_path = os.path.join(save_checkpoint_dir, f"{substance_name}_fold_{fold + 1}.joblib")
        os.makedirs(save_checkpoint_dir, exist_ok=True)
        dump({"model": model, "mean_X": mean_X, "std_X": std_X, "mean_y": mean_y, "std_y": std_y}, save_model_path)

        fold_histories.append(history)

    avg_r2 = np.mean([h["val_r2"][0] for h in fold_histories])
    avg_rmse = np.mean([h["val_rmse"][0] for h in fold_histories])
    print(f"Average validation R2 across folds for {substance_name}: {avg_r2:.4f} (RMSE: {avg_rmse:.4f})")


class SpectralImageDataset(Dataset):
    """Wraps an already-fitted RegressionNIRSDataset, converting each
    normalized 1D spectrum to a (3, img_size, img_size) image.
    Wrapping (not subclassing) keeps RegressionNIRSDataset's own
    normalization/fold-fitting untouched, reused identically to SMART-NIR.

    Images are precomputed once in __init__ rather than lazily per
    __getitem__: spectrum_to_image depends only on the (already-fitted,
    fixed-for-this-fold) input spectrum, not on model state, so recomputing
    it on every epoch -- as a naive lazy __getitem__ would -- is pure
    redundant work, up to `epochs`-fold more of the O(N^2) full-resolution
    GAF/RP/Corr cost than necessary. Precomputing trades one upfront pass
    per fold for a flat per-epoch cost, same as the model's own forward
    pass.
    """

    def __init__(self, base_ds, img_size: int = 64):
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
