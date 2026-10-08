"""Joint multi-target XSpecMamba (model/xspecmamba_model.py) on the Danang
pesticide dataset, restricted to an explicit substance subset instead of
the per-substance-independent convention regression.py's METHOD=xspecmamba
normally uses (one model per substance). One shared backbone predicts all
N_SUBSTANCES concentrations at once via XSpecMamba's existing
TargetSpecificAttentionPooling + per-output head design (n_outputs=K) --
no model changes needed, XSpecMamba was already built for this.

Unlike dataset/joint_regression_dataset.py's JointRegressionNIRSDataset
(used by the Mango/Grainit/Rapeseed "*_xspecmamba_joint_regression_engine.py"
scripts), this does NOT require every target to be valid on the same row:
checked directly against the real data, 0 rows on either machine have all
of Thiamethoxam/Permethrin/Azoxystrobin/Difenoconazole/Cypermethrin/
Chlothianidin present at once (pesticide presence is sparse and
non-overlapping, not a missing-reading situation like Grainit/Rapeseed's
few sensor-flagged rows) -- "require all present" would leave zero training
rows. dataset/pesticide_multitask_dataset.py keeps every row and masks each
substance's loss/metrics by its own presence column instead, the same way
../MT-SMART-NIR's MaskedRegressionLoss handles concentration for its
detection-gated regression head.

Usage:
    MACHINE=FLAMENIR python3 xspecmamba_joint_pesticide_engine.py
    MACHINE=OCEANFX BIN=8 python3 xspecmamba_joint_pesticide_engine.py
    SUBSTANCES=A,B,C MACHINE=FLAMENIR python3 xspecmamba_joint_pesticide_engine.py
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from tqdm import tqdm

from dataset.pesticide_multitask_dataset import PesticideMultiTaskDataset
from model.xspecmamba_model import XSpecMamba, XSpecMambaConfig, spectrum_to_image

load_dotenv()

DEFAULT_SUBSTANCES = [
    "Thiamethoxam", "Permethrin", "Azoxystrobin",
    "Difenoconazole", "Cypermethrin", "Chlothianidin",
]


class PesticideImageDataset(Dataset):
    """Same precompute-once-per-fold strategy as regression.py's
    SpectralImageDataset, wrapping PesticideMultiTaskDataset (presence +
    log1p-concentration vector targets) instead of a single scalar target."""

    def __init__(self, base_ds: PesticideMultiTaskDataset, img_size: int = 64):
        self.base_ds = base_ds
        self.img_size = img_size
        n = len(base_ds)
        self.images = np.empty((n, 3, img_size, img_size), dtype=np.float32)
        for i in range(n):
            spectrum, _, _, _ = base_ds[i]
            self.images[i] = spectrum_to_image(spectrum.numpy(), img_size=img_size)

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        _, _, pres, conc_log = self.base_ds[idx]
        return torch.from_numpy(self.images[idx]), pres, conc_log


def masked_huber(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                  delta: float = 0.5, alpha: float = 0.1) -> torch.Tensor:
    """Two-term masked Huber (ported from ../MT-SMART-NIR's
    MaskedRegressionLoss): fit present entries to their true value, and
    lightly suppress absent entries toward 0 (the correct log1p(0) target)
    so the regressor doesn't drift on substances it never needs to predict
    for most rows."""
    mask_b = mask.bool()
    if mask_b.any():
        l_present = F.huber_loss(pred[mask_b], target[mask_b], delta=delta, reduction="mean")
    else:
        l_present = pred.sum() * 0.0
    absent = ~mask_b
    if absent.any() and alpha > 0:
        l_absent = F.huber_loss(pred[absent], torch.zeros_like(pred[absent]), delta=delta, reduction="mean")
    else:
        l_absent = pred.sum() * 0.0
    return l_present + alpha * l_absent


def evaluate(model, loader, device, n_substances):
    model.eval()
    pred_all, true_all, mask_all = [], [], []
    with torch.no_grad():
        for X_b, pres_b, conc_b in loader:
            X_b = X_b.to(device)
            out = model(X_b)
            pred_all.append(out.cpu().numpy())
            true_all.append(conc_b.numpy())
            mask_all.append(pres_b.numpy())
    pred_all = np.concatenate(pred_all, axis=0)
    true_all = np.concatenate(true_all, axis=0)
    mask_all = np.concatenate(mask_all, axis=0) == 1

    pred_mgkg = np.expm1(pred_all)
    true_mgkg = np.expm1(true_all)

    maes, r2s = [], []
    for j in range(n_substances):
        m = mask_all[:, j]
        if m.sum() >= 2:
            maes.append(float(mean_absolute_error(true_mgkg[m, j], pred_mgkg[m, j])))
            r2s.append(float(r2_score(true_mgkg[m, j], pred_mgkg[m, j])))
        else:
            maes.append(float("nan"))
            r2s.append(float("nan"))
    return maes, r2s


def train_fold(model, train_loader, val_loader, device, max_epochs, optimizer, scheduler,
               substances, patience, save_history_path, save_best_model_path,
               grad_clip_norm=1.0, use_amp=True):
    n_sub = len(substances)
    amp_enabled = use_amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    history = {"train_loss": [], "val_loss": []}
    for name in substances:
        history[f"val_mae_{name}"] = []
        history[f"val_r2_{name}"] = []

    best_score, best_state, bad_epochs = -1e9, None, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        running = 0.0
        for X_b, pres_b, conc_b in tqdm(train_loader, desc=f"Epoch {epoch}/{max_epochs}", leave=False):
            X_b, pres_b, conc_b = X_b.to(device), pres_b.to(device), conc_b.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                out = model(X_b)
                loss = masked_huber(out, conc_b, pres_b)
            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                if amp_enabled:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * X_b.size(0)
        train_loss = running / len(train_loader.dataset)

        maes, r2s = evaluate(model, val_loader, device, n_sub)
        val_loss = float(np.nanmean(maes))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        for j, name in enumerate(substances):
            history[f"val_mae_{name}"].append(maes[j])
            history[f"val_r2_{name}"].append(r2s[j])

        if scheduler is not None:
            scheduler.step()

        score = float(np.nanmean([max(0.0, r) for r in r2s]))
        r2_str = " | ".join(f"R2_{n}: {r2s[j]:.3f}" for j, n in enumerate(substances))
        print(f"Epoch [{epoch}/{max_epochs}] Train Loss: {train_loss:.4f} | "
              f"mean_mae: {val_loss:.4f} | {r2_str} | score: {score:.4f}", flush=True)

        if score > best_score:
            best_score, bad_epochs = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, save_best_model_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    os.makedirs(os.path.dirname(save_history_path), exist_ok=True)
    with open(save_history_path, "w") as f:
        json.dump(history, f, indent=4)
    return model, history


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_epochs = 200
    patience = 20
    k_folds = 5
    img_size = 64
    machine = os.environ["MACHINE"]
    task = "pesticide_regression_xspecmamba_joint"

    substances = DEFAULT_SUBSTANCES
    if os.environ.get("SUBSTANCES"):
        substances = [s.strip() for s in os.environ["SUBSTANCES"].split(",")]

    full_path = f"{os.environ['DATASET_ROOT']}/{machine}/ALL.csv"
    bin_factor = int(os.environ.get("BIN", 1))
    sample_n = int(os.environ["SAMPLE_N"]) if os.environ.get("SAMPLE_N") else None
    full_ds = PesticideMultiTaskDataset(full_path, substances, bin_factor=bin_factor, sample_n=sample_n)
    print(f"n samples: {len(full_ds.food_raw)}, substances: {substances}")

    save_history_dir = f"history/{task}/{machine}"
    save_checkpoint_dir = f"checkpoint/{task}/{machine}"
    save_data_dir = f"data/{task}/{machine}"
    os.makedirs(save_history_dir, exist_ok=True)
    os.makedirs(save_checkpoint_dir, exist_ok=True)

    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    folds = list(kf.split(np.arange(len(full_ds.food_raw)), full_ds.food_raw))

    fold_histories = []
    # FOLDS=1,2,3 restricts this process to only those (1-indexed) folds --
    # e.g. to split the 5 folds across 2 GPUs by running two processes, one
    # per CUDA_VISIBLE_DEVICES, each with a disjoint FOLDS set.
    only_folds = None
    if os.environ.get("FOLDS"):
        only_folds = {int(f.strip()) for f in os.environ["FOLDS"].split(",")}

    for fold, (train_idx, val_idx) in enumerate(folds):
        if only_folds is not None and (fold + 1) not in only_folds:
            continue
        save_history_path = f"{save_history_dir}/xspecmamba_joint_fold{fold + 1}.json"
        if os.path.exists(save_history_path):
            print(f"Skipping fold {fold + 1}/{k_folds}: already completed")
            with open(save_history_path) as f:
                fold_histories.append(json.load(f))
            continue

        print(f"Fold {fold + 1}/{k_folds}")
        save_fold_dir = f"{save_data_dir}/fold_{fold + 1}"
        full_ds.fit_normalization_and_labels(train_idx, save_dir=save_fold_dir)
        img_ds = PesticideImageDataset(full_ds, img_size=img_size)

        cfg = XSpecMambaConfig(img_size=img_size, n_outputs=len(substances), emb_dim=128, depth=1, n_dirs=4,
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

        save_best_model_path = f"{save_checkpoint_dir}/checkpoint_fold{fold + 1}.pth"
        model, history = train_fold(
            model, train_loader, val_loader, device, max_epochs, optimizer, scheduler,
            substances, patience, save_history_path, save_best_model_path,
        )
        fold_histories.append(history)

    for name in substances:
        best_r2s = [max(h[f"val_r2_{name}"]) for h in fold_histories]
        print(f"{name}: average best val_r2 across {len(fold_histories)} fold(s): {np.mean(best_r2s):.4f}")
