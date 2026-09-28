"""Food-category classification (9 classes) for the four baselines, one file.

  METHOD=smartnir     SMART-NIR (focal loss, KAN head; was classification_engine.py)
  METHOD=guideddcnet  GuidedDCNet (CE pretraining + diffusion; was guideddcnet_classification_engine.py)
  METHOD=xgboost      XGBoost multi:softprob on GPU (was classification_xgboost_engine.py)
  METHOD=lightgbm     LightGBM multiclass (was classification_lightgbm_engine.py)

All use 5 StratifiedKFold splits (random_state=42) on the category label and report accuracy and
macro precision / recall / F1. Run e.g.
    MACHINE=FLAMENIR METHOD=smartnir python3 food_classification.py
    MACHINE=OCEANFX BIN=8 METHOD=smartnir python3 food_classification.py   # OCEANFX: bin wavelengths by 8
The shared training helpers (FocalLoss, train, pretrain, train_diffusion) are importable by the
scripts in additional_experiments/.
"""


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from dotenv import load_dotenv
from model.guideddcnet_model import GuidedDCNet, GuidedDCNetConfig
import time
import joblib
from sklearn.preprocessing import LabelEncoder
from dataset.preprocessing import preprocess_spectra


# ======================================================================
# smartnir  (from classification_engine.py)
# ======================================================================

load_dotenv()


class FocalLoss(nn.Module):
    """Multiclass focal loss with optional per-class alpha weighting, so
    well-classified (usually majority-class) examples contribute less to
    the gradient and rare classes get relatively more weight.
    """
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        loss = -((1 - pt) ** self.gamma) * log_pt
        if self.alpha is not None:
            loss = loss * self.alpha[targets]
        return loss.mean()


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

        # Scheduler (nếu có) - warmup+cosine steps per-epoch, no metric needed
        if scheduler is not None:
            scheduler.step()

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


# ======================================================================
# guideddcnet  (from guideddcnet_classification_engine.py)
# ======================================================================

def pretrain(model, data_head, train_loader, val_loader, device, epochs, optimizer):
    """Stage 1: CE loss on MCGM.pretrain_logits(x) (MCGM's two guidance
    vectors + a temporary Data Encoder head, discarded after this stage).

    Tracks and restores the best-val-acc (model, data_head) state at the end
    -- without this, Stage 2 would start from whatever epoch happened to run
    last, which can be meaningfully worse than the best epoch seen (val_acc
    is noisy enough here that the last epoch is often not the best one).
    """
    best_acc = 0.0
    best_model_wts = None
    best_data_head_wts = None

    for epoch in range(1, epochs + 1):
        model.train()
        data_head.train()
        running_loss = 0.0
        for X_batch, y_batch in tqdm(train_loader, desc=f"[Pretrain] Epoch {epoch}/{epochs}", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model.pretrain_logits(X_batch, data_head)
            loss = F.cross_entropy(logits, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        data_head.eval()
        val_running_loss = 0.0
        y_true, y_pred = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model.pretrain_logits(X_batch, data_head)
                loss = F.cross_entropy(logits, y_batch)
                val_running_loss += loss.item() * X_batch.size(0)
                preds = torch.argmax(logits, dim=1)
                y_true.extend(y_batch.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = accuracy_score(y_true, y_pred)

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_data_head_wts = {k: v.detach().clone() for k, v in data_head.state_dict().items()}

        print(
            f"[Pretrain] Epoch [{epoch}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
        data_head.load_state_dict(best_data_head_wts)
        print(f"[Pretrain] Restored best state (Val Acc: {best_acc:.4f})")


def train_diffusion(model, train_loader, val_loader, device, epochs, num_classes, optimizer,
                     scheduler=None, lam=0.5, patience=None,
                     save_history_path="history/guideddcnet_classification.json",
                     save_fig_path="history/plot.png",
                     save_best_model_path="checkpoint/checkpoint.pth"):
    """Stage 2: full diffusion training. Loss = noise MSE (dual guidance) +
    lam * (MMD_global + MMD_local). Validation predicts via full T-step
    reverse sampling (model.reverse_sample), matching how the model is
    actually used at inference time.

    patience=None (default) disables early stopping: the paper trains for a
    fixed epoch count, and val_acc here is doubly noisy (random-t diffusion
    loss + stochastic reverse_sample init), so a patience-based stop can
    trigger on a lucky/unlucky streak rather than true convergence -- seen
    concretely on OCEANFX fold 1, which early-stopped at epoch 49 (best at
    29) while every other fold ran 130-200+ epochs before its best. The best
    checkpoint (by val_acc) is still tracked and restored regardless.
    """
    best_acc = 0.0
    best_model_wts = None
    early_stop_counter = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
    }

    for epoch in tqdm(range(1, epochs + 1)):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y0 = F.one_hot(y_batch, num_classes).float()

            optimizer.zero_grad()
            loss_eps, loss_mmd_g, loss_mmd_l = model.training_losses(X_batch, y0)
            loss = loss_eps + lam * (loss_mmd_g + loss_mmd_l)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_running_loss = 0.0
        y_true, y_pred = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y0 = F.one_hot(y_batch, num_classes).float()

                loss_eps, loss_mmd_g, loss_mmd_l = model.training_losses(X_batch, y0)
                loss = loss_eps + lam * (loss_mmd_g + loss_mmd_l)
                val_running_loss += loss.item() * X_batch.size(0)

                y0_hat = model.reverse_sample(X_batch)
                preds = torch.argmax(y0_hat, dim=1)
                y_true.extend(y_batch.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = accuracy_score(y_true, y_pred)
        val_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        val_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        val_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_precision"].append(val_precision)
        history["val_recall"].append(val_recall)
        history["val_f1"].append(val_f1)

        if scheduler is not None:
            scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict()
            torch.save(best_model_wts, save_best_model_path)
            early_stop_counter = 0
        elif patience is not None:
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

    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    with open(save_history_path, "w") as f:
        json.dump(history, f, indent=4)

    epochs_range = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(20, 16))

    plt.subplot(2, 3, 1)
    plt.plot(epochs_range, history["train_loss"], label="Train Loss")
    plt.plot(epochs_range, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss"); plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(epochs_range, history["val_acc"], label="Accuracy", color="g")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Validation Accuracy"); plt.legend()

    plt.subplot(2, 3, 3)
    plt.plot(epochs_range, history["val_precision"], label="Precision", color="orange")
    plt.xlabel("Epoch"); plt.ylabel("Precision"); plt.title("Validation Precision"); plt.legend()

    plt.subplot(2, 3, 4)
    plt.plot(epochs_range, history["val_recall"], label="Recall", color="purple")
    plt.xlabel("Epoch"); plt.ylabel("Recall"); plt.title("Validation Recall"); plt.legend()

    plt.subplot(2, 3, 5)
    plt.plot(epochs_range, history["val_f1"], label="F1-score", color="red")
    plt.xlabel("Epoch"); plt.ylabel("F1"); plt.title("Validation F1-score"); plt.legend()

    plt.tight_layout()
    plt.savefig(save_fig_path)
    plt.close()

    return model, history


# ======================================================================
# xgboost  (from classification_xgboost_engine.py)
# ======================================================================

def fit_xgboost(X, y, tr, va, n_classes, ckpt):
    import xgboost as xgb
    dtr, dva = xgb.DMatrix(X[tr], label=y[tr]), xgb.DMatrix(X[va], label=y[va])
    params = dict(objective="multi:softprob", num_class=n_classes, eta=0.1, max_depth=6,
                  tree_method="hist", device="cuda", eval_metric="mlogloss")
    b = xgb.train(params, dtr, 300, evals=[(dva, "val")], early_stopping_rounds=20, verbose_eval=False)
    b.save_model(ckpt + ".json")
    return b.predict(dva, iteration_range=(0, b.best_iteration + 1)).argmax(1), int(b.best_iteration)


# ======================================================================
# lightgbm  (from classification_lightgbm_engine.py)
# ======================================================================

def fit_lightgbm(X, y, tr, va, n_classes, ckpt):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(objective="multiclass", n_estimators=300, learning_rate=0.1, max_depth=6,
                           num_leaves=63, feature_fraction=0.5, n_jobs=int(os.environ.get("N_JOBS", 48)),
                           random_state=42, verbose=-1, force_col_wise=True)
    m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], eval_metric="multi_logloss",
          callbacks=[lgb.early_stopping(20, verbose=False)])
    m.booster_.save_model(ckpt + ".txt")
    return m.predict(X[va]), int(m.best_iteration_ or 300)


# ======================================================================
# entry point: METHOD=smartnir
# ======================================================================
def main_smartnir():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_epochs = 200
    patience = 10
    k_folds = 5
    machine = f"{os.environ['MACHINE']}"
    task = "category_classification"

    full_path = f"{os.environ['DATASET_ROOT']}/{machine}/ALL.csv"
    # BIN=8 averages groups of 8 neighbouring wavelengths (used for OCEANFX; default: no binning)
    full_ds = ClassificationNIRSDataset(full_path, bin_factor=int(os.environ.get("BIN", 1)))

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

        # class-frequency alpha so Focal Loss weights rare categories more
        class_counts = np.bincount(full_ds.y[train_idx], minlength=full_ds.n_classes)
        alpha = (class_counts.sum() / (full_ds.n_classes * np.maximum(class_counts, 1)))
        alpha = torch.tensor(alpha, dtype=torch.float32, device=device)
        criterion = FocalLoss(gamma=2.0, alpha=alpha)

        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        warmup_epochs = max(1, int(0.05 * max_epochs))
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs
        )
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs - warmup_epochs
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
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


# ======================================================================
# entry point: METHOD=guideddcnet
# ======================================================================
def main_guideddcnet():
    """Trains GuidedDCNet (model/guideddcnet_model.py) as a baseline against
    SMART-NIR (food_classification.py) on the same 9-class vegetable
    classification task -- reuses the exact same ClassificationNIRSDataset and
    StratifiedKFold split as food_classification.py so both models are
    evaluated on identical folds. Standalone comparison work, not part of the
    progress report.
    
    Two-stage training per the paper: (1) 25-epoch Cross-Entropy pretraining of
    MCGM + Data Encoder, (2) up to 200-epoch full diffusion training (noise MSE
    + MMD regularization), with reverse-diffusion sampling used for validation
    accuracy and best-checkpoint selection.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pretrain_epochs = 25
    max_epochs = 200
    patience = None  # no early stopping: fixed epoch budget for every fold, see train_diffusion docstring
    k_folds = 5
    lam = 0.5
    machine = f"{os.environ['MACHINE']}"
    task = "category_classification_guideddcnet"

    # GuidedDCNet is tiny next to SMART-NIR (plain ResNet1D encoders, no
    # attention) and the original batch_size=512 left the GPU at ~10%
    # utilization -- epoch time was dominated by Python/DataLoader overhead
    # (225 epochs/fold x re-spawning 4 worker processes each epoch, since
    # persistent_workers defaults to False) rather than by GPU compute; fixing
    # that (persistent_workers + pin_memory below) accounts for most of the
    # speedup. Batch size itself was benchmarked in isolation on an otherwise
    # idle GPU (pure compute, no DataLoader, no other job running) across
    # 1024-16384: throughput is NOT monotonic in batch size, for either
    # stage. Stage 2 (diffusion training, 200 of the 225 epochs/fold, 3 UNet
    # forward/backward passes per step for the dual/global-only/local-only
    # guidance terms) bottoms out at 3072 (~9.6 s/epoch), a bit worse at 2048
    # (~10.9 s/epoch) and 4096 (~12.0 s/epoch), and clearly worse beyond that
    # (~19 s/epoch at 8192). Stage 1 (CE pretraining) is fastest at 3072-4096
    # (~5.5-5.7 s/epoch). 3072 is the sweet spot for both. Adam's updates
    # scale roughly with sqrt(batch size) (unlike plain SGD's linear rule),
    # so each lr below is scaled by sqrt(batch_size / 512) to keep the
    # optimization comparable at the new batch size instead of just diluting
    # the gradient signal.
    batch_size = int(os.environ.get("GDC_BATCH_SIZE", 3072))
    num_workers = int(os.environ.get("GDC_NUM_WORKERS", 8))
    lr_scale = (batch_size / 512) ** 0.5

    full_path = f"{os.environ['DATASET_ROOT']}/{machine}/ALL.csv"
    # BIN=8 averages groups of 8 neighbouring wavelengths (used for OCEANFX; default: no binning)
    full_ds = ClassificationNIRSDataset(full_path, bin_factor=int(os.environ.get("BIN", 1)))

    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_histories = []

    save_history_dir = f"history/{task}/{machine}"
    save_checkpoint_dir = f"checkpoint/{task}/{machine}"
    os.makedirs(save_history_dir, exist_ok=True)
    os.makedirs(save_checkpoint_dir, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(full_ds.y_raw)), full_ds.y_raw)):
        save_history_path = f"{save_history_dir}/guideddcnet_classification_fold{fold + 1}.json"

        # Resume support: a long run can get cut off mid-way, and a restart
        # otherwise re-trains every fold (pretrain + full diffusion
        # training) from scratch. A fold's history file is only written
        # after that fold's full training run succeeds, so its presence
        # reliably marks it done.
        if os.path.exists(save_history_path):
            print(f"Skipping fold {fold + 1}/{k_folds}: already completed")
            with open(save_history_path) as f:
                fold_histories.append(json.load(f))
            continue

        print(f"Fold {fold + 1}/{k_folds}")

        save_fold_dir = f"data/{task}/{machine}/fold_{fold + 1}"
        full_ds.fit_normalization_and_labels(train_idx, save_dir=save_fold_dir)

        cfg = GuidedDCNetConfig(num_classes=full_ds.n_classes)
        model = GuidedDCNet(cfg).to(device)
        data_head = nn.Linear(model.data_encoder.embed_dim, full_ds.n_classes).to(device)

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        # drop_last on train: GuidedDCNet has many more BatchNorm1d layers
        # than SMART-NIR (3 ResNet1D encoders + UNet blocks), so avoid ever
        # feeding a batch of size 1 (BatchNorm needs >1 sample in train mode).
        # persistent_workers keeps the worker pool alive across the 225
        # epochs/fold instead of respawning it every epoch; pin_memory speeds
        # up the host->GPU transfer.
        train_loader = DataLoader(full_ds, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers,
                                  drop_last=True, pin_memory=True, persistent_workers=num_workers > 0)
        val_loader = DataLoader(full_ds, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers,
                                pin_memory=True, persistent_workers=num_workers > 0)

        # ---- Stage 1: pretrain MCGM + Data Encoder (Cross-Entropy) ----
        pretrain_params = (
            list(model.mcgm.parameters())
            + list(model.data_encoder.parameters())
            + list(data_head.parameters())
        )
        pretrain_optimizer = optim.Adam(pretrain_params, lr=5e-4 * lr_scale)
        pretrain(model, data_head, train_loader, val_loader, device, pretrain_epochs, pretrain_optimizer)

        # ---- Stage 2: full diffusion training ----
        optimizer = optim.Adam([
            {"params": list(model.mcgm.parameters()) + list(model.data_encoder.parameters()), "lr": 5e-4 * lr_scale},
            {"params": model.unet.parameters(), "lr": 1e-3 * lr_scale},
        ])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

        save_fig_path = f"{save_history_dir}/plot_fold{fold + 1}.png"
        save_best_model_path = f"{save_checkpoint_dir}/checkpoint_fold{fold + 1}.pth"

        model, history = train_diffusion(
            model, train_loader, val_loader, device, max_epochs, full_ds.n_classes, optimizer,
            scheduler=scheduler, lam=lam, patience=patience,
            save_history_path=save_history_path, save_fig_path=save_fig_path,
            save_best_model_path=save_best_model_path,
        )

        fold_histories.append(history)

    avg_val_acc = np.mean([max(h["val_acc"]) for h in fold_histories])
    print(f"Average best validation accuracy across folds: {avg_val_acc:.4f}")


# ======================================================================
# entry point: METHOD=xgboost
# ======================================================================
def main_xgboost():
    """Food-category classification (9 classes) with XGBoost (multi:softprob, GPU
    histogram trees), as the ML baseline next to food_classification.py
    (SMART-NIR) and food_classification.py (GuidedDCNet).
    
    Same protocol as those engines: 5 StratifiedKFold splits (random_state=42) on
    the category label, accuracy and macro precision/recall/F1 per fold. Trees are
    invariant to feature scaling, so the SG+SNV spectra are used as they are.
    Early stopping on validation mlogloss.
    
    Env: MACHINE (FLAMENIR | OCEANFX).
    """
    machine = os.environ["MACHINE"]
    task = "category_classification_xgboost"
    hist_dir, ckpt_dir = f"history/{task}/{machine}", f"checkpoint/{task}/{machine}"
    os.makedirs(hist_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    df = pd.read_csv(f"{os.environ['DATASET_ROOT']}/{machine}/ALL.csv")
    X, keep = preprocess_spectra(df[[c for c in df.columns if c.startswith("w_")]].values.astype(np.float32))
    le = LabelEncoder().fit(df["category"].values[keep])
    y = le.transform(df["category"].values[keep])

    for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y), start=1):
        print(f"Fold {fold}/5", flush=True)
        t0 = time.time()
        os.makedirs(f"data/{task}/{machine}/fold_{fold}", exist_ok=True)
        joblib.dump(le, f"data/{task}/{machine}/fold_{fold}/label_encoder.pkl")
        pred, best_it = fit_xgboost(X, y, tr, va, len(le.classes_), f"{ckpt_dir}/fold{fold}")
        yv = y[va]
        hist = {
            "val_acc": [accuracy_score(yv, pred)],
            "val_precision": [precision_score(yv, pred, average="macro", zero_division=0)],
            "val_recall": [recall_score(yv, pred, average="macro", zero_division=0)],
            "val_f1": [f1_score(yv, pred, average="macro", zero_division=0)],
            "best_iteration": [best_it], "seconds": [time.time() - t0],
        }
        with open(f"{hist_dir}/fold{fold}.json", "w") as f:
            json.dump(hist, f, indent=2)
        print(f"  -> acc={hist['val_acc'][0]:.4f} prec={hist['val_precision'][0]:.4f} "
              f"rec={hist['val_recall'][0]:.4f} f1={hist['val_f1'][0]:.4f} ({hist['seconds'][0]:.0f}s)", flush=True)


# ======================================================================
# entry point: METHOD=lightgbm
# ======================================================================
def main_lightgbm():
    """Food-category classification (9 classes) with LightGBM (multiclass), as the
    second ML baseline next to food_classification.py (SMART-NIR) and
    food_classification.py (GuidedDCNet).
    
    Same protocol as those engines: 5 StratifiedKFold splits (random_state=42) on
    the category label, accuracy and macro precision/recall/F1 per fold. Trees are
    invariant to feature scaling, so the SG+SNV spectra are used as they are.
    Early stopping on validation multi_logloss; feature_fraction 0.5 keeps the
    2136-wavelength OCEANFX spectra tractable on CPU.
    
    Env: MACHINE (FLAMENIR | OCEANFX), N_JOBS.
    """
    machine = os.environ["MACHINE"]
    task = "category_classification_lightgbm"
    hist_dir, ckpt_dir = f"history/{task}/{machine}", f"checkpoint/{task}/{machine}"
    os.makedirs(hist_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    df = pd.read_csv(f"{os.environ['DATASET_ROOT']}/{machine}/ALL.csv")
    X, keep = preprocess_spectra(df[[c for c in df.columns if c.startswith("w_")]].values.astype(np.float32))
    le = LabelEncoder().fit(df["category"].values[keep])
    y = le.transform(df["category"].values[keep])

    for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y), start=1):
        print(f"Fold {fold}/5", flush=True)
        t0 = time.time()
        os.makedirs(f"data/{task}/{machine}/fold_{fold}", exist_ok=True)
        joblib.dump(le, f"data/{task}/{machine}/fold_{fold}/label_encoder.pkl")
        pred, best_it = fit_lightgbm(X, y, tr, va, len(le.classes_), f"{ckpt_dir}/fold{fold}")
        yv = y[va]
        hist = {
            "val_acc": [accuracy_score(yv, pred)],
            "val_precision": [precision_score(yv, pred, average="macro", zero_division=0)],
            "val_recall": [recall_score(yv, pred, average="macro", zero_division=0)],
            "val_f1": [f1_score(yv, pred, average="macro", zero_division=0)],
            "best_iteration": [best_it], "seconds": [time.time() - t0],
        }
        with open(f"{hist_dir}/fold{fold}.json", "w") as f:
            json.dump(hist, f, indent=2)
        print(f"  -> acc={hist['val_acc'][0]:.4f} prec={hist['val_precision'][0]:.4f} "
              f"rec={hist['val_recall'][0]:.4f} f1={hist['val_f1'][0]:.4f} ({hist['seconds'][0]:.0f}s)", flush=True)


if __name__ == "__main__":
    _method = os.environ.get("METHOD")
    _mains = {'smartnir': main_smartnir, 'guideddcnet': main_guideddcnet, 'xgboost': main_xgboost, 'lightgbm': main_lightgbm}
    if _method not in _mains:
        raise SystemExit("set METHOD to one of: " + ", ".join(_mains))
    _mains[_method]()

