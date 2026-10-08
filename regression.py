"""Earlier per-substance concentration regression (Stage-2 regression), one file.

  METHOD=stage2      SMART-NIR regressor, per-substance KFold (was regression_stage2_engine.py)
  METHOD=ebar        EBAR (was ebar_regression_engine.py)
  METHOD=nirmacnet   NirMACNet (was nirmacnet_regression_engine.py)
  METHOD=xspecmamba  XSpecMamba (was xspecmamba_regression_engine.py)

The report no longer includes a regression task; these remain as baselines and are imported by the
scripts in additional_experiments/. Run e.g.
    MACHINE=FLAMENIR METHOD=stage2 python3 regression.py
"""


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import json
import os
from dotenv import load_dotenv
from model.regression_model import SMARTNIRRegressor, SmartNIRRegressionConfig
from dataset.regression_dataset import RegressionNIRSDataset
import warnings
from joblib import dump
from sklearn.exceptions import ConvergenceWarning
from model.ebar_model import build_ebar_model
from model.nirmacnet_model import NirMACNet, NirMACNetConfig
from torch.utils.data import Dataset
from model.xspecmamba_model import XSpecMamba, XSpecMambaConfig, spectrum_to_image


# ======================================================================
# stage2  (from regression_stage2_engine.py)
# ======================================================================

load_dotenv()


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

        # Lưu vào history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)
        history["val_rmse"].append(val_rmse)
        history["val_r2"].append(val_r2)

        # Scheduler (nếu có)
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


# ======================================================================
# ebar  (from ebar_regression_engine.py)
# ======================================================================

# LassoCV/ElasticNetCV probe a range of regularization strengths internally;
# the weakest ones are near-unregularized least-squares on highly collinear
# NIR wavelengths, which coordinate descent converges to slowly/imperfectly.
# Benign (the CV-selected alpha is rarely one of these), just noisy.
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def fit_xy_scaler(X_train, y_train):
    """Z-score for X; log1p + z-score for y.

    The paper's own preprocessing is plain standardization (no log1p), but
    several substances here are far more right-skewed than the paper's
    dataset -- e.g. Permethrin's median is ~1.02 while its 90th percentile
    is ~216 (top 10% of samples carry ~74% of the target's total variance).
    Under plain z-scoring, an MSE-driven ensemble is graded almost entirely
    on that thin high-concentration tail; base learners undershoot it badly
    (SVR members in particular fit on a random <=3000-row subsample that
    rarely contains enough tail examples), which was producing R2 as low as
    -2.46 on Permethrin -- worse than predicting the mean. log1p first (the
    same transform RegressionNIRSDataset/SMART-NIR/every other baseline
    here already uses) keeps this a fair apples-to-apples comparison instead
    of penalizing EBAR for a data-scaling artifact unrelated to the
    stacking architecture itself.
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

        # Resume support: a long multi-machine run can get cut off mid-way
        # (e.g. a Kaggle session time limit) with no way to continue from
        # where it stopped, since each restart re-trains every substance
        # from scratch. A completed fold's history file is only ever
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


# ======================================================================
# xspecmamba  (from xspecmamba_regression_engine.py)
# ======================================================================

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

    def __init__(self, base_ds: RegressionNIRSDataset, img_size: int = 64):
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


# ======================================================================
# entry point: METHOD=stage2
# ======================================================================
def main_stage2():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_epochs = 500
    patience = 50
    k_folds = 5
    machine = f"{os.environ['MACHINE']}"
    task = "substance_regression"

    full_path = f"{os.environ['DATASET_ROOT']}/{machine}/ALL.csv"
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

        folds = make_folds(full_ds.y_raw, k_folds)

        fold_histories = []

        save_history_dir = f"history/{task}/stage2/{machine}/{substance}"
        save_fig_dir = f"history/{task}/stage2/{machine}/{substance}"
        save_best_model_dir = f"checkpoint/{task}/stage2/{machine}/{substance}"
        os.makedirs(save_history_dir, exist_ok=True)
        os.makedirs(save_fig_dir, exist_ok=True)
        os.makedirs(save_best_model_dir, exist_ok=True)

        for fold, (train_idx, val_idx) in enumerate(folds):
            save_history_path = f"{save_history_dir}/smart_nir_regression_fold{fold + 1}.json"

            # Resume support: a long run can get cut off mid-way, and a
            # restart otherwise re-trains every substance from scratch. A
            # fold's history file is only written after that fold's full
            # training run succeeds, so its presence reliably marks it done.
            if os.path.exists(save_history_path):
                print(f"Skipping fold {fold + 1}/{k_folds} for {substance}: already completed")
                with open(save_history_path) as f:
                    fold_histories.append(json.load(f))
                continue

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

            save_fig_path = f"{save_fig_dir}/plot_fold{fold + 1}.png"
            save_best_model_path = f"{save_best_model_dir}/checkpoint_fold{fold + 1}.pth"

            model, history = train(
                model, train_loader, val_loader, device, max_epochs, criterion, optimizer,
                scheduler=scheduler, patience=patience, inverse_transform_y=full_ds.inverse_transform_y,
                save_history_path=save_history_path, save_fig_path=save_fig_path,
                save_best_model_path=save_best_model_path
            )

            fold_histories.append(history)

        if fold_histories:
            avg_best_r2 = np.mean([max(h["val_r2"]) for h in fold_histories])
            print(f"Average best validation R² across folds for {substance}: {avg_best_r2:.4f}")


# ======================================================================
# entry point: METHOD=ebar
# ======================================================================
def main_ebar():
    """Trains EBAR (model/ebar_model.py) as a Stage 2 (concentration prediction)
    baseline against SMART-NIR (regression.py) -- reuses the exact
    same RegressionNIRSDataset preprocessing and make_folds() fold split per
    substance so both models are evaluated on identical folds. Standalone
    comparison work, not part of the progress report.
    
    Unlike SMART-NIR, EBAR is not epoch-trained: each fold is a single
    fit-and-evaluate, mirroring how stage1_detection.py (XGBoost)
    records single-value "epoch" lists rather than per-epoch curves.
    """
    machine = f"{os.environ['MACHINE']}"
    task = "substance_regression"
    k_folds = 5

    full_path = f"{os.environ['DATASET_ROOT']}/{machine}/ALL.csv"

    substances = [
        'Thiamethoxam', 'Permethrin', 'Metalaxyl', 'Azoxystrobin',
        'Imidaclopird', 'Difenoconazole', 'Cypermethrin', 'Cyhalothrin',
        'Chlorantraniliprol', 'Chlopyrifos Methyl', 'Emamectin benzoate',
        'Chlorothalonil', 'Triadimefon', 'Cyantraniliprole', 'Flutolanil',
        'Indoxacarb', 'Abamectin', 'Propamocarb.HCL', 'Chlothianidin'
    ]

    for substance in substances:
        print(f"Training for {substance}")

        save_history_dir = f"history/{task}/stage2_ebar/{machine}/{substance}"
        save_checkpoint_dir = f"checkpoint/{task}/stage2_ebar/{machine}/{substance}"

        try:
            full_ds = RegressionNIRSDataset(full_path, substance)
        except ValueError as e:
            print(f"Skipping {substance}: {e}")
            os.makedirs(save_history_dir, exist_ok=True)
            os.makedirs(save_checkpoint_dir, exist_ok=True)
            continue

        # same fold split as SMART-NIR Stage 2 for a fair, apples-to-apples comparison
        folds = make_folds(full_ds.y_raw, k_folds)

        train_for_substance(full_ds.X_raw, full_ds.y_raw, substance, folds,
                             save_history_dir, save_checkpoint_dir)


# ======================================================================
# entry point: METHOD=nirmacnet
# ======================================================================
def main_nirmacnet():
    """Trains NirMACNet (model/nirmacnet_model.py) as a Stage 2 (concentration
    prediction) baseline against SMART-NIR -- reuses regression.py's
    `train()` loop and `make_folds()` split verbatim (only the model/optimizer
    differ), and the same RegressionNIRSDataset preprocessing/target transform,
    so both models are evaluated on identical folds. Standalone comparison work,
    not part of the progress report.
    
    Training follows the paper: SGD (lr=1e-5) with linear warmup + cosine
    annealing, MSE loss, up to 200 epochs -- unlike SMART-NIR's Adam.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_epochs = 200
    patience = 20
    k_folds = 5
    machine = f"{os.environ['MACHINE']}"
    task = "substance_regression"

    full_path = f"{os.environ['DATASET_ROOT']}/{machine}/ALL.csv"

    substances = [
        'Thiamethoxam', 'Permethrin', 'Metalaxyl', 'Azoxystrobin',
        'Imidaclopird', 'Difenoconazole', 'Cypermethrin', 'Cyhalothrin',
        'Chlorantraniliprol', 'Chlopyrifos Methyl', 'Emamectin benzoate',
        'Chlorothalonil', 'Triadimefon', 'Cyantraniliprole', 'Flutolanil',
        'Indoxacarb', 'Abamectin', 'Propamocarb.HCL', 'Chlothianidin'
    ]
    # REVERSE=1 lets a second process work the list back-to-front so it can
    # run alongside a forward pass (e.g. one process resuming an interrupted
    # Kaggle run top-down, another started locally bottom-up) without both
    # racing for the same next substance -- the per-fold history-file skip
    # check still prevents any duplicate training if they do meet.
    if os.environ.get("REVERSE") == "1":
        substances = list(reversed(substances))

    for substance in substances:
        print(f"Training for {substance}")
        if substance not in pd.read_csv(full_path, nrows=0).columns:
            print(f"Skipping {substance}: column not found")
            continue

        save_history_dir = f"history/{task}/stage2_nirmacnet/{machine}/{substance}"
        save_fig_dir = save_history_dir
        save_best_model_dir = f"checkpoint/{task}/stage2_nirmacnet/{machine}/{substance}"

        try:
            full_ds = RegressionNIRSDataset(full_path, substance)
        except ValueError as e:
            print(f"Skipping {substance}: {e}")
            os.makedirs(save_history_dir, exist_ok=True)
            os.makedirs(save_best_model_dir, exist_ok=True)
            continue

        # same fold split as SMART-NIR Stage 2 for a fair, apples-to-apples comparison
        folds = make_folds(full_ds.y_raw, k_folds)

        fold_histories = []
        os.makedirs(save_history_dir, exist_ok=True)
        os.makedirs(save_best_model_dir, exist_ok=True)

        for fold, (train_idx, val_idx) in enumerate(folds):
            save_history_path = f"{save_history_dir}/nirmacnet_regression_fold{fold + 1}.json"

            # Resume support: a Kaggle session can get cut off mid-run, and
            # a restart otherwise re-trains every substance from scratch.
            # A fold's history file is only written after that fold's full
            # training run succeeds, so its presence reliably marks it done.
            if os.path.exists(save_history_path):
                print(f"Skipping fold {fold + 1}/{k_folds} for {substance}: already completed")
                with open(save_history_path) as f:
                    fold_histories.append(json.load(f))
                continue

            print(f"Fold {fold + 1}/{k_folds} for {substance}")

            save_fold_dir = f"data/{task}/stage2_nirmacnet/{machine}/{substance}/fold_{fold + 1}"
            full_ds.fit_normalization(train_idx, save_dir=save_fold_dir)

            cfg = NirMACNetConfig(num_targets=1)
            model = NirMACNet(cfg).to(device)

            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)
            # micro_batch=128 + accum_steps=4 (effective batch 512, the
            # project's usual size), not batch_size=512 directly: unlike
            # SMART-NIR's backbone (which downsamples early via stride-4
            # MultiKernelBlock convs), NirMACNet's 3 parallel branches keep
            # the full sequence length through several channels-256 Conv1d
            # layers, so per-forward-pass activation memory scales directly
            # with signal length. FLAMENIR (128 wavelengths) fit fine at a
            # real batch_size=512, but OCEANFX (2136 wavelengths, ~17x
            # longer) OOM'd immediately on a 15GB Kaggle T4 at the same
            # batch size -- confirmed by machine, not sample count, since
            # FLAMENIR has *more* rows and ran without issue. Gradient
            # accumulation keeps the effective batch (and thus the
            # optimizer/schedule behavior) identical across both machines
            # while capping peak per-step memory to the micro-batch's.
            micro_batch = 128
            accum_steps = 4
            train_loader = DataLoader(full_ds, batch_size=micro_batch, sampler=train_sampler, num_workers=4,
                                       drop_last=True)
            val_loader = DataLoader(full_ds, batch_size=micro_batch, sampler=val_sampler, num_workers=4)

            criterion = nn.MSELoss()
            # lr=1e-3, momentum=0.9 -- not the paper's lr=1e-5/momentum=0:
            # that combination barely moves train_loss in 200 epochs (0.845
            # -> 0.818) and val_r2 sits flat at roughly -0.08 the entire run
            # (confirmed on both machines, so not specific to one dataset) --
            # essentially untrained, worse than predicting the mean. Isolated
            # testing (15 epochs, Thiamethoxam/OCEANFX) showed lr=1e-3 +
            # momentum=0.9 reaches val_r2=0.86 in that same budget, so the
            # paper's rate is simply too small for SGD to make progress on
            # this architecture in a practical epoch count.
            optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
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

            save_fig_path = f"{save_fig_dir}/plot_fold{fold + 1}.png"
            save_best_model_path = f"{save_best_model_dir}/checkpoint_fold{fold + 1}.pth"

            model, history = train(
                model, train_loader, val_loader, device, max_epochs, criterion, optimizer,
                scheduler=scheduler, patience=patience, inverse_transform_y=full_ds.inverse_transform_y,
                accum_steps=accum_steps,
                save_history_path=save_history_path, save_fig_path=save_fig_path,
                save_best_model_path=save_best_model_path,
            )

            fold_histories.append(history)

        if fold_histories:
            avg_best_r2 = np.mean([max(h["val_r2"]) for h in fold_histories])
            print(f"Average best validation R2 across folds for {substance}: {avg_best_r2:.4f}")


# ======================================================================
# entry point: METHOD=xspecmamba
# ======================================================================
def main_xspecmamba():
    """Trains XSpecMamba/SpectralMamba (model/xspecmamba_model.py) as a Stage 2
    (concentration prediction) baseline against SMART-NIR -- reuses
    regression.py's `train()` loop and `make_folds()` split
    verbatim, and the same RegressionNIRSDataset preprocessing/target transform,
    so both models are evaluated on identical folds. Standalone comparison work,
    not part of the progress report.
    
    Each 1D spectrum is converted to a 3-channel (GAF/RP/Corr) 64x64 image via
    `SpectralImageDataset` (model/xspecmamba_model.spectrum_to_image), which now
    builds GAF/RP/Corr at the spectrum's native length and bilinear-downsamples
    to 64x64 (matching the reference paper's own preprocessing order), rather
    than the cheaper PAA-resize-then-transform shortcut this file used to take.
    Because that's real O(N^2) work, SpectralImageDataset precomputes every
    sample's image once in __init__ instead of lazily per __getitem__ -- doing
    it lazily would repeat the full-resolution transform on every epoch for no
    reason, since it depends only on the (fixed, already-fold-normalized) input
    spectrum, not on model state.
    
    Optimizer/scheduler: AdamW (lr=5e-4 peak, 5% linear warmup + cosine
    annealing) with gradient clipping (max-norm 1.0), matching the reference
    paper's stated training recipe.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_epochs = 200
    patience = 20
    k_folds = 5
    img_size = 64
    machine = f"{os.environ['MACHINE']}"
    task = "substance_regression"

    full_path = f"{os.environ['DATASET_ROOT']}/{machine}/ALL.csv"

    substances = [
        'Thiamethoxam', 'Permethrin', 'Metalaxyl', 'Azoxystrobin',
        'Imidaclopird', 'Difenoconazole', 'Cypermethrin', 'Cyhalothrin',
        'Chlorantraniliprol', 'Chlopyrifos Methyl', 'Emamectin benzoate',
        'Chlorothalonil', 'Triadimefon', 'Cyantraniliprole', 'Flutolanil',
        'Indoxacarb', 'Abamectin', 'Propamocarb.HCL', 'Chlothianidin'
    ]
    # Optional SUBSTANCES=A,B,C env override to restrict this run to a
    # subset (e.g. a later report scoped to fewer compounds) without
    # touching the full-19 default used by prior runs.
    if os.environ.get("SUBSTANCES"):
        substances = [s.strip() for s in os.environ["SUBSTANCES"].split(",")]

    for substance in substances:
        print(f"Training for {substance}")
        if substance not in pd.read_csv(full_path, nrows=0).columns:
            print(f"Skipping {substance}: column not found")
            continue

        save_history_dir = f"history/{task}/stage2_xspecmamba/{machine}/{substance}"
        save_best_model_dir = f"checkpoint/{task}/stage2_xspecmamba/{machine}/{substance}"

        try:
            # apply_savgol_snv=False: the reference paper explicitly does
            # NOT apply Savitzky-Golay or scatter correction externally --
            # its NIR Gradient Enhancement module (model/xspecmamba_model.py)
            # is designed to learn that transform end-to-end from
            # relatively raw spectra instead.
            full_ds = RegressionNIRSDataset(full_path, substance, apply_savgol_snv=False)
        except ValueError as e:
            print(f"Skipping {substance}: {e}")
            os.makedirs(save_history_dir, exist_ok=True)
            os.makedirs(save_best_model_dir, exist_ok=True)
            continue

        # same fold split as SMART-NIR Stage 2 for a fair, apples-to-apples comparison
        folds = make_folds(full_ds.y_raw, k_folds)

        fold_histories = []
        os.makedirs(save_history_dir, exist_ok=True)
        os.makedirs(save_best_model_dir, exist_ok=True)

        for fold, (train_idx, val_idx) in enumerate(folds):
            save_history_path = f"{save_history_dir}/xspecmamba_regression_fold{fold + 1}.json"

            # Resume support: a Kaggle session can get cut off mid-run, and
            # a restart otherwise re-trains every substance from scratch.
            # A fold's history file is only written after that fold's full
            # training run succeeds, so its presence reliably marks it done.
            if os.path.exists(save_history_path):
                print(f"Skipping fold {fold + 1}/{k_folds} for {substance}: already completed")
                with open(save_history_path) as f:
                    fold_histories.append(json.load(f))
                continue

            print(f"Fold {fold + 1}/{k_folds} for {substance}")

            save_fold_dir = f"data/{task}/stage2_xspecmamba/{machine}/{substance}/fold_{fold + 1}"
            full_ds.fit_normalization(train_idx, save_dir=save_fold_dir, apply_zscore_X=False)
            img_ds = SpectralImageDataset(full_ds, img_size=img_size)

            cfg = XSpecMambaConfig(img_size=img_size, n_outputs=1, emb_dim=128, depth=1, n_dirs=4,
                                    backbone="mamba")
            model = XSpecMamba(cfg).to(device)

            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)
            train_loader = DataLoader(img_ds, batch_size=64, sampler=train_sampler, num_workers=4, drop_last=True)
            val_loader = DataLoader(img_ds, batch_size=64, sampler=val_sampler, num_workers=4)

            criterion = nn.MSELoss()
            # AdamW lr=5e-4 peak + 5% linear warmup + cosine annealing, grad
            # clip max-norm 1.0 -- matches the reference paper's training
            # recipe (beta1=0.9, beta2=0.999, weight_decay=0.01 are AdamW's
            # own defaults already).
            optimizer = optim.AdamW(model.parameters(), lr=5e-4)
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

            save_fig_path = f"{save_history_dir}/plot_fold{fold + 1}.png"
            save_best_model_path = f"{save_best_model_dir}/checkpoint_fold{fold + 1}.pth"

            model, history = train(
                model, train_loader, val_loader, device, max_epochs, criterion, optimizer,
                scheduler=scheduler, patience=patience, inverse_transform_y=full_ds.inverse_transform_y,
                grad_clip_norm=1.0,
                save_history_path=save_history_path, save_fig_path=save_fig_path,
                save_best_model_path=save_best_model_path,
            )

            fold_histories.append(history)

        if fold_histories:
            avg_best_r2 = np.mean([max(h["val_r2"]) for h in fold_histories])
            print(f"Average best validation R2 across folds for {substance}: {avg_best_r2:.4f}")


if __name__ == "__main__":
    _method = os.environ.get("METHOD")
    _mains = {'stage2': main_stage2, 'ebar': main_ebar, 'nirmacnet': main_nirmacnet, 'xspecmamba': main_xspecmamba}
    if _method not in _mains:
        raise SystemExit("set METHOD to one of: " + ", ".join(_mains))
    _mains[_method]()

