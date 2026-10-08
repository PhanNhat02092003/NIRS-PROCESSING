import os

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from dataset.preprocessing import preprocess_spectra


class PesticideMultiTaskDataset(Dataset):
    """Joint (food category, per-substance presence, per-substance log1p
    concentration) dataset for an explicit pesticide subset -- shared by
    the MT-SMARTNIR 3-head engine (multitask_mt_smartnir_engine.py) and the
    joint multi-output XSpecMamba baseline (xspecmamba_joint_pesticide_engine.py).

    Unlike JointRegressionNIRSDataset (dataset/joint_regression_dataset.py),
    this does NOT drop rows unless every target is valid: pesticide presence
    is sparse and effectively non-overlapping (checked directly against the
    real data: 0 of 335k+ FLAMENIR rows and 0 of 380k+ OCEANFX rows have
    Thiamethoxam/Permethrin/Azoxystrobin/Difenoconazole/Cypermethrin/
    Chlothianidin all present at once), so "require all present" would
    leave zero training rows. Every row is kept; each substance's presence
    mask (1 = present) doubles as that substance's regression loss mask,
    matching ../MT-SMART-NIR's own MaskedRegressionLoss convention -- the
    target for absent rows is exactly log1p(0) = 0, not a placeholder.
    """

    def __init__(self, data_filepath: str, pesticide_cols: list, bin_factor: int = 1,
                 sample_n: int = None, sample_seed: int = 42):
        df = pd.read_csv(data_filepath)

        if sample_n is not None and sample_n < len(df):
            # Stratified by category so a small subset doesn't accidentally
            # drop or starve a food class -- done before preprocess_spectra
            # so the expensive SG/SNV pass only runs on the kept rows.
            df = (
                df.groupby("category", group_keys=False)
                .apply(lambda g: g.sample(
                    n=max(1, round(sample_n * len(g) / len(df))),
                    random_state=sample_seed))
                .reset_index(drop=True)
            )
            print(f"[PesticideMultiTaskDataset] subsampled to {len(df)} rows (sample_n={sample_n})")

        X_raw = df[[c for c in df.columns if c.startswith("w_")]].values.astype(np.float32)
        food_raw = df["category"].values.ravel()
        raw_pest = df[pesticide_cols].values.astype(np.float32)  # -1 sentinel = absent

        X_raw, keep_mask = preprocess_spectra(X_raw)
        n_dropped = (~keep_mask).sum()
        if n_dropped:
            print(f"[PesticideMultiTaskDataset] dropped {n_dropped} outlier spectra out of {len(keep_mask)}")

        if bin_factor > 1:
            n_out = (X_raw.shape[1] // bin_factor) // 8 * 8
            X_raw = X_raw[:, :n_out * bin_factor].reshape(len(X_raw), n_out, bin_factor).mean(axis=2).astype(np.float32)
            print(f"[PesticideMultiTaskDataset] binned wavelengths by {bin_factor}: -> {n_out} points")

        self.pesticide_cols = list(pesticide_cols)
        self.n_pesticides = len(pesticide_cols)
        self.X_raw = X_raw
        self.food_raw = food_raw[keep_mask]
        raw_pest = raw_pest[keep_mask]
        self.pres_raw = (raw_pest > 0).astype(np.float32)  # (N, n_pest)
        conc_mgkg = np.where(raw_pest > 0, raw_pest, 0.0).astype(np.float32)
        # Fixed (not fold-dependent): log1p(0)=0 for absent rows is the
        # correct "no concentration" target, not a placeholder needing a
        # per-fold mean/std -- see MaskedRegressionLoss's absent-suppression
        # term, which specifically targets this same zero.
        self.conc_log = np.log1p(conc_mgkg)  # (N, n_pest)
        self.signal_length = X_raw.shape[1]

        self.mean = None
        self.std = None
        self.label_encoder = None
        self.X = None
        self.food_y = None
        self.n_classes = None

    def fit_normalization_and_labels(self, train_indices, save_dir=None):
        X_train = self.X_raw[train_indices]
        self.mean = X_train.mean(axis=0, keepdims=True)
        self.std = X_train.std(axis=0, keepdims=True) + 1e-8
        self.X = (self.X_raw - self.mean) / self.std

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.food_raw[train_indices])
        self.food_y = self.label_encoder.transform(self.food_raw)
        self.n_classes = len(self.label_encoder.classes_)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            np.savez(os.path.join(save_dir, "stats.npz"), mean=self.mean, std=self.std)
            joblib.dump(self.label_encoder, os.path.join(save_dir, "label_encoder.pkl"))

    def __len__(self):
        if self.food_y is None:
            raise ValueError("Dataset not fitted yet. Call fit_normalization_and_labels first.")
        return len(self.food_y)

    def __getitem__(self, idx):
        if self.X is None:
            raise ValueError("Dataset not fitted yet. Call fit_normalization_and_labels first.")
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.food_y[idx], dtype=torch.long),
            torch.tensor(self.pres_raw[idx], dtype=torch.float32),
            torch.tensor(self.conc_log[idx], dtype=torch.float32),
        )
