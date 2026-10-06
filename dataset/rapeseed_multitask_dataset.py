import os

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from dataset.preprocessing import preprocess_spectra


class RapeseedMultiTaskDataset(Dataset):
    """Joint (tissue type, N_content, C_content, N regime) dataset for the
    4-head CSSE model (model/mt_smartnir_model.py: RapeseedCSSEModel).

    Like Mango/Grainit, there is no detection head: N_content/C_content are
    always-measured tissue properties, not analytes that can be
    "undetected" (see prepare_rapeseed_dataset.py). A per-target outlier
    flag nulls some N_content/C_content rows to -1 independently, so each
    carries its own validity mask (same pattern as GrainitMultiTaskDataset).

    N regime (N-/N+, the experimental N-fertilization treatment label) is
    used as the "quality" classification head instead of a threshold on
    N_content: there is no single %DM cutoff that is valid across tissue
    types (N_content differs ~4x between e.g. Flowers and Pods) or growth
    stages (the agronomic standard is a biomass-dependent critical-N-dilution
    curve, not a fixed number, and this dataset has no biomass column) --
    N regime is the actual ground-truth label the INRAE study used, so it is
    learned directly as its own classification task rather than derived
    post-hoc from a guessed threshold.
    """

    def __init__(self, data_filepath: str):
        df = pd.read_csv(data_filepath)
        X_raw = df[[c for c in df.columns if c.startswith("w_")]].values.astype(np.float32)
        tissue_raw = df["category"].values.ravel()
        n_content_raw = df["N_content"].values.astype(np.float32)
        c_content_raw = df["C_content"].values.astype(np.float32)
        n_regime_raw = df["N regime"].values.ravel()

        X_raw, keep_mask = preprocess_spectra(X_raw)
        n_dropped = (~keep_mask).sum()
        if n_dropped:
            print(f"[RapeseedMultiTaskDataset] dropped {n_dropped} outlier spectra out of {len(keep_mask)}")

        self.X_raw = X_raw
        self.tissue_raw = tissue_raw[keep_mask]
        self.n_content_raw = n_content_raw[keep_mask]
        self.c_content_raw = c_content_raw[keep_mask]
        self.n_regime_raw = n_regime_raw[keep_mask]
        self.n_content_valid = (self.n_content_raw != -1).astype(np.float32)
        self.c_content_valid = (self.c_content_raw != -1).astype(np.float32)
        self.signal_length = X_raw.shape[1]

        self.mean = None
        self.std = None
        self.label_encoder = None
        self.regime_encoder = None
        self.mean_n = None
        self.std_n = None
        self.mean_c = None
        self.std_c = None
        self.X = None
        self.food_y = None
        self.n_content_y = None
        self.c_content_y = None
        self.regime_y = None
        self.n_classes = None

    def fit_normalization_and_labels(self, train_indices, save_dir=None):
        X_train = self.X_raw[train_indices]
        self.mean = X_train.mean(axis=0, keepdims=True)
        self.std = X_train.std(axis=0, keepdims=True) + 1e-8
        self.X = (self.X_raw - self.mean) / self.std

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.tissue_raw[train_indices])
        self.food_y = self.label_encoder.transform(self.tissue_raw)
        self.n_classes = len(self.label_encoder.classes_)

        self.regime_encoder = LabelEncoder()
        self.regime_encoder.fit(self.n_regime_raw[train_indices])
        self.regime_y = self.regime_encoder.transform(self.n_regime_raw)

        # Invalid (-1 sentinel) rows get a finite dummy (0.0) before log1p so
        # the whole array stays finite; the per-target mask zeroes their
        # loss contribution out regardless of this placeholder's value.
        n_filled = np.where(self.n_content_valid == 1, self.n_content_raw, 0.0)
        c_filled = np.where(self.c_content_valid == 1, self.c_content_raw, 0.0)

        n_log = np.log1p(n_filled)
        train_valid_n = train_indices[self.n_content_valid[train_indices] == 1]
        self.mean_n = float(n_log[train_valid_n].mean())
        self.std_n = float(n_log[train_valid_n].std() + 1e-8)
        self.n_content_y = (n_log - self.mean_n) / self.std_n

        c_log = np.log1p(c_filled)
        train_valid_c = train_indices[self.c_content_valid[train_indices] == 1]
        self.mean_c = float(c_log[train_valid_c].mean())
        self.std_c = float(c_log[train_valid_c].std() + 1e-8)
        self.c_content_y = (c_log - self.mean_c) / self.std_c

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            np.savez(os.path.join(save_dir, "stats.npz"), mean=self.mean, std=self.std,
                     mean_n=self.mean_n, std_n=self.std_n, mean_c=self.mean_c, std_c=self.std_c)
            joblib.dump(self.label_encoder, os.path.join(save_dir, "label_encoder.pkl"))
            joblib.dump(self.regime_encoder, os.path.join(save_dir, "regime_encoder.pkl"))

    def inverse_transform_n(self, y_normalized):
        return np.expm1(y_normalized * self.std_n + self.mean_n)

    def inverse_transform_c(self, y_normalized):
        return np.expm1(y_normalized * self.std_c + self.mean_c)

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
            torch.tensor(self.n_content_y[idx], dtype=torch.float32),
            torch.tensor(self.n_content_valid[idx], dtype=torch.float32),
            torch.tensor(self.c_content_y[idx], dtype=torch.float32),
            torch.tensor(self.c_content_valid[idx], dtype=torch.float32),
            torch.tensor(self.regime_y[idx], dtype=torch.long),
        )
