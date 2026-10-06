import os

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from dataset.preprocessing import preprocess_spectra


class MangoMultiTaskDataset(Dataset):
    """Joint (cultivar, dry_matter) dataset for the 2-head CSSE model
    (model/mt_smartnir_model.py). Unlike RegressionNIRSDataset, no rows are
    filtered on the target: dry_matter is always measured for every sample
    (no presence/absence concept), so there is no detection task here.
    """

    def __init__(self, data_filepath: str, target_column: str = "dry_matter"):
        df = pd.read_csv(data_filepath)
        X_raw = df[[c for c in df.columns if c.startswith("w_")]].values.astype(np.float32)
        cultivar_raw = df["category"].values.ravel()
        y_raw = df[target_column].values.astype(np.float32)

        X_raw, keep_mask = preprocess_spectra(X_raw)
        n_dropped = (~keep_mask).sum()
        if n_dropped:
            print(f"[MangoMultiTaskDataset] dropped {n_dropped} outlier spectra out of {len(keep_mask)}")

        self.X_raw = X_raw
        self.cultivar_raw = cultivar_raw[keep_mask]
        self.y_raw = y_raw[keep_mask]
        self.signal_length = X_raw.shape[1]

        self.mean = None
        self.std = None
        self.label_encoder = None
        self.mean_y = None
        self.std_y = None
        self.X = None
        self.food_y = None
        self.reg_y = None
        self.n_classes = None

    def fit_normalization_and_labels(self, train_indices, save_dir=None):
        X_train = self.X_raw[train_indices]
        self.mean = X_train.mean(axis=0, keepdims=True)
        self.std = X_train.std(axis=0, keepdims=True) + 1e-8
        self.X = (self.X_raw - self.mean) / self.std

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.cultivar_raw[train_indices])
        self.food_y = self.label_encoder.transform(self.cultivar_raw)
        self.n_classes = len(self.label_encoder.classes_)

        # Same log1p + z-score convention as RegressionNIRSDataset.
        y_log = np.log1p(self.y_raw)
        y_train_log = y_log[train_indices]
        self.mean_y = float(y_train_log.mean())
        self.std_y = float(y_train_log.std() + 1e-8)
        self.reg_y = (y_log - self.mean_y) / self.std_y

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            np.savez(os.path.join(save_dir, "stats.npz"), mean=self.mean, std=self.std,
                     mean_y=self.mean_y, std_y=self.std_y)
            joblib.dump(self.label_encoder, os.path.join(save_dir, "label_encoder.pkl"))

    def inverse_transform_y(self, y_normalized):
        """Undo z-score + log1p to get back to original dry-matter units."""
        return np.expm1(y_normalized * self.std_y + self.mean_y)

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
            torch.tensor(self.reg_y[idx], dtype=torch.float32),
        )
