import os

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from dataset.preprocessing import preprocess_spectra


class QualityGradeDataset(Dataset):
    """Same shape/convention as ClassificationNIRSDataset (dataset/classification_dataset.py),
    but the label is a ground-truth quality grade computed by `label_fn` from
    the raw dataframe (e.g. Mango's per-cultivar dry-matter threshold, or
    Rapeseed's N regime column) instead of the `category` column directly --
    used to benchmark SMART-NIR/GuidedDCNet on the same quality-grade task
    the CSSE-MT multi-task model's "quality" head is scored on.

    `label_fn(df) -> np.ndarray` returns one label per row, with None/NaN for
    rows where the grade is undefined (e.g. a missing required reading) --
    those rows are dropped, same as preprocess_spectra's outlier rows.
    """

    def __init__(self, data_filepath: str, label_fn):
        df = pd.read_csv(data_filepath)
        X_raw = df[[c for c in df.columns if c.startswith("w_")]].values.astype(np.float32)
        y_raw = np.asarray(label_fn(df), dtype=object)

        label_valid = pd.notna(y_raw)
        n_undefined = int((~label_valid).sum())
        if n_undefined:
            print(f"[QualityGradeDataset] dropped {n_undefined} row(s) with undefined quality grade "
                  f"out of {len(label_valid)}")
        X_raw, y_raw = X_raw[label_valid], y_raw[label_valid]

        X_raw, keep_mask = preprocess_spectra(X_raw)
        n_dropped = (~keep_mask).sum()
        if n_dropped:
            print(f"[QualityGradeDataset] dropped {n_dropped} outlier spectra out of {len(keep_mask)}")

        self.X_raw = X_raw
        self.y_raw = y_raw[keep_mask].astype(str)

        self.mean = None
        self.std = None
        self.label_encoder = None
        self.X = None
        self.y = None
        self.n_classes = None
        self.signal_length = X_raw.shape[1]

    def fit_normalization_and_labels(self, train_indices, save_dir=None):
        X_train = self.X_raw[train_indices]
        self.mean = X_train.mean(axis=0, keepdims=True)
        self.std = X_train.std(axis=0, keepdims=True) + 1e-8
        self.X = (self.X_raw - self.mean) / self.std

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.y_raw[train_indices])
        self.y = self.label_encoder.transform(self.y_raw)
        self.n_classes = len(self.label_encoder.classes_)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            np.savez(os.path.join(save_dir, "stats.npz"), mean=self.mean, std=self.std)
            joblib.dump(self.label_encoder, os.path.join(save_dir, "label_encoder.pkl"))

    def __len__(self):
        if self.y is None:
            raise ValueError("Dataset not fitted yet. Call fit_normalization_and_labels first.")
        return len(self.y)

    def __getitem__(self, idx):
        if self.X is None or self.y is None:
            raise ValueError("Dataset not fitted yet. Call fit_normalization_and_labels first.")
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.long),
        )
