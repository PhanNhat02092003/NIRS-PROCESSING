import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
import joblib

from dataset.preprocessing import preprocess_spectra

class ClassificationNIRSDataset(Dataset):
    def __init__(self, data_filepath: str):
        super().__init__()
        df = pd.read_csv(data_filepath)
        X_raw = df[[wl for wl in df.columns if "w_" in wl]].values.astype(np.float32)
        y_raw = df['category'].values.ravel()

        X_raw, keep_mask = preprocess_spectra(X_raw)
        n_dropped = (~keep_mask).sum()
        if n_dropped:
            print(f"[ClassificationNIRSDataset] dropped {n_dropped} outlier spectra out of {len(keep_mask)}")
        self.X_raw = X_raw
        self.y_raw = y_raw[keep_mask]

        self.mean = None
        self.std = None
        self.label_encoder = None
        self.X = None
        self.y = None
        self.n_classes = None
        self.signal_length = len([wl for wl in df.columns if "w_" in wl])

    def fit_normalization_and_labels(self, train_indices, save_dir=None):
        X_train = self.X_raw[train_indices]
        self.mean = X_train.mean(axis=0, keepdims=True)
        self.std = X_train.std(axis=0, keepdims=True) + 1e-8

        self.X = (self.X_raw - self.mean) / self.std

        self.label_encoder = LabelEncoder()
        y_train = self.y_raw[train_indices]
        self.label_encoder.fit(y_train)

        self.y = self.label_encoder.transform(self.y_raw)

        self.n_classes = len(self.label_encoder.classes_)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            stats_filepath = os.path.join(save_dir, "stats.npz")
            np.savez(stats_filepath, mean=self.mean, std=self.std)
            label_encoder_filepath = os.path.join(save_dir, "label_encoder.pkl")
            joblib.dump(self.label_encoder, label_encoder_filepath)

    def __len__(self):
        if self.y is None:
            raise ValueError("Dataset not fitted yet. Call fit_normalization_and_labels first.")
        return len(self.y)

    def __getitem__(self, idx):
        if self.X is None or self.y is None:
            raise ValueError("Dataset not fitted yet. Call fit_normalization_and_labels first.")
        spectrum = self.X[idx]
        label = self.y[idx]

        spectrum = torch.tensor(spectrum, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return spectrum, label