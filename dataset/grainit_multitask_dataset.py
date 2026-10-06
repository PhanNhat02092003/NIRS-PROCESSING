import os

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from dataset.preprocessing import preprocess_spectra


class GrainitMultiTaskDataset(Dataset):
    """Joint (cereal type, Moisture, Protein) dataset for the 3-head CSSE
    model (model/mt_smartnir_model.py: GrainitCSSEModel). Like Mango, there
    is no detection head: Moisture/Protein are always-measured grain
    properties, not analytes that can be "undetected" (see
    prepare_grainit_dataset.py). Unlike Mango, a handful of rows are missing
    one of the two targets (the raw-file 0%-placeholder rows nulled to -1 by
    prepare_grainit_dataset.py), so each target carries its own validity
    mask and the engine must mask those rows out of that target's loss/metrics.
    """

    def __init__(self, data_filepath: str):
        df = pd.read_csv(data_filepath)
        X_raw = df[[c for c in df.columns if c.startswith("w_")]].values.astype(np.float32)
        cultivar_raw = df["category"].values.ravel()
        moisture_raw = df["Moisture"].values.astype(np.float32)
        protein_raw = df["Protein"].values.astype(np.float32)

        X_raw, keep_mask = preprocess_spectra(X_raw)
        n_dropped = (~keep_mask).sum()
        if n_dropped:
            print(f"[GrainitMultiTaskDataset] dropped {n_dropped} outlier spectra out of {len(keep_mask)}")

        self.X_raw = X_raw
        self.cultivar_raw = cultivar_raw[keep_mask]
        self.moisture_raw = moisture_raw[keep_mask]
        self.protein_raw = protein_raw[keep_mask]
        self.moisture_valid = (self.moisture_raw != -1).astype(np.float32)
        self.protein_valid = (self.protein_raw != -1).astype(np.float32)
        self.signal_length = X_raw.shape[1]

        self.mean = None
        self.std = None
        self.label_encoder = None
        self.mean_moisture = None
        self.std_moisture = None
        self.mean_protein = None
        self.std_protein = None
        self.X = None
        self.food_y = None
        self.moisture_y = None
        self.protein_y = None
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

        # Invalid (-1 sentinel) rows get a finite dummy (0.0) before log1p so
        # the whole array stays finite; the per-target mask zeroes their
        # loss contribution out regardless of this placeholder's value.
        moisture_filled = np.where(self.moisture_valid == 1, self.moisture_raw, 0.0)
        protein_filled = np.where(self.protein_valid == 1, self.protein_raw, 0.0)

        moisture_log = np.log1p(moisture_filled)
        train_valid_moisture = train_indices[self.moisture_valid[train_indices] == 1]
        self.mean_moisture = float(moisture_log[train_valid_moisture].mean())
        self.std_moisture = float(moisture_log[train_valid_moisture].std() + 1e-8)
        self.moisture_y = (moisture_log - self.mean_moisture) / self.std_moisture

        protein_log = np.log1p(protein_filled)
        train_valid_protein = train_indices[self.protein_valid[train_indices] == 1]
        self.mean_protein = float(protein_log[train_valid_protein].mean())
        self.std_protein = float(protein_log[train_valid_protein].std() + 1e-8)
        self.protein_y = (protein_log - self.mean_protein) / self.std_protein

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            np.savez(os.path.join(save_dir, "stats.npz"), mean=self.mean, std=self.std,
                     mean_moisture=self.mean_moisture, std_moisture=self.std_moisture,
                     mean_protein=self.mean_protein, std_protein=self.std_protein)
            joblib.dump(self.label_encoder, os.path.join(save_dir, "label_encoder.pkl"))

    def inverse_transform_moisture(self, y_normalized):
        return np.expm1(y_normalized * self.std_moisture + self.mean_moisture)

    def inverse_transform_protein(self, y_normalized):
        return np.expm1(y_normalized * self.std_protein + self.mean_protein)

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
            torch.tensor(self.moisture_y[idx], dtype=torch.float32),
            torch.tensor(self.moisture_valid[idx], dtype=torch.float32),
            torch.tensor(self.protein_y[idx], dtype=torch.float32),
            torch.tensor(self.protein_valid[idx], dtype=torch.float32),
        )
