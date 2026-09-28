import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os

from dataset.preprocessing import preprocess_spectra

class RegressionNIRSDataset(Dataset):
    def __init__(self, data_filepath: str, target_column: str, apply_savgol_snv: bool = True):
        super().__init__()
        df = pd.read_csv(data_filepath)
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in the dataset.")
        
        y_col = df[target_column]
        
        if (y_col == -1).all():
            raise ValueError(f"All values in target column '{target_column}' are -1.")
        
        non_neg = y_col[y_col != -1]
        
        if len(non_neg.unique()) <= 1:
            raise ValueError(f"Target column '{target_column}' has only one unique value different from -1 or none.")
        
        df = df[df[target_column] != -1]

        X_raw = df[[wl for wl in df.columns if "w_" in wl]].values.astype(np.float32)
        y_raw = df[target_column].values.astype(np.float32)

        X_raw, keep_mask = preprocess_spectra(X_raw, apply_savgol_snv=apply_savgol_snv)
        n_dropped = (~keep_mask).sum()
        if n_dropped:
            print(f"[RegressionNIRSDataset:{target_column}] dropped {n_dropped} outlier spectra out of {len(keep_mask)}")
        self.X_raw = X_raw
        self.y_raw = y_raw[keep_mask]

        self.mean_X = None
        self.std_X = None
        self.mean_y = None
        self.std_y = None
        self.X = None
        self.y = None

    def fit_normalization(self, train_indices, save_dir=None, apply_zscore_X: bool = True):
        """apply_zscore_X=False skips the per-band z-score of X (self.X
        becomes self.X_raw unchanged), for callers whose own downstream
        pipeline does its own input normalization -- e.g. XSpecMamba's
        spectrum_to_image, which does a per-sample min-max rescale to
        [-1, 1] that's the reference paper's *only* stated preprocessing
        step; stacking a train-fold band-wise z-score in front of that
        isn't what the paper describes. mean_X/std_X are still computed
        and saved (as the identity mean=0/std=1 vectors) so downstream code
        that expects them to exist keeps working unchanged."""
        X_train = self.X_raw[train_indices]
        if apply_zscore_X:
            self.mean_X = X_train.mean(axis=0, keepdims=True)
            self.std_X = X_train.std(axis=0, keepdims=True) + 1e-8
            self.X = (self.X_raw - self.mean_X) / self.std_X
        else:
            self.mean_X = np.zeros((1, self.X_raw.shape[1]), dtype=np.float32)
            self.std_X = np.ones((1, self.X_raw.shape[1]), dtype=np.float32)
            self.X = self.X_raw

        # concentrations are right-skewed (few large values dominate a plain
        # MSE); log1p first so the network is fit in a roughly symmetric
        # space, then z-score as before. Safe since -1 rows are filtered out
        # in __init__, so y_raw is always > 0 here.
        y_log = np.log1p(self.y_raw)
        y_train_log = y_log[train_indices]
        self.mean_y = y_train_log.mean()
        self.std_y = y_train_log.std() + 1e-8

        self.y = (y_log - self.mean_y) / self.std_y

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            stats_filepath = os.path.join(save_dir, "stats.npz")
            np.savez(stats_filepath, mean_X=self.mean_X, std_X=self.std_X, mean_y=self.mean_y, std_y=self.std_y)

    def inverse_transform_y(self, y_normalized):
        """Undo z-score + log1p to get back to original concentration units."""
        return np.expm1(y_normalized * self.std_y + self.mean_y)

    def __len__(self):
        if self.y is None:
            raise ValueError("Dataset not fitted yet. Call fit_normalization first.")
        return len(self.y)

    def __getitem__(self, idx):
        if self.X is None or self.y is None:
            raise ValueError("Dataset not fitted yet. Call fit_normalization first.")
        spectrum = self.X[idx]
        target = self.y[idx]

        spectrum = torch.tensor(spectrum, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)

        return spectrum, target