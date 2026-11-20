import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dataset.preprocessing import preprocess_spectra


class JointRegressionNIRSDataset(Dataset):
    """Like RegressionNIRSDataset, but keeps K target columns jointly (only
    rows valid -- not -1 -- for *every* target), for training a model with
    n_outputs=K that predicts all targets simultaneously from one shared
    backbone (e.g. XSpecMamba's actual reported headline numbers, which come
    from joint multi-target training, not the per-substance-independent
    convention the rest of this project's Stage 2 baselines use).
    """

    def __init__(self, data_filepath: str, target_columns: list[str], apply_savgol_snv: bool = True):
        super().__init__()
        df = pd.read_csv(data_filepath)

        for c in target_columns:
            if c not in df.columns:
                raise ValueError(f"Target column '{c}' not found in the dataset.")

        valid = np.all([df[c].values != -1 for c in target_columns], axis=0)
        if valid.sum() == 0:
            raise ValueError("No rows valid for all target columns.")
        df = df[valid].reset_index(drop=True)

        self.target_columns = target_columns
        X_raw = df[[wl for wl in df.columns if "w_" in wl]].values.astype(np.float32)
        y_raw = df[target_columns].values.astype(np.float32)

        X_raw, keep_mask = preprocess_spectra(X_raw, apply_savgol_snv=apply_savgol_snv)
        n_dropped = (~keep_mask).sum()
        if n_dropped:
            print(f"[JointRegressionNIRSDataset] dropped {n_dropped} outlier spectra out of {len(keep_mask)}")
        self.X_raw = X_raw
        self.y_raw = y_raw[keep_mask]

        self.mean_X = None
        self.std_X = None
        self.mean_y = None  # (K,) per-target log1p-space mean
        self.std_y = None
        self.X = None
        self.y = None

    def fit_normalization(self, train_indices, save_dir=None, apply_zscore_X: bool = True):
        """apply_zscore_X=False: see RegressionNIRSDataset.fit_normalization
        (dataset/regression_dataset.py) for why."""
        X_train = self.X_raw[train_indices]
        if apply_zscore_X:
            self.mean_X = X_train.mean(axis=0, keepdims=True)
            self.std_X = X_train.std(axis=0, keepdims=True) + 1e-8
            self.X = (self.X_raw - self.mean_X) / self.std_X
        else:
            self.mean_X = np.zeros((1, self.X_raw.shape[1]), dtype=np.float32)
            self.std_X = np.ones((1, self.X_raw.shape[1]), dtype=np.float32)
            self.X = self.X_raw

        y_log = np.log1p(self.y_raw)  # (N, K); safe since -1 rows are excluded, all targets > 0
        y_train_log = y_log[train_indices]
        self.mean_y = y_train_log.mean(axis=0, keepdims=True)  # (1, K)
        self.std_y = y_train_log.std(axis=0, keepdims=True) + 1e-8
        self.y = (y_log - self.mean_y) / self.std_y

        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            np.savez(os.path.join(save_dir, "stats.npz"),
                     mean_X=self.mean_X, std_X=self.std_X, mean_y=self.mean_y, std_y=self.std_y)

    def inverse_transform_y(self, y_normalized):
        """y_normalized: (N, K) -> original units, (N, K)."""
        return np.expm1(y_normalized * self.std_y + self.mean_y)

    def __len__(self):
        if self.y is None:
            raise ValueError("Dataset not fitted yet. Call fit_normalization first.")
        return len(self.y)

    def __getitem__(self, idx):
        if self.X is None or self.y is None:
            raise ValueError("Dataset not fitted yet. Call fit_normalization first.")
        spectrum = torch.tensor(self.X[idx], dtype=torch.float32)
        target = torch.tensor(self.y[idx], dtype=torch.float32)
        return spectrum, target
