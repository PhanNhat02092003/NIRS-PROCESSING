import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import LabelEncoder
import joblib

class RegressionNIRSDataset(Dataset):
    def __init__(self, data_filepath: str, mode="train"):
        super().__init__()
        spectrum_filepath = os.path.join(data_filepath, "spectrum.csv")
        label_filepath = os.path.join(data_filepath, "label.csv")

        self.X = pd.read_csv(spectrum_filepath, index_col=0).values.astype(np.float32)  
        self.y = pd.read_csv(label_filepath, index_col=0).values.astype(np.float32) 
        self.y[self.y == -1] = 0

        if mode == "train":
            self.X_mean = self.X.mean(axis=0, keepdims=True)  
            self.X_std = self.X.std(axis=0, keepdims=True) + 1e-8
            self.X = (self.X - self.X_mean) / self.X_std
            
            x_stats_filepath = os.path.join(data_filepath, "x_stats.npz")
            np.savez(x_stats_filepath, mean=self.X_mean, std=self.X_std)

            self.y_mean = self.y.mean(axis=0, keepdims=True)  
            self.y_std = self.y.std(axis=0, keepdims=True) + 1e-8
            self.y = (self.y - self.y_mean) / self.y_std

            y_stats_filepath = os.path.join(data_filepath, "y_stats.npz")
            np.savez(y_stats_filepath, mean=self.y_mean, std=self.y_std)

        else:
            X_stats_filepath = os.path.join(data_filepath.replace("val", "train"), "x_stats.npz")
            X_stats = np.load(X_stats_filepath)

            self.X_mean, self.X_std = X_stats["mean"], X_stats["std"]
            self.X = (self.X - self.X_mean) / self.X_std

            y_stats_filepath = os.path.join(data_filepath.replace("val", "train"), "y_stats.npz")
            y_stats = np.load(y_stats_filepath)

            self.y_mean, self.y_std = y_stats["mean"], y_stats["std"]
            self.y = (self.y - self.y_mean) / self.y_std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        spectrum = self.X[idx]
        label = self.y[idx]

        spectrum = torch.tensor(spectrum, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        return spectrum, label
