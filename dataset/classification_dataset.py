import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import LabelEncoder
import joblib

class ClassificationNIRSDataset(Dataset):
    def __init__(self, data_filepath: str, mode="train"):
        super().__init__()
        spectrum_filepath = os.path.join(data_filepath, "spectrum.csv")
        label_filepath = os.path.join(data_filepath, "label.csv")

        self.X = pd.read_csv(spectrum_filepath, index_col=0).values.astype(np.float32)  
        self.y = pd.read_csv(label_filepath, index_col=0).values

        if mode == "train":
            self.mean = self.X.mean(axis=0, keepdims=True)  
            self.std = self.X.std(axis=0, keepdims=True) + 1e-8
            self.X = (self.X - self.mean) / self.std
            
            stats_filepath = os.path.join(data_filepath, "stats.npz")
            np.savez(stats_filepath, mean=self.mean, std=self.std)

            label_encoder = LabelEncoder()
            self.y = label_encoder.fit_transform(self.y.ravel())

            label_encoder_filepath = os.path.join(data_filepath, "label_encoder.pkl")
            joblib.dump(label_encoder, label_encoder_filepath)

        else:
            stats_filepath = os.path.join(data_filepath.replace("val", "train"), "stats.npz")
            stats = np.load(stats_filepath)

            self.mean, self.std = stats["mean"], stats["std"]
            self.X = (self.X - self.mean) / self.std

            label_encoder_filepath = os.path.join(data_filepath.replace("val", "train"), "label_encoder.pkl")
            label_encoder = joblib.load(label_encoder_filepath)
            self.y = label_encoder.transform(self.y.ravel())

        self.n_classes = len(set(self.y))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        spectrum = self.X[idx]
        label = self.y[idx]

        spectrum = torch.tensor(spectrum, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return spectrum, label
