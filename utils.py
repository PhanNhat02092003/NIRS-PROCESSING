import os
import joblib

from model.classification_model import *
from model.regression_model import *
from dataset.preprocessing import savgol_smooth, snv

import numpy as np
import xgboost as xgb
from pydantic import BaseModel
from typing import List
from torch.nn.functional import softmax
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class NirsRequest(BaseModel):
    spectrum: List[List[float]]
    machine: Literal['FLAMENIR', 'OCEANFX']

def infer_category_classification(spectra: np.ndarray, machine: str):
    task = "category_classification"
    k_folds = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load configuration from fold 1 (assuming signal_len consistent)
    save_fold_dir = f"data/{task}/{machine}/fold_1"
    stats_path = os.path.join(save_fold_dir, "stats.npz")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Stats file not found for {machine} fold 1")

    stats = np.load(stats_path)
    signal_len = stats['mean'].shape[1]

    # Check input shape
    if spectra.shape[1] != signal_len:
        raise ValueError(f"Input spectrum length {spectra.shape[1]} does not match expected {signal_len}")

    # Match the Savitzky-Golay smoothing + SNV scatter correction applied to
    # spectra before the training normalization stats were computed
    # (dataset/preprocessing.py) -- skipping this causes train/serve skew.
    spectra = snv(savgol_smooth(spectra)).astype(np.float32)

    # Load means, stds, models, and label_encoders for all folds
    means = []
    stds = []
    models = []
    label_encoders = []
    num_classes_list = []
    for fold in range(1, k_folds + 1):
        save_fold_dir = f"data/{task}/{machine}/fold_{fold}"
        stats = np.load(os.path.join(save_fold_dir, "stats.npz"))
        means.append(stats['mean'])
        stds.append(stats['std'])

        labels_path = os.path.join(save_fold_dir, "label_encoder.pkl")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Label encoder file not found for {machine} fold {fold}")
        label_encoder = joblib.load(labels_path)
        label_encoders.append(label_encoder)
        num_classes_list.append(len(label_encoder.classes_))

        cfg = SmartNIRClassificationConfig(
            signal_len=signal_len,
            out_ch_per_branch=64,
            d_model=128,
            depth=3,
            n_heads=4,
            classifier="kan",
            num_classes=num_classes_list[-1]
        )

        model = SMARTNIRClassifier(cfg).to(device)
        model_path = f"checkpoint/{task}/{machine}/checkpoint_fold{fold}.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found for {machine} fold {fold}")

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models.append(model)

    # Inference
    spectra_tensor = torch.from_numpy(spectra).float().to(device)  # (batch_size, signal_len)
    all_labels = []

    for mean, std, model, label_encoder in zip(means, stds, models, label_encoders):
        mean_t = torch.from_numpy(mean).float().to(device)
        std_t = torch.from_numpy(std).float().to(device)
        norm_x = (spectra_tensor - mean_t) / std_t
        with torch.no_grad():
            outputs = model(norm_x)
            probs = softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            fold_labels = label_encoder.inverse_transform(preds)
        all_labels.append(fold_labels)

    # Voting for predictions on labels
    all_labels = np.array(all_labels)  # (k_folds, batch_size)
    voted_preds = []
    for i in range(spectra.shape[0]):
        votes = Counter(all_labels[:, i])
        most_common = votes.most_common(1)
        voted_preds.append(most_common[0][0])

    return voted_preds

def infer_substances_detection(spectra: np.ndarray, machine: str):
    task = "substance_regression"
    substances = [
        'Thiamethoxam', 'Permethrin', 'Metalaxyl', 'Azoxystrobin',
        'Imidaclopird', 'Difenoconazole', 'Cypermethrin', 'Cyhalothrin',
        'Chlorantraniliprol', 'Chlopyrifos Methyl', 'Emamectin benzoate',
        'Chlorothalonil', 'Triadimefon', 'Cyantraniliprole', 'Flutolanil',
        'Indoxacarb', 'Abamectin', 'Propamocarb.HCL', 'Chlothianidin'
    ]
    k_folds = 5

    # Same preprocessing as training (dataset/preprocessing.py) to avoid
    # train/serve skew.
    spectra = snv(savgol_smooth(spectra)).astype(np.float32)

    results = []
    for i in range(spectra.shape[0]):
        sample = spectra[i:i+1]
        detected = []
        for substance in substances:
            save_scaler_dir = f"data/{task}/stage1/{machine}/{substance}"
            save_best_model_dir = f"checkpoint/{task}/stage1/{machine}/{substance}"
            if not os.path.exists(save_scaler_dir) or not os.path.exists(save_best_model_dir):
                continue

            all_votes = []
            for fold in range(1, k_folds + 1):
                scaler_path = os.path.join(save_scaler_dir, f"{substance}_fold_{fold}_scaler.pkl")
                if not os.path.exists(scaler_path):
                    continue
                scaler = joblib.load(scaler_path)
                sample_scaled = scaler.transform(sample)

                model_path = os.path.join(save_best_model_dir, f"{substance}_fold_{fold}.json")
                if not os.path.exists(model_path):
                    continue
                model = xgb.Booster()
                model.load_model(model_path)

                dsample = xgb.DMatrix(sample_scaled)
                prob = model.predict(dsample)[0]
                vote = 1 if prob > 0.5 else 0
                all_votes.append(vote)

            if all_votes:
                num_positive = sum(all_votes)
                if num_positive > len(all_votes) / 2:
                    detected.append(substance)

        results.append(detected)

    return results

def infer_substances_prediction(spectra: np.ndarray, machine: str, detected_list: list[list[str]]):
    task = "substance_regression"
    k_folds = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Same preprocessing as training (dataset/preprocessing.py) to avoid
    # train/serve skew.
    spectra = snv(savgol_smooth(spectra)).astype(np.float32)

    results = []
    for i in range(spectra.shape[0]):
        sample = spectra[i:i+1]  # (1, signal_len)
        detected = detected_list[i]
        concentrations = {substance: None for substance in detected}
        for substance in detected:
            save_fold_dir_base = f"data/{task}/stage2/{machine}/{substance}"
            save_best_model_dir = f"checkpoint/{task}/stage2/{machine}/{substance}"
            if not os.path.exists(save_fold_dir_base) or not os.path.exists(save_best_model_dir):
                continue  # Remains None

            all_preds = []
            signal_len = None
            for fold in range(1, k_folds + 1):
                save_fold_dir = f"{save_fold_dir_base}/fold_{fold}"
                stats_path = os.path.join(save_fold_dir, "stats.npz")
                if not os.path.exists(stats_path):
                    continue
                stats = np.load(stats_path)
                mean_X = stats['mean_X']
                std_X = stats['std_X']
                mean_y = stats['mean_y']
                std_y = stats['std_y']

                if signal_len is None:
                    signal_len = mean_X.shape[1]
                elif signal_len != mean_X.shape[1]:
                    continue  # Inconsistent signal length

                sample_norm = (sample - mean_X) / std_X
                sample_tensor = torch.from_numpy(sample_norm).float().to(device)

                cfg = SmartNIRRegressionConfig(
                    signal_len=signal_len,
                    out_ch_per_branch=64,
                    d_model=128,
                    depth=3,
                    n_heads=4,
                    classifier="kan",
                    num_targets=1,
                    kan_basis=8
                )

                model = SMARTNIRRegressor(cfg).to(device)
                model_path = f"{save_best_model_dir}/checkpoint_fold{fold}.pth"
                if not os.path.exists(model_path):
                    continue

                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()

                with torch.no_grad():
                    output = model(sample_tensor).squeeze(-1).cpu().numpy()[0]

                # Denormalize
                pred = output * std_y + mean_y
                all_preds.append(pred)

            if all_preds:
                avg_pred = np.mean(all_preds)
                concentrations[substance] = float(avg_pred)

        results.append(concentrations)

    return results
