"""Evaluates the SMART-NIR food classifier retrained by
food_classification_downsampled.py (task=category_classification_downsampled,
trained on {machine}/ALL_downsampled.csv with StratifiedGroupKFold on
_group_id) against the external held-out test-data set
(DATASET_ROOT/test-data/{Machine}.csv), to see whether removing batch-level
leakage + redundant near-identical rows produces a classifier that
generalizes better than the original (StratifiedKFold on raw ALL.csv,
93.98%/96.01% CV but only 74.25%/75.63% on test-data -- see
evaluate_deployed_pipeline_testdata.py and evaluate_deployed_pipeline_smartnir_food.py).

Only the food-classification step is swapped in here; ground truth / test
loading is reused as-is from evaluate_deployed_pipeline_testdata.py.

Usage: python3 evaluate_downsampled_food_classifier.py
"""

from collections import Counter

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score

from evaluate_deployed_pipeline_testdata import load_test_set
import deploy_utils
from model.classification_model import SMARTNIRClassifier, SmartNIRClassificationConfig

TASK = "category_classification_downsampled"
K_FOLDS = 5


def predict(spectra: np.ndarray, machine: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # deploy_utils._preprocess already does SG+SNV *and* BIN_FACTOR binning
    # per machine -- the same reduction ClassificationNIRSDataset(bin_factor=...)
    # applied at training time, so don't bin again here.
    spectra = deploy_utils._preprocess(spectra, machine)
    spectra_tensor = torch.from_numpy(spectra).float().to(device)

    all_labels = []
    for fold in range(1, K_FOLDS + 1):
        fold_dir = f"data/{TASK}/{machine}/fold_{fold}"
        stats = np.load(f"{fold_dir}/stats.npz")
        mean_t = torch.from_numpy(stats["mean"]).float().to(device)
        std_t = torch.from_numpy(stats["std"]).float().to(device)
        label_encoder = joblib.load(f"{fold_dir}/label_encoder.pkl")

        cfg = SmartNIRClassificationConfig(
            signal_len=spectra.shape[1], out_ch_per_branch=64, d_model=128,
            depth=3, n_heads=4, classifier="kan", num_classes=len(label_encoder.classes_),
        )
        model = SMARTNIRClassifier(cfg).to(device)
        model.load_state_dict(torch.load(
            f"checkpoint/{TASK}/{machine}/checkpoint_fold{fold}.pth", map_location=device))
        model.eval()

        norm_x = (spectra_tensor - mean_t) / std_t
        with torch.no_grad():
            preds = model(norm_x).argmax(dim=1).cpu().numpy()
        all_labels.append(label_encoder.inverse_transform(preds))

    all_labels = np.array(all_labels)
    voted_preds = []
    for i in range(spectra.shape[0]):
        label, _votes = Counter(all_labels[:, i]).most_common(1)[0]
        voted_preds.append(label)
    return voted_preds


def evaluate_machine(machine: str, chunk_size: int = 4000):
    print(f"=== {machine} ===")
    spectra, food_id_true, _presence_true, _severity_true, _safe_true = load_test_set(machine)
    n = len(spectra)
    print(f"n test rows: {n}")

    food_name_pred = []
    for start in range(0, n, chunk_size):
        chunk = spectra[start:start + chunk_size]
        food_name_pred.extend(predict(chunk, machine))
        print(f"  {min(start + chunk_size, n)}/{n} rows done", flush=True)

    food_id_pred = np.array([deploy_utils.FOOD_NAME_TO_ID[name] for name in food_name_pred])
    acc = accuracy_score(food_id_true, food_id_pred) * 100.0
    print(f"Food classification accuracy (downsampled, group-safe CV model): {acc:.2f}%\n")


if __name__ == "__main__":
    for machine in ("FLAMENIR", "OCEANFX"):
        evaluate_machine(machine)
