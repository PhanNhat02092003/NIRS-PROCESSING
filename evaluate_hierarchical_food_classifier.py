"""Evaluates the 2-stage hierarchical food classifier
(food_classification_hierarchical.py: gate + leafy + nonleafy, task prefix
category_classification_hier_*) end-to-end against the external test-data
set, for direct comparison against the flat downsampled model (73.73%/
73.23%, evaluate_downsampled_food_classifier.py) and the deployed
GuidedDCNet pipeline (74.25%/75.63%, evaluate_deployed_pipeline_testdata.py).

Usage: python3 evaluate_hierarchical_food_classifier.py
"""

from collections import Counter

import joblib
import numpy as np
import torch
from scipy.signal import savgol_filter
from sklearn.metrics import accuracy_score

import deploy_utils
from evaluate_deployed_pipeline_testdata import load_test_set
from model.classification_model import SMARTNIRClassifier, SmartNIRClassificationConfig

K_FOLDS = 5


def preprocess_derivative_no_filter(spectra: np.ndarray) -> np.ndarray:
    """Same transform as dataset.preprocessing.preprocess_spectra_derivative
    (SNV -> SG 1st derivative) but skipping the outlier mask -- the mask
    only drops rows, it never changes the per-row SNV/SG transform of the
    rows that survive, so this is exact for every non-glitch row (same
    reasoning deploy_utils._preprocess already relies on for the plain/
    non-derivative branch)."""
    mean = spectra.mean(axis=1, keepdims=True)
    std = spectra.std(axis=1, keepdims=True) + 1e-8
    X = (spectra - mean) / std
    return savgol_filter(X, window_length=11, polyorder=2, deriv=1, axis=1).astype(np.float32)


def ensemble_predict(spectra_pp: np.ndarray, task: str, machine: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spectra_tensor = torch.from_numpy(spectra_pp).float().to(device)

    all_labels = []
    for fold in range(1, K_FOLDS + 1):
        fold_dir = f"data/{task}/{machine}/fold_{fold}"
        stats = np.load(f"{fold_dir}/stats.npz")
        mean_t = torch.from_numpy(stats["mean"]).float().to(device)
        std_t = torch.from_numpy(stats["std"]).float().to(device)
        label_encoder = joblib.load(f"{fold_dir}/label_encoder.pkl")

        cfg = SmartNIRClassificationConfig(
            signal_len=spectra_pp.shape[1], out_ch_per_branch=64, d_model=128,
            depth=3, n_heads=4, classifier="kan", num_classes=len(label_encoder.classes_),
        )
        model = SMARTNIRClassifier(cfg).to(device)
        model.load_state_dict(torch.load(
            f"checkpoint/{task}/{machine}/checkpoint_fold{fold}.pth", map_location=device))
        model.eval()

        norm_x = (spectra_tensor - mean_t) / std_t
        with torch.no_grad():
            preds = model(norm_x).argmax(dim=1).cpu().numpy()
        all_labels.append(label_encoder.inverse_transform(preds))

    all_labels = np.array(all_labels)
    voted = []
    for i in range(spectra_pp.shape[0]):
        label, _votes = Counter(all_labels[:, i]).most_common(1)[0]
        voted.append(label)
    return np.array(voted)


def predict_hierarchical(spectra: np.ndarray, machine: str):
    plain_pp = deploy_utils._preprocess(spectra, machine)  # SG+SNV(+bin), no outlier filter
    deriv_pp = preprocess_derivative_no_filter(spectra)
    # NOTE: binning for the derivative branch is folded in by the model's
    # own stats.npz width check below (FLAMENIR bin_factor=1 => no-op).
    bin_factor = deploy_utils.BIN_FACTOR[machine]
    if bin_factor > 1:
        n_out = (deriv_pp.shape[1] // bin_factor) // 8 * 8
        deriv_pp = deriv_pp[:, :n_out * bin_factor].reshape(len(deriv_pp), n_out, bin_factor).mean(axis=2)
        deriv_pp = deriv_pp.astype(np.float32)

    is_leafy_pred = ensemble_predict(plain_pp, "category_classification_hier_gate", machine)
    is_leafy_pred = is_leafy_pred == "True"

    final = np.empty(len(spectra), dtype=object)
    if is_leafy_pred.any():
        leafy_out = ensemble_predict(deriv_pp[is_leafy_pred], "category_classification_hier_leafy", machine)
        final[is_leafy_pred] = leafy_out
    if (~is_leafy_pred).any():
        nonleafy_out = ensemble_predict(plain_pp[~is_leafy_pred], "category_classification_hier_nonleafy", machine)
        final[~is_leafy_pred] = nonleafy_out
    return final


def evaluate_machine(machine: str, chunk_size: int = 4000):
    print(f"=== {machine} ===")
    spectra, food_id_true, _presence_true, _severity_true, _safe_true = load_test_set(machine)
    n = len(spectra)
    print(f"n test rows: {n}")

    food_name_pred = []
    for start in range(0, n, chunk_size):
        chunk = spectra[start:start + chunk_size]
        food_name_pred.extend(predict_hierarchical(chunk, machine).tolist())
        print(f"  {min(start + chunk_size, n)}/{n} rows done", flush=True)

    food_id_pred = np.array([deploy_utils.FOOD_NAME_TO_ID[name] for name in food_name_pred])
    acc = accuracy_score(food_id_true, food_id_pred) * 100.0
    print(f"Food classification accuracy (hierarchical gate+leafy+nonleafy): {acc:.2f}%\n")


if __name__ == "__main__":
    for machine in ("FLAMENIR",):
        evaluate_machine(machine)
