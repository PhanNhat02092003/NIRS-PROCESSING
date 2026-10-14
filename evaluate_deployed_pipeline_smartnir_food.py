"""Same end-to-end evaluation as evaluate_deployed_pipeline_testdata.py, but
swaps the deployed pipeline's food-classification step from GuidedDCNet
(checkpoint/category_classification_guideddcnet) to the plain SMART-NIR
classifier (checkpoint/category_classification, model/classification_model.py:
SMARTNIRClassifier) -- same 5-fold majority-vote ensembling convention as
deploy_utils.infer_category_classification, just a different backbone/
checkpoint directory.

Stage 1/2 (SmartNIRWithFood) are already SMART-NIR and are left untouched;
this isolates whether the food-classification step's own accuracy (74-76%
with GuidedDCNet on the new test-data, see evaluate_deployed_pipeline_testdata.py)
is corrupting Stage 1's food-conditioning one-hot input enough to explain
Stage 1's very low detection recall there, by trying a different food
classifier and re-running the identical downstream pipeline/metrics.
"""

from collections import Counter

import joblib
import numpy as np
import torch

import deploy_utils
from evaluate_deployed_pipeline_testdata import evaluate_machine
from model.classification_model import SMARTNIRClassifier, SmartNIRClassificationConfig


def infer_category_classification_smartnir(spectra: np.ndarray, machine: str):
    """Plain SMART-NIR classifier, 5-fold ensemble by majority vote
    (checkpoint/category_classification) -- drop-in replacement for
    deploy_utils.infer_category_classification (same signature/return
    shape), swapped in via monkeypatch below."""
    task = "category_classification"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spectra = deploy_utils._preprocess(spectra, machine)
    spectra_tensor = torch.from_numpy(spectra).float().to(device)

    all_labels = []
    for fold in range(1, deploy_utils.K_FOLDS + 1):
        fold_dir = f"data/{task}/{machine}/fold_{fold}"
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
            f"checkpoint/{task}/{machine}/checkpoint_fold{fold}.pth", map_location=device))
        model.eval()

        norm_x = (spectra_tensor - mean_t) / std_t
        with torch.no_grad():
            preds = model(norm_x).argmax(dim=1).cpu().numpy()
        all_labels.append(label_encoder.inverse_transform(preds))

    all_labels = np.array(all_labels)
    voted_preds, confidences = [], []
    for i in range(spectra.shape[0]):
        label, votes = Counter(all_labels[:, i]).most_common(1)[0]
        voted_preds.append(label)
        confidences.append(votes / deploy_utils.K_FOLDS)
    return voted_preds, confidences


# analyze_spectrum looks up infer_category_classification in deploy_utils'
# own module namespace at call time, so patching the name here is enough to
# redirect it without touching infer_substances_detection/_severity.
deploy_utils.infer_category_classification = infer_category_classification_smartnir


if __name__ == "__main__":
    for machine in ("FLAMENIR", "OCEANFX"):
        evaluate_machine(machine)
