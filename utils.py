import os
import json
import joblib

from model.classification_model import *
from model.guideddcnet_model import GuidedDCNet, GuidedDCNetConfig
from model.smartnir_food_model import SmartNIRWithFood, predict_logit_diff, food_prior_shrinkage
from dataset.preprocessing import savgol_smooth, snv

import numpy as np
from pydantic import BaseModel
from typing import List, Optional
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class NirsRequest(BaseModel):
    spectrum: List[List[float]]
    machine: Literal['FLAMENIR', 'OCEANFX']

K_FOLDS = 5

# OCEANFX's 2136-point spectra are averaged in groups of 8 for every deep
# model in the pipeline (GuidedDCNet food classification, SMART-NIR Buoc 1/2);
# FLAMENIR's 128 points are used at full resolution. Must match training.
BIN_FACTOR = {"FLAMENIR": 1, "OCEANFX": 8}

# 9 food categories, in the same order used to build the one-hot input at
# training time (list(food_ids) from DATASET_ROOT/food_ids.json, F01..F09) --
# hardcoded here so serving has no dependency on the external dataset folder.
# The API reports the F01..F09 id (never the Vietnamese name) -- see
# FOOD_ID_LEGEND in app.py for the id -> name table shown in Swagger.
FOOD_NAME_TO_INDEX = {
    "Xà Lách": 0,
    "Cải Bẹ Xanh": 1,
    "Cải Thìa": 2,
    "Mồng Tơi": 3,
    "Cà Chua": 4,
    "Cà Rốt": 5,
    "Dưa Leo": 6,
    "Khổ Qua": 7,
    "Đậu Cove": 8,
}
N_FOOD = len(FOOD_NAME_TO_INDEX)
FOOD_IDS = [f"F{i + 1:02d}" for i in range(N_FOOD)]  # index -> "F01".."F09"
FOOD_NAME_TO_ID = {name: FOOD_IDS[idx] for name, idx in FOOD_NAME_TO_INDEX.items()}

# 19 pesticide substances, in the same order as DATASET_ROOT/pesticide_ids.json
# (P01..P19) -- see CLAUDE.md; not alphabetical. The API reports the P01..P19
# id -- see SUBSTANCE_ID_LEGEND in app.py for the id -> chemical-name table
# shown in Swagger.
SUBSTANCES = [
    'Thiamethoxam', 'Permethrin', 'Metalaxyl', 'Azoxystrobin', 'Difenoconazole',
    'Cypermethrin', 'Cyhalothrin', 'Chlorantraniliprol', 'Emamectin benzoate',
    'Chlorothalonil', 'Triadimefon', 'Cyantraniliprole', 'Flutolanil', 'Indoxacarb',
    'Abamectin', 'Propamocarb.HCL', 'Imidaclopird', 'Chlopyrifos Methyl', 'Chlothianidin',
]
SUBSTANCE_NAME_TO_ID = {name: f"P{i + 1:02d}" for i, name in enumerate(SUBSTANCES)}


def _bin_spectra(X: np.ndarray, machine: str) -> np.ndarray:
    factor = BIN_FACTOR[machine]
    if factor <= 1:
        return X
    n_out = (X.shape[1] // factor) // 8 * 8
    return X[:, :n_out * factor].reshape(len(X), n_out, factor).mean(axis=2).astype(np.float32)


def _preprocess(spectra: np.ndarray, machine: str) -> np.ndarray:
    # Same Savitzky-Golay smoothing + SNV scatter correction applied before
    # training normalization stats were computed (dataset/preprocessing.py).
    # The outlier filter in preprocess_spectra() is a dataset-wide statistic
    # (median/MAD across rows) and doesn't apply to a single live request, so
    # only the per-sample steps are replicated here.
    spectra = snv(savgol_smooth(spectra)).astype(np.float32)
    return _bin_spectra(spectra, machine)


def _food_onehot(categories: List[str]) -> np.ndarray:
    F = np.zeros((len(categories), N_FOOD), dtype=np.float32)
    for i, cat in enumerate(categories):
        idx = FOOD_NAME_TO_INDEX.get(cat)
        if idx is not None:
            F[i, idx] = 1.0
    return F


def infer_category_classification(spectra: np.ndarray, machine: str):
    """GuidedDCNet, 5-fold ensemble by majority vote
    (checkpoint/category_classification_guideddcnet)."""
    task = "category_classification_guideddcnet"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stats_path = f"data/{task}/{machine}/fold_1/stats.npz"
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Stats file not found for {machine} fold 1")
    signal_len = np.load(stats_path)['mean'].shape[1]

    spectra = _preprocess(spectra, machine)
    if spectra.shape[1] != signal_len:
        raise ValueError(f"Input spectrum length {spectra.shape[1]} does not match expected {signal_len}")
    spectra_tensor = torch.from_numpy(spectra).float().to(device)

    all_labels = []
    for fold in range(1, K_FOLDS + 1):
        save_fold_dir = f"data/{task}/{machine}/fold_{fold}"
        stats = np.load(f"{save_fold_dir}/stats.npz")
        mean_t = torch.from_numpy(stats['mean']).float().to(device)
        std_t = torch.from_numpy(stats['std']).float().to(device)
        label_encoder = joblib.load(f"{save_fold_dir}/label_encoder.pkl")

        cfg = GuidedDCNetConfig(num_classes=len(label_encoder.classes_))
        model = GuidedDCNet(cfg).to(device)
        model_path = f"checkpoint/{task}/{machine}/checkpoint_fold{fold}.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found for {machine} fold {fold}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        norm_x = (spectra_tensor - mean_t) / std_t
        with torch.no_grad():
            y0_hat = model.reverse_sample(norm_x)
            preds = torch.argmax(y0_hat, dim=1).cpu().numpy()
        all_labels.append(label_encoder.inverse_transform(preds))

    all_labels = np.array(all_labels)  # (k_folds, batch_size)
    voted_preds = []
    confidences = []
    for i in range(spectra.shape[0]):
        label, votes = Counter(all_labels[:, i]).most_common(1)[0]
        voted_preds.append(label)
        confidences.append(votes / K_FOLDS)  # fraction of the 5 folds agreeing
    return voted_preds, confidences


def _smartnir_ensemble(spectra: np.ndarray, Ft: "torch.Tensor", data_dir: str, ckpt_dir: str,
                        substance: str, device, food_idx: Optional[np.ndarray] = None):
    """Averages calibrated (optionally food-prior-shrunk, for Buoc 2)
    probabilities and decision thresholds over the 5 fold models -- same
    ensembling as evaluate_holdout_stage1.py (validated against K-Fold
    cross-validation numbers on a held-out sample test). Returns
    (avg_prob, avg_threshold), or (None, None) if no fold's artefacts exist.
    """
    probs, thrs = [], []
    idx = torch.arange(len(spectra), device=device)
    for fold in range(1, K_FOLDS + 1):
        norm_path = f"{data_dir}/fold_{fold}_norm.npz"
        model_path = f"{ckpt_dir}/{substance}_fold_{fold}.pth"
        calib_path = f"{data_dir}/{substance}_fold_{fold}_calibrator.pkl"
        thr_path = f"{data_dir}/{substance}_fold_{fold}_threshold.json"
        prior_path = f"{data_dir}/{substance}_fold_{fold}_food_prior.json"
        needed = (norm_path, model_path, calib_path, thr_path) + ((prior_path,) if food_idx is not None else ())
        if not all(os.path.exists(p) for p in needed):
            continue

        nz = np.load(norm_path)
        Xn = torch.tensor((spectra - nz["mean"]) / nz["std"], dtype=torch.float32, device=device)
        model = SmartNIRWithFood(Xn.shape[1], Ft.shape[1]).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        score = predict_logit_diff(model, Xn, Ft, idx, 256)

        calib = joblib.load(calib_path)
        p = calib.predict_proba(score.reshape(-1, 1))[:, 1]
        if food_idx is not None:
            prior_by_food = np.array(json.load(open(prior_path))["prior_by_food"])
            p = food_prior_shrinkage(p, food_idx, prior_by_food)

        probs.append(p)
        thrs.append(json.load(open(thr_path))["threshold"])

    if not probs:
        return None, None
    return np.mean(probs, axis=0), float(np.mean(thrs))


def _threshold_confidence(prob: np.ndarray, thr: float, temperature: float = 1.0) -> np.ndarray:
    """Recenters a calibrated probability around its own decision threshold
    via a logit shift, so the score is exactly 0.5 right at the threshold
    and saturates towards 0/1 as the probability moves away from it in
    either direction. Unlike the raw probability, this doesn't look
    misleadingly low for a positive call when the threshold itself is low
    (Bước 1/2 thresholds are deliberately tuned for Recall>=0.9, so e.g.
    prob=0.16 against a threshold of 0.05 is actually a confident positive,
    not a weak one).
    """
    eps = 1e-6
    p = np.clip(prob, eps, 1 - eps)
    t = np.clip(thr, eps, 1 - eps)
    logit_p = np.log(p / (1 - p))
    logit_t = np.log(t / (1 - t))
    z = (logit_p - logit_t) / temperature
    return 1.0 / (1.0 + np.exp(-z))


def infer_substances_detection(spectra: np.ndarray, machine: str, categories: List[str]):
    """Buoc 1 -- presence/absence, SMART-NIR + food one-hot
    (checkpoint/substance_regression/stage1_smartnir_os2), 5-fold ensemble."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spectra = _preprocess(spectra, machine)
    Ft = torch.tensor(_food_onehot(categories), device=device)

    results = [[] for _ in range(len(spectra))]
    confidences = [{} for _ in range(len(spectra))]
    for substance in SUBSTANCES:
        data_dir = f"data/substance_regression/stage1_smartnir_os2/{machine}/{substance}"
        ckpt_dir = f"checkpoint/substance_regression/stage1_smartnir_os2/{machine}/{substance}"
        avg_prob, avg_thr = _smartnir_ensemble(spectra, Ft, data_dir, ckpt_dir, substance, device)
        if avg_prob is None:
            continue
        conf = _threshold_confidence(avg_prob, avg_thr)
        for i, detected in enumerate(avg_prob > avg_thr):
            if detected:
                results[i].append(substance)
                confidences[i][substance] = float(conf[i])
    return results, confidences


def infer_substances_severity(spectra: np.ndarray, machine: str, categories: List[str],
                               detected_list: List[List[str]]):
    """Buoc 2 -- An toan / Vuot nguong, SMART-NIR + food one-hot + per-food
    prior shrinkage (checkpoint/substance_severity_smartnir), 5-fold ensemble."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spectra = _preprocess(spectra, machine)
    Ft = torch.tensor(_food_onehot(categories), device=device)
    food_idx = np.array([FOOD_NAME_TO_INDEX.get(c, 0) for c in categories])

    results = [{s: None for s in detected} for detected in detected_list]
    confidences = [{} for _ in detected_list]
    substances_needed = sorted({s for detected in detected_list for s in detected})
    for substance in substances_needed:
        data_dir = f"data/substance_severity_smartnir/{machine}/{substance}"
        ckpt_dir = f"checkpoint/substance_severity_smartnir/{machine}/{substance}"
        avg_prob, avg_thr = _smartnir_ensemble(spectra, Ft, data_dir, ckpt_dir, substance, device, food_idx=food_idx)
        if avg_prob is None:
            continue
        verdict = np.where(avg_prob > avg_thr, "Vượt ngưỡng", "An toàn")
        conf = _threshold_confidence(avg_prob, avg_thr)
        for i, detected in enumerate(detected_list):
            if substance in detected:
                results[i][substance] = verdict[i]
                confidences[i][substance] = float(conf[i])
    return results, confidences


def _coded(code: str, conf: float) -> dict:
    return {"code": code, "conf-score": round(conf, 4)}


def analyze_spectrum(spectra: np.ndarray, machine: str):
    """Full pipeline for one request: vegetable category (GuidedDCNet) ->
    detected pesticides (SMART-NIR Bước 1) -> safety verdict per detected
    substance (SMART-NIR Bước 2). Overall `safe` is True only if every
    detected substance is "An toàn"; `substances_over_threshold` is always
    derived from that same per-substance verdict, so it can never name a
    substance outside `substances_detected`.

    Every classification result is a `{"code": ..., "conf-score": ...}`
    object: `category` reports F01..F09 (FOOD_NAME_TO_ID), confidence = the
    fraction of the 5 GuidedDCNet folds that agreed on it. Substances in
    `substances_detected`/`substances_over_threshold` report P01..P19
    (SUBSTANCE_NAME_TO_ID); confidence there is `_threshold_confidence`'s
    logit-shifted score (0.5 at that substance's own decision threshold,
    saturating towards 1 the further past it), not the raw calibrated
    probability -- Bước 1/2 thresholds are deliberately low (Recall>=0.9),
    so the raw probability alone can look unconvincingly low for a clear-cut
    positive call. See FOOD_ID_LEGEND / SUBSTANCE_ID_LEGEND in app.py for
    the id -> name table shown in Swagger. `safe` stays a plain boolean:
    it's an aggregate over all detected substances, not itself a single
    model prediction.
    """
    categories, cat_conf = infer_category_classification(spectra, machine)
    detected_list, detect_conf = infer_substances_detection(spectra, machine, categories)
    severity_list, severity_conf = infer_substances_severity(spectra, machine, categories, detected_list)

    results = []
    for i, (category, detected, severity) in enumerate(zip(categories, detected_list, severity_list)):
        over_threshold = [s for s, verdict in severity.items() if verdict == "Vượt ngưỡng"]
        results.append({
            "category": _coded(FOOD_NAME_TO_ID[category], cat_conf[i]),
            "substances_detected": [_coded(SUBSTANCE_NAME_TO_ID[s], detect_conf[i][s]) for s in detected],
            "safe": len(over_threshold) == 0,
            "substances_over_threshold": [_coded(SUBSTANCE_NAME_TO_ID[s], severity_conf[i][s]) for s in over_threshold],
        })
    return results
