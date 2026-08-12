import numpy as np
from scipy.signal import savgol_filter


def _auto_gap_threshold(row_max_z: np.ndarray, min_gap_ratio: float = 10.0,
                         max_outlier_frac: float = 0.05) -> float:
    """Find the largest multiplicative jump in the sorted per-row robust
    z-scores, restricted to its upper tail. A real sensor glitch sits many
    orders of magnitude above the bulk of natural spectral variation, so it
    shows up as one huge jump; ordinary variation only changes gradually.
    Returns inf (keep everything) if no jump of at least `min_gap_ratio` is
    found, so datasets with no real glitches are left untouched.
    """
    n = len(row_max_z)
    s = np.sort(row_max_z)
    tail_start = max(1, int(n * (1 - max_outlier_frac)))
    best_ratio, best_i = 1.0, None
    for i in range(tail_start, n):
        if s[i - 1] <= 0:
            continue
        ratio = s[i] / s[i - 1]
        if ratio > best_ratio:
            best_ratio, best_i = ratio, i
    if best_i is not None and best_ratio >= min_gap_ratio:
        return float(np.sqrt(s[best_i - 1] * s[best_i]))  # midpoint of the gap, log-scale
    return np.inf


def remove_spectral_outliers(X: np.ndarray) -> np.ndarray:
    """Boolean mask (True = keep) flagging spectra whose wavelength-wise
    robust z-score (median/MAD, not mean/std, so a glitch can't inflate its
    own detection threshold) sits in the isolated tail found by
    `_auto_gap_threshold`. Catches sensor-glitch rows (e.g. a single
    wavelength reading many orders of magnitude off) without a hardcoded
    cutoff that would need re-tuning per dataset.
    """
    med = np.median(X, axis=0, keepdims=True)
    mad = np.median(np.abs(X - med), axis=0, keepdims=True) + 1e-8
    robust_z = np.abs(X - med) / (1.4826 * mad)
    row_max_z = robust_z.max(axis=1)
    threshold = _auto_gap_threshold(row_max_z)
    return row_max_z <= threshold


def savgol_smooth(X: np.ndarray, window_length: int = 11, polyorder: int = 2) -> np.ndarray:
    """Savitzky-Golay smoothing along the wavelength axis, per spectrum."""
    return savgol_filter(X, window_length=window_length, polyorder=polyorder, axis=1)


def snv(X: np.ndarray) -> np.ndarray:
    """Standard Normal Variate: per-spectrum scatter correction."""
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True) + 1e-8
    return (X - mean) / std


def preprocess_spectra(X: np.ndarray):
    """Full per-sample pipeline: outlier filter -> SG smoothing -> SNV.
    Returns (X_processed, keep_mask) where keep_mask marks which input rows
    survived (caller must apply it to any parallel arrays, e.g. targets).
    """
    keep_mask = remove_spectral_outliers(X)
    X = X[keep_mask]
    X = savgol_smooth(X)
    X = snv(X)
    return X.astype(np.float32), keep_mask
