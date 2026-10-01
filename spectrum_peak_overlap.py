"""Compares, per pesticide substance, where the real spectrum actually
changes with concentration (from DATASET_ROOT/spectrum-differencce-clean,
see clean_spectrum_difference.py) against which wavelengths the trained
SMART-NIR Bước 1 detector (checkpoint/substance_regression/stage1_smartnir_os2)
actually relies on, via input-gradient saliency on real positive samples.

For each (machine, substance) with cleaned concentration-series data:
  1. "Thực tế" signal: mean absorbance at the highest concentration minus
     the lowest (sign kept for plotting; magnitude used for the
     correlation below).
  2. "Mô hình" signal: |d(logit_diff)/d(normalized input)|, averaged over
     ~200 real positive samples and over the 5 CV folds, then smoothed
     with a SMOOTH_POINTS-wide moving average (a raw per-point gradient is
     noisy in a way the already-smoothed, SNV + Savitzky-Golay
     real-difference curve isn't).
  3. Overlap: Pearson correlation between |real difference| and the
     (smoothed) model saliency curve across the FULL wavelength grid --
     not just a handful of peak positions. This is a much stronger ask
     than peak-matching (it requires the two curves to actually track each
     other point-for-point, not merely have a few local maxima nearby), so
     expect lower numbers for substances where the model is only sensitive
     to a few of the real bands rather than the whole shape.

OCEANFX's model operates on BIN=8-averaged wavelengths (264 points, not the
raw 2136), so the real-spectrum difference curve is averaged the same way
before correlating, so both curves are compared on the same wavelength
grid the model actually sees.

CNN edge artifact: on FLAMENIR, all 16/16 independently-trained models'
single largest saliency value landed on the exact same grid point (index
121/127, 6 points from the right edge) -- 16 separately-fit models can't
coincidentally learn the same "chemistry" at one shared pixel, so this is a
boundary/zero-padding artifact of SmartNIRWithFood's MultiKernelBlock (conv
kernels up to size 32, stride 4) rather than a real signal. EDGE_EXCLUDE
points (default 16 = half the largest kernel) are trimmed off both ends of
BOTH curves before correlating, so the artifact can't inflate or deflate r.

Peaks are still marked on each substance's chart (top-K by prominence, per
curve independently) purely as a visual reference -- they no longer feed
into the headline overlap number.

Usage: python3 spectrum_peak_overlap.py [TOP_K] [SMOOTH_POINTS] [EDGE_EXCLUDE]
Writes reports/figs/spectrum_overlap/{machine}_{substance}.png and
results/spectrum_peak_overlap.json (+ .csv).
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from scipy.signal import find_peaks
from scipy.stats import pearsonr

from dataset.preprocessing import preprocess_spectra, savgol_smooth, snv
from stage1_detection import SmartNIRWithFood

load_dotenv()

TOP_K = int(sys.argv[1]) if len(sys.argv) > 1 else 10
SMOOTH_POINTS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
EDGE_EXCLUDE = int(sys.argv[3]) if len(sys.argv) > 3 else 16
N_SAMPLES = 200
K_FOLDS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = os.environ["DATASET_ROOT"]
CLEAN_DIR = f"{ROOT}/spectrum-differencce-clean"
FIG_DIR = "reports/figs/spectrum_overlap"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs("results", exist_ok=True)

with open(f"{ROOT}/food_ids.json") as f:
    FOOD_IDS = json.load(f)


def bin_array(values, factor):
    """Groups a 1D array into chunks of `factor`, trimmed to a multiple of
    8 (matches dataset/preprocessing.py's BIN logic), returns the mean per
    group."""
    n_out = (len(values) // factor) // 8 * 8
    return values[:n_out * factor].reshape(n_out, factor).mean(axis=1)


def wavelength_axis(machine: str, bin_factor: int):
    """Canonical wavelength-per-pixel array for `machine`, read from one of
    our own cleaned concentration-series files (same physical instrument
    pixel grid as the w_* columns in ALL.csv), binned to match model space."""
    any_csv = next(f for f in os.listdir(f"{CLEAN_DIR}/{machine}") if f.endswith(".csv"))
    df = pd.read_csv(f"{CLEAN_DIR}/{machine}/{any_csv}")
    wl = np.sort(df["wavelength"].unique())
    return bin_array(wl, bin_factor) if bin_factor > 1 else wl


def real_difference_curve(machine: str, substance: str, bin_factor: int):
    """Mean(rank 1) - mean(rank 5) (highest concentration minus lowest),
    in the SAME preprocessing domain the model actually sees: each raw
    replicate spectrum is Savitzky-Golay smoothed + SNV-corrected (matching
    dataset/preprocessing.py's preprocess_spectra, applied to the raw
    instrument export, not the already-dark/reference-corrected mean) and
    *then* averaged per concentration rank. Comparing against the model's
    saliency without this is comparing two different numerical spaces --
    the model never sees raw absorbance, it sees SNV output.
    """
    df = pd.read_csv(f"{CLEAN_DIR}/{machine}/{substance}.csv")
    wl = np.sort(df["wavelength"].unique())
    mean_by_rank = {}
    for rank, g in df.groupby("rank"):
        # pivot_table (not pivot): a few (replicate, wavelength) pairs collide
        # across sub-batches within the same rank (replicate numbering isn't
        # globally unique), so duplicates are averaged rather than erroring.
        spectra = g.pivot_table(index="replicate", columns="wavelength", values="absorbance", aggfunc="mean")
        spectra = spectra.reindex(columns=wl).dropna(how="any").values.astype(np.float64)
        processed = snv(savgol_smooth(spectra))
        mean_by_rank[rank] = processed.mean(axis=0)
    diff = mean_by_rank[1] - mean_by_rank[5]  # rank 1 = highest conc, 5 = lowest
    return bin_array(diff, bin_factor) if bin_factor > 1 else diff


def load_machine_data(machine: str, bin_factor: int):
    df = pd.read_csv(f"{ROOT}/{machine}/ALL.csv")
    w_cols = [c for c in df.columns if c.startswith("w_")]
    X, keep = preprocess_spectra(df[w_cols].values.astype(np.float32))
    df = df[keep].reset_index(drop=True)
    if bin_factor > 1:
        n_out = (X.shape[1] // bin_factor) // 8 * 8
        X = X[:, :n_out * bin_factor].reshape(len(X), n_out, bin_factor).mean(axis=2).astype(np.float32)
    name_to_id = {v["name"]: k for k, v in FOOD_IDS.items()}
    food_idx = df["category"].map(name_to_id)
    F = pd.get_dummies(food_idx).reindex(columns=list(FOOD_IDS), fill_value=0).astype(np.float32).values
    return df, X, F


def model_saliency_curve(machine: str, substance: str, df, X, F, rng):
    tag_dir = f"data/substance_regression/stage1_smartnir_os2/{machine}/{substance}"
    ckpt_dir = f"checkpoint/substance_regression/stage1_smartnir_os2/{machine}/{substance}"
    pos_idx = np.where(df[substance].values != -1)[0]
    if len(pos_idx) == 0:
        return None
    pick = rng.choice(pos_idx, size=min(N_SAMPLES, len(pos_idx)), replace=False)
    Xb, Fb = X[pick], F[pick]

    saliencies = []
    for fold in range(1, K_FOLDS + 1):
        norm_path = f"{tag_dir}/fold_{fold}_norm.npz"
        model_path = f"{ckpt_dir}/{substance}_fold_{fold}.pth"
        if not (os.path.exists(norm_path) and os.path.exists(model_path)):
            continue
        nz = np.load(norm_path)
        Xn = torch.tensor((Xb - nz["mean"]) / nz["std"], dtype=torch.float32, device=DEVICE, requires_grad=True)
        Fn = torch.tensor(Fb, device=DEVICE)

        model = SmartNIRWithFood(Xn.shape[1], Fn.shape[1]).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()

        logits = model(Xn, Fn)
        score = (logits[:, 1] - logits[:, 0]).sum()
        model.zero_grad(set_to_none=True)
        if Xn.grad is not None:
            Xn.grad.zero_()
        score.backward()
        saliencies.append(Xn.grad.detach().abs().mean(dim=0).cpu().numpy())

    if not saliencies:
        return None
    return np.mean(saliencies, axis=0)


def smooth_curve(curve: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return curve
    kernel = np.ones(window) / window
    return np.convolve(curve, kernel, mode="same")


def pick_peaks(curve: np.ndarray, k: int, edge_exclude: int = 0):
    mag = np.abs(curve)
    if edge_exclude > 0:
        mag = mag.copy()
        mag[:edge_exclude] = 0
        mag[len(mag) - edge_exclude:] = 0
    idx, props = find_peaks(mag, prominence=mag.std() * 0.1)
    if len(idx) == 0:
        return np.array([], dtype=int)
    order = np.argsort(-props["prominences"])
    return idx[order][:k]


def full_spectrum_correlation(real_diff: np.ndarray, model_sal: np.ndarray, edge_exclude: int):
    """Pearson r between |real difference| and model saliency, both curves
    trimmed by `edge_exclude` points on each end first so the CNN boundary
    artifact (see module docstring) can't influence the result."""
    lo, hi = edge_exclude, len(real_diff) - edge_exclude
    r, p = pearsonr(np.abs(real_diff[lo:hi]), model_sal[lo:hi])
    return float(r), float(p)


def plot_substance(machine, substance, wl, real_diff, model_sal, real_peaks, model_peaks, r, p):
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(wl, real_diff, color="#1b6ca8", lw=1)
    axes[0].plot(wl[real_peaks], real_diff[real_peaks], "o", color="#d1495b", ms=5)
    axes[0].set_ylabel("Chênh lệch hấp thụ\n(nồng độ cao - thấp)")
    axes[0].set_title(f"{substance} -- {machine} (r={r:.2f}, p={p:.3f})")

    axes[1].plot(wl, model_sal, color="#2a9d8f", lw=1)
    axes[1].plot(wl[model_peaks], model_sal[model_peaks], "o", color="#d1495b", ms=5)
    axes[1].set_ylabel("Độ nhạy mô hình\n|d(logit)/d(input)|")
    axes[1].set_xlabel("Bước sóng (nm)")

    fig.tight_layout()
    safe_name = substance.replace("/", "-")
    fig.savefig(f"{FIG_DIR}/{machine}_{safe_name}.png", dpi=130)
    plt.close(fig)


def main():
    rng = np.random.default_rng(42)
    results = []

    for machine in ("FLAMENIR", "OCEANFX"):
        bin_factor = 8 if machine == "OCEANFX" else 1
        machine_dir = f"{CLEAN_DIR}/{machine}"
        if not os.path.isdir(machine_dir):
            continue
        substances = sorted(f[:-4] for f in os.listdir(machine_dir) if f.endswith(".csv"))
        print(f"=== {machine}: {len(substances)} substances with concentration data ===")

        wl = wavelength_axis(machine, bin_factor)
        df, X, F = load_machine_data(machine, bin_factor)

        for substance in substances:
            real_diff = real_difference_curve(machine, substance, bin_factor)
            if len(real_diff) != len(wl):
                print(f"  {substance}: SKIP (length mismatch real={len(real_diff)} wl={len(wl)})")
                continue
            model_sal = model_saliency_curve(machine, substance, df, X, F, rng)
            if model_sal is None:
                print(f"  {substance}: SKIP (no trained fold checkpoints)")
                continue
            model_sal = smooth_curve(model_sal, SMOOTH_POINTS)

            real_peaks = pick_peaks(real_diff, TOP_K)
            model_peaks = pick_peaks(model_sal, TOP_K, edge_exclude=EDGE_EXCLUDE)
            r, p = full_spectrum_correlation(real_diff, model_sal, EDGE_EXCLUDE)

            plot_substance(machine, substance, wl, real_diff, model_sal, real_peaks, model_peaks, r, p)
            results.append({
                "machine": machine, "substance": substance,
                "pearson_r": round(r, 4), "pearson_p": round(p, 4),
                "real_peak_wavelengths": [round(float(x), 1) for x in wl[real_peaks]],
                "model_peak_wavelengths": [round(float(x), 1) for x in wl[model_peaks]],
            })
            print(f"  {substance:20s} r={r:+.3f}  p={p:.4f}")

    with open("results/spectrum_peak_overlap.json", "w") as f:
        json.dump({
            "top_k": TOP_K, "smooth_points": SMOOTH_POINTS, "edge_exclude_points": EDGE_EXCLUDE,
            "metric": "pearson_r between |real concentration-difference| and smoothed model saliency, full spectrum",
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    res_df = pd.DataFrame(results).sort_values("pearson_r")
    res_df.to_csv("results/spectrum_peak_overlap.csv", index=False)
    print(f"\nWrote results/spectrum_peak_overlap.{{json,csv}} ({len(results)} rows), figures under {FIG_DIR}/")

    labels = res_df["machine"] + " / " + res_df["substance"]
    colors = ["#d1495b" if r < 0 else "#2a9d8f" for r in res_df["pearson_r"]]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(res_df))))
    ax.barh(labels, res_df["pearson_r"], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Tương quan Pearson (|chênh lệch thực tế| vs. độ nhạy mô hình, toàn phổ)")
    ax.set_title("Tương quan toàn phổ: thực tế vs. độ nhạy mô hình, theo thuốc", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/_summary.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
