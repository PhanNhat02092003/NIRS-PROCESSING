"""Compares, per pesticide substance, where the real spectrum actually
changes with concentration (from DATASET_ROOT/spectrum-differencce-clean,
see clean_spectrum_difference.py) against which wavelengths the trained
SMART-NIR Bước 1 detector (checkpoint/substance_regression/stage1_smartnir_os2)
actually relies on, via input-gradient saliency on real positive samples.

For each (machine, substance) with cleaned concentration-series data:
  1. "Thực tế" signal: mean absorbance at the highest concentration minus
     the lowest, |difference| peak-picked with scipy.signal.find_peaks.
  2. "Mô hình" signal: |d(logit_diff)/d(normalized input)|, averaged over
     ~200 real positive samples and over the 5 CV folds, peak-picked the
     same way.
  3. Overlap: greedy-matches the top-K peaks of each signal within a
     wavelength tolerance, reported as a Jaccard-style ratio
     matched / (2K - matched).

OCEANFX's model operates on BIN=8-averaged wavelengths (264 points, not the
raw 2136), so the real-spectrum difference curve is averaged the same way
before peak-picking, so both curves are compared on the same wavelength
grid the model actually sees.

Usage: python3 spectrum_peak_overlap.py [TOP_K] [TOLERANCE_NM]
Writes reports/figs/spectrum_overlap/{machine}_{substance}.png and
results/spectrum_peak_overlap.json (+ .csv).
"""
import json
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from scipy.signal import find_peaks

from dataset.preprocessing import preprocess_spectra
from stage1_detection import SmartNIRWithFood

load_dotenv()

TOP_K = int(sys.argv[1]) if len(sys.argv) > 1 else 10
TOLERANCE_NM = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
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
    df = pd.read_csv(f"{CLEAN_DIR}/{machine}/{substance}.csv")
    mean_by_rank = df.groupby(["rank", "wavelength"])["absorbance"].mean().unstack("rank")
    mean_by_rank = mean_by_rank.sort_index()
    diff = (mean_by_rank[1] - mean_by_rank[5]).values  # rank 1 = highest conc, 5 = lowest
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


def pick_peaks(curve: np.ndarray, k: int):
    mag = np.abs(curve)
    idx, props = find_peaks(mag, prominence=mag.std() * 0.1)
    if len(idx) == 0:
        return np.array([], dtype=int)
    order = np.argsort(-props["prominences"])
    return idx[order][:k]


def jaccard_overlap(wl_a, peaks_a, wl_b, peaks_b, tol):
    pos_a = wl_a[peaks_a]
    pos_b = list(wl_b[peaks_b])
    matched = 0
    for pa in pos_a:
        for i, pb in enumerate(pos_b):
            if abs(pa - pb) <= tol:
                matched += 1
                pos_b.pop(i)
                break
    denom = len(peaks_a) + len(peaks_b) - matched
    return matched, (matched / denom if denom > 0 else 0.0)


def plot_substance(machine, substance, wl, real_diff, model_sal, real_peaks, model_peaks, jac, matched):
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(wl, real_diff, color="#1b6ca8", lw=1)
    axes[0].plot(wl[real_peaks], real_diff[real_peaks], "o", color="#d1495b", ms=5)
    axes[0].set_ylabel("Chênh lệch hấp thụ\n(nồng độ cao - thấp)")
    axes[0].set_title(f"{substance} -- {machine} (Jaccard={jac:.2f}, khớp {matched}/{TOP_K})")

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

            real_peaks = pick_peaks(real_diff, TOP_K)
            model_peaks = pick_peaks(model_sal, TOP_K)
            matched, jac = jaccard_overlap(wl, real_peaks, wl, model_peaks, TOLERANCE_NM)

            plot_substance(machine, substance, wl, real_diff, model_sal, real_peaks, model_peaks, jac, matched)
            results.append({
                "machine": machine, "substance": substance,
                "n_real_peaks": int(len(real_peaks)), "n_model_peaks": int(len(model_peaks)),
                "matched": int(matched), "jaccard": round(float(jac), 4),
                "real_peak_wavelengths": [round(float(x), 1) for x in wl[real_peaks]],
                "model_peak_wavelengths": [round(float(x), 1) for x in wl[model_peaks]],
            })
            print(f"  {substance:20s} real_peaks={len(real_peaks):2d} model_peaks={len(model_peaks):2d} "
                  f"matched={matched:2d} jaccard={jac:.3f}")

    with open("results/spectrum_peak_overlap.json", "w") as f:
        json.dump({"top_k": TOP_K, "tolerance_nm": TOLERANCE_NM, "results": results}, f, indent=2, ensure_ascii=False)
    res_df = pd.DataFrame(results).sort_values("jaccard")
    res_df.to_csv("results/spectrum_peak_overlap.csv", index=False)
    print(f"\nWrote results/spectrum_peak_overlap.{{json,csv}} ({len(results)} rows), figures under {FIG_DIR}/")

    labels = res_df["machine"] + " / " + res_df["substance"]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(res_df))))
    ax.barh(labels, res_df["jaccard"], color="#2a9d8f")
    ax.set_xlabel(f"Jaccard (top-{TOP_K} đỉnh, dung sai ±{TOLERANCE_NM:g}nm)")
    ax.set_title("Độ trùng lặp peak thực tế vs. độ nhạy mô hình, theo thuốc", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/_summary.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
