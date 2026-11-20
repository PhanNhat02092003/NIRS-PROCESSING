"""EDA for additional/DanangLeafy6/data/{machine}/ALL.csv (built by
prepare_danangleafy6_dataset.py): per machine, produces the figures below
into additional/DanangLeafy6/figs/{machine}/ and a text summary into
additional/DanangLeafy6/figs/{machine}/SUMMARY.md.

  class_counts.png          sample count per class (bar chart)
  mean_std_spectra.png      mean +/- std spectrum per class, raw and after
                             SG+SNV (same 2 of the 3 stages used by
                             main/analyze_leafy_positions.py)
  target_distributions.png  one subplot per substance: log-scale strip of
                             PRESENT (!= -1) concentrations by class, with
                             a red tick at each class's own MRL (MRL is
                             (class, substance)-specific, looked up via
                             the *original* food name for "Rau La" --
                             exactly like prepare_danangleafy6_dataset.py's
                             safe_label computation)
  safety_breakdown.png      per substance x class: stacked count of
                             absent / present-safe / present-unsafe

Usage: python3 analyze_dataset.py [FLAMENIR OCEANFX]
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))  # repo root: model/, dataset/, shared engines
sys.path.insert(0, os.path.join(_ROOT, "main"))

from dataset.preprocessing import preprocess_spectra

DATASET_ROOT = os.environ.get("DATASET_ROOT", "/home/phannhat/all-dataset/Danang-NIR")
FIG_DIR = f"{_HERE}/figs"

SUBSTANCES = ['Thiamethoxam', 'Permethrin', 'Azoxystrobin',
              'Difenoconazole', 'Cypermethrin', 'Chlothianidin']
CLASS_ORDER = ["Rau Lá", "Cà Chua", "Cà Rốt", "Dưa Leo", "Khổ Qua", "Đậu Cove"]
# MRL lookup needs the *original* food name ("Rau La" is Cai Thia's rows,
# relabeled -- see prepare_danangleafy6_dataset.py).
CLASS_TO_THRESHOLD_NAME = {"Rau Lá": "Cải Thìa"}
COLORS = dict(zip(CLASS_ORDER, ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]))


def load_maps():
    food_ids = json.load(open(f"{DATASET_ROOT}/food_ids.json"))
    pest_ids = json.load(open(f"{DATASET_ROOT}/pesticide_ids.json"))
    thresholds = json.load(open(f"{DATASET_ROOT}/thresholds.json"))
    name_to_fid = {v["name"]: k for k, v in food_ids.items()}
    name_to_pid = {v["name"]: k for k, v in pest_ids.items()}
    return name_to_fid, name_to_pid, thresholds


def mrl(cls: str, substance: str, name_to_fid, name_to_pid, thresholds) -> float:
    fid = name_to_fid[CLASS_TO_THRESHOLD_NAME.get(cls, cls)]
    pid = name_to_pid[substance]
    return thresholds[fid][pid]["mrl"]


def plot_class_counts(df: pd.DataFrame, out_path: str, machine: str):
    counts = df["category"].value_counts().reindex(CLASS_ORDER)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(counts.index, counts.values, color=[COLORS[c] for c in counts.index])
    for i, v in enumerate(counts.values):
        ax.text(i, v + max(counts.values) * 0.01, str(int(v)), ha="center", fontsize=9)
    ax.set_ylabel("Số mẫu")
    ax.set_title(f"DanangLeafy6 ({machine}): số mẫu theo lớp")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_mean_std_spectra(df: pd.DataFrame, out_path: str, machine: str):
    w_cols = [c for c in df.columns if c.startswith("w_")]
    X_raw = df[w_cols].values.astype(np.float32)
    y = df["category"].values
    X_sgsnv, keep = preprocess_spectra(X_raw)
    y_sgsnv = y[keep]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (X, yy, title) in zip(axes, [(X_raw, y, "Phổ thô"), (X_sgsnv, y_sgsnv, "Sau SG + SNV")]):
        for cls in CLASS_ORDER:
            mask = yy == cls
            if mask.sum() == 0:
                continue
            mean = X[mask].mean(axis=0)
            std = X[mask].std(axis=0)
            idx = np.arange(len(mean))
            ax.plot(idx, mean, label=f"{cls} (n={mask.sum()})", color=COLORS[cls])
            ax.fill_between(idx, mean - std, mean + std, alpha=0.2, color=COLORS[cls])
        ax.set_title(f"DanangLeafy6 ({machine}): {title}")
        ax.set_xlabel("Chỉ số bước sóng")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_target_distributions(df: pd.DataFrame, out_path: str, machine: str, name_to_fid, name_to_pid, thresholds):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, substance in zip(axes.flat, SUBSTANCES):
        for i, cls in enumerate(CLASS_ORDER):
            vals = df.loc[df["category"] == cls, substance]
            present = vals[vals != -1]
            if len(present):
                jitter = np.random.default_rng(42).normal(0, 0.06, size=len(present))
                ax.scatter(np.full(len(present), i) + jitter, present, s=8, alpha=0.5, color=COLORS[cls])
            m = mrl(cls, substance, name_to_fid, name_to_pid, thresholds)
            ax.hlines(m, i - 0.3, i + 0.3, color="red", linewidth=2, zorder=5)
        ax.set_yscale("symlog", linthresh=0.1)
        ax.set_xticks(range(len(CLASS_ORDER)))
        ax.set_xticklabels([c.replace(" ", "\n") for c in CLASS_ORDER], fontsize=8)
        n_present = int((df[substance] != -1).sum())
        ax.set_title(f"{substance} (n_present={n_present}/{len(df)})", fontsize=10)
        ax.set_ylabel("Nồng độ (thang log)")
    fig.suptitle(f"DanangLeafy6 ({machine}): phân phối nồng độ theo lớp (vạch đỏ = MRL của lớp đó)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_safety_breakdown(df: pd.DataFrame, out_path: str, machine: str):
    name_to_fid, name_to_pid, thresholds = load_maps()
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, substance in zip(axes.flat, SUBSTANCES):
        absent, safe, unsafe = [], [], []
        for cls in CLASS_ORDER:
            sub = df.loc[df["category"] == cls, substance]
            present_mask = sub != -1
            m = mrl(cls, substance, name_to_fid, name_to_pid, thresholds)
            unsafe_mask = present_mask & (sub > m)
            absent.append(int((~present_mask).sum()))
            unsafe.append(int(unsafe_mask.sum()))
            safe.append(int(present_mask.sum() - unsafe_mask.sum()))
        x = np.arange(len(CLASS_ORDER))
        ax.bar(x, absent, label="Không có mặt", color="lightgray")
        ax.bar(x, safe, bottom=absent, label="Có mặt - an toàn", color="tab:green")
        ax.bar(x, unsafe, bottom=np.array(absent) + np.array(safe), label="Có mặt - vượt ngưỡng", color="tab:red")
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace(" ", "\n") for c in CLASS_ORDER], fontsize=8)
        ax.set_title(substance, fontsize=10)
        ax.set_ylabel("Số mẫu")
    axes.flat[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"DanangLeafy6 ({machine}): an toàn / vượt ngưỡng theo từng hoạt chất x lớp")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_summary(df: pd.DataFrame, out_path: str, machine: str, name_to_fid, name_to_pid, thresholds):
    lines = [f"# DanangLeafy6 ({machine}) -- thống kê dataset\n"]
    lines.append(f"Tổng số mẫu: {len(df)}\n")
    lines.append("## Số mẫu theo lớp\n")
    lines.append(df["category"].value_counts().reindex(CLASS_ORDER).to_markdown() + "\n")
    lines.append("\n## Nhãn an toàn tổng hợp (safe_label, gộp cả 6 hoạt chất)\n")
    lines.append(df["safe_label"].value_counts().to_markdown() + "\n")
    lines.append("\n## Theo từng hoạt chất x lớp: số có mặt / an toàn / vượt ngưỡng (MRL)\n")
    rows = []
    for substance in SUBSTANCES:
        for cls in CLASS_ORDER:
            sub = df.loc[df["category"] == cls, substance]
            present_mask = sub != -1
            m = mrl(cls, substance, name_to_fid, name_to_pid, thresholds)
            unsafe_mask = present_mask & (sub > m)
            rows.append({
                "substance": substance, "class": cls, "mrl": m,
                "n_present": int(present_mask.sum()),
                "n_safe": int(present_mask.sum() - unsafe_mask.sum()),
                "n_unsafe": int(unsafe_mask.sum()),
            })
    summary_df = pd.DataFrame(rows)
    lines.append(summary_df.to_markdown(index=False) + "\n")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    summary_df.to_csv(out_path.replace(".md", ".csv"), index=False)


def run(machine: str):
    csv_path = f"{_HERE}/data/{machine}/ALL.csv"
    if not os.path.exists(csv_path):
        print(f"SKIP {machine}: {csv_path} not found -- run prepare_danangleafy6_dataset.py first")
        return
    df = pd.read_csv(csv_path)
    out_dir = f"{FIG_DIR}/{machine}"
    os.makedirs(out_dir, exist_ok=True)

    name_to_fid, name_to_pid, thresholds = load_maps()

    plot_class_counts(df, f"{out_dir}/class_counts.png", machine)
    plot_mean_std_spectra(df, f"{out_dir}/mean_std_spectra.png", machine)
    plot_target_distributions(df, f"{out_dir}/target_distributions.png", machine, name_to_fid, name_to_pid, thresholds)
    plot_safety_breakdown(df, f"{out_dir}/safety_breakdown.png", machine)
    write_summary(df, f"{out_dir}/SUMMARY.md", machine, name_to_fid, name_to_pid, thresholds)
    print(f"{machine}: wrote figures + SUMMARY.md/.csv -> {out_dir}")


if __name__ == "__main__":
    machines = sys.argv[1:] or ["FLAMENIR", "OCEANFX"]
    for m in machines:
        run(m)
