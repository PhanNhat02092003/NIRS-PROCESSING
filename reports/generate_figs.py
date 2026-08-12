"""
Regenerates the EDA figures used in reports/main.tex (Hinh 4/5/6: spectra,
crosstab heatmap, concentration boxplot) for both machines.

Run from anywhere, e.g.:
    python3 reports/generate_figs.py

Reads from ../all-dataset/Danang-NIR/{FLAMENIR,OCEANFX}/ALL.csv and writes
PNGs into reports/figs/. Adjust SUBSTANCES/CAT_ORDER or the plotting blocks
below and re-run to refresh the figures after any data or styling change.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataset.preprocessing import remove_spectral_outliers

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(REPO_ROOT, "reports", "figs")
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 17,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
})

SUBSTANCES = [
    'Thiamethoxam', 'Permethrin', 'Metalaxyl', 'Azoxystrobin',
    'Imidaclopird', 'Difenoconazole', 'Cypermethrin', 'Cyhalothrin',
    'Chlorantraniliprol', 'Chlopyrifos Methyl', 'Emamectin benzoate',
    'Chlorothalonil', 'Triadimefon', 'Cyantraniliprole', 'Flutolanil',
    'Indoxacarb', 'Abamectin', 'Propamocarb.HCL', 'Chlothianidin'
]

CAT_ORDER = ['Khổ Qua', 'Mồng Tơi', 'Cải Thìa', 'Cà Chua', 'Cải Bẹ Xanh',
             'Dưa Leo', 'Xà Lách', 'Đậu Cove', 'Cà Rốt']

data = {}
crosstabs = {}

for machine in ['FLAMENIR', 'OCEANFX']:
    print(f"=== {machine} ===")
    df = pd.read_csv(os.path.join(REPO_ROOT, "..", "all-dataset", "Danang-NIR", machine, "ALL.csv"))
    w_cols = [c for c in df.columns if c.startswith('w_')]

    keep_mask = remove_spectral_outliers(df[w_cols].values.astype(np.float32))
    n_dropped = int((~keep_mask).sum())
    if n_dropped:
        print(f"[{machine}] dropped {n_dropped} outlier spectra (sensor glitch)")
    df = df[keep_mask].reset_index(drop=True)
    data[machine] = df

    crosstab_pct = pd.DataFrame(index=SUBSTANCES, columns=CAT_ORDER, dtype=float)
    for cat in CAT_ORDER:
        sub_df = df[df['category'] == cat]
        for s in SUBSTANCES:
            crosstab_pct.loc[s, cat] = 100.0 * (sub_df[s] > 0).mean() if len(sub_df) else np.nan
    crosstabs[machine] = crosstab_pct

vmax_shared = max(crosstabs['FLAMENIR'].values.max(), crosstabs['OCEANFX'].values.max())

# ---------------- Figure: NIR spectra mean +/- std by category ----------------
for machine in ['FLAMENIR', 'OCEANFX']:
    df = data[machine]
    w_cols = [c for c in df.columns if c.startswith('w_')]
    n_wave = len(w_cols)
    plt.figure(figsize=(8, 6.5))
    x = np.arange(n_wave)
    for cat in CAT_ORDER:
        sub = df[df['category'] == cat][w_cols].values
        mean = sub.mean(axis=0)
        std = sub.std(axis=0)
        plt.plot(x, mean, label=cat, linewidth=1.6)
        plt.fill_between(x, mean - std, mean + std, alpha=0.12)
    plt.xlabel("Chỉ số bước sóng")
    plt.ylabel("Cường độ phổ")
    plt.legend(fontsize=11, ncol=2, loc='upper right')
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/spectra_{machine}.png", dpi=220)
    plt.close()

# ---------------- Figure: substance (rows) x category (cols) heatmap, annotated ----------------
for machine in ['FLAMENIR', 'OCEANFX']:
    ct = crosstabs[machine]
    fig, ax = plt.subplots(figsize=(7.2, 11))
    im = ax.imshow(ct.values.astype(float), aspect='auto', cmap='YlOrRd', vmin=0, vmax=vmax_shared)
    ax.set_xticks(range(len(CAT_ORDER)))
    ax.set_xticklabels(CAT_ORDER, rotation=45, ha='right', fontsize=13)
    ax.set_yticks(range(len(SUBSTANCES)))
    ax.set_yticklabels(SUBSTANCES, fontsize=13)
    # annotate each cell with 1-decimal percentage, matching table-style precision
    for i in range(len(SUBSTANCES)):
        for j in range(len(CAT_ORDER)):
            val = ct.values[i, j]
            color = 'white' if val > 0.55 * vmax_shared else 'black'
            ax.text(j, i, f"{val:.1f}", ha='center', va='center', fontsize=10.5, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label('% mẫu có chứa', fontsize=13)
    cbar.ax.tick_params(labelsize=11)
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/crosstab_{machine}.png", dpi=220)
    plt.close()

# ---------------- Figure: concentration boxplot (0-1 normalized) per substance ----------------
for machine in ['FLAMENIR', 'OCEANFX']:
    df = data[machine]
    box_data, box_labels = [], []
    for s in SUBSTANCES:
        vals = df.loc[df[s] > 0, s].values
        if len(vals) == 0:
            continue
        vmin, vmax = vals.min(), vals.max()
        norm = (vals - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(vals)
        box_data.append(norm)
        box_labels.append(s)
    plt.figure(figsize=(8, 7.5))
    plt.boxplot(box_data, labels=box_labels, showfliers=True, vert=False)
    plt.yticks(fontsize=13)
    plt.xlabel("Nồng độ (chuẩn hoá 0-1 theo từng chất)")
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/concbox_{machine}.png", dpi=220)
    plt.close()

print("DONE")
