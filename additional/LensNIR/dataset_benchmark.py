"""
Public NIR benchmark loaders: Rapeseed, Grainit, Mango.

Each dataset ships a pre-built ALL.csv with the same layout:
  w_0 … w_{L-1}  : float  spectral features
  category        : str    classification label
  <targets>       : float  regression targets (-1 = not measured)
  <metadata>      : ignored (except the optional group column)

| Dataset  | Rows   | Bands | Classes                         | Regression targets      |
|----------|--------|-------|---------------------------------|-------------------------|
| Rapeseed |  2 427 | 1 153 | 5 organs (Leaves, Roots, …)     | N_content, C_content (%)|
| Grainit  |    683 |   349 | 3 cereals (wheat, corn, barley) | Moisture, Protein (%)   |
| Mango    | 78 159 |   173 | 10 cultivars                    | dry_matter (%)          |

Missing targets (-1) are returned as NaN and masked out of the loss / metrics.

Mango has ~8 spectra per fruit (same `reference_no`, identical dry_matter),
so its CV folds must be grouped by fruit — otherwise repeated scans of the
same fruit land in both train and test. `groups` is returned for that.

Spectra are linearly resampled to the largest length ≤ n_bands that is
divisible by 4 (Conv2d stride-4 requirement), then per-sample min-max
normalised to [0, 1].
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.preprocessing import LabelEncoder

FOOD_COL = "category"
MISSING_VALUE = -1.0

DEFAULT_DATA_ROOT = Path(
    os.getenv("DATA_ROOT", Path(__file__).resolve().parent.parent / "all-dataset")
)

# Quality task (PASS/FAIL or label), one per dataset:
#   "quality_col"   — a real label column, learned by a second classification
#                     head (Rapeseed: N fertilisation regime N-/N+). No numeric
#                     N_content threshold holds across organs / growth stages.
#   "quality_rules" — derived post-hoc (no extra head): PASS iff every listed
#                     target meets its bound for the sample's class, using the
#                     model's predicted class and regression outputs.
#                     {class: {target: ("min"|"max", value)}}, "_default" = fallback.
DATASETS: dict = {
    "Rapeseed": {
        "path":        "RechercheDataGouv-Rapeseed/Rapeseed/ALL.csv",
        "targets":     ["N_content", "C_content"],
        "group_col":   None,
        "quality_col": "N regime",
    },
    "Grainit": {
        "path":      "Zenodo-Grainit/Grainit/ALL.csv",
        "targets":   ["Moisture", "Protein"],
        "group_col": None,
        "quality_rules": {
            "wheat":  {"Moisture": ("max", 14.5), "Protein": ("min", 11.5)},
            "barley": {"Moisture": ("max", 14.5)},
            "corn":   {"Moisture": ("max", 13.5)},
        },
    },
    "Mango": {
        "path":      "Mendeley-Mango/Mango/ALL.csv",
        "targets":   ["dry_matter"],
        "group_col": "reference_no",
        "quality_rules": {
            "r2e2":     {"dry_matter": ("min", 13.0)},
            "_default": {"dry_matter": ("min", 15.0)},
        },
    },
    # The Danang pesticide dataset's CAP=25 downsampled data, 4 leafy greens
    # collapsed into one Cai Thia-only "Rau La" class -- see
    # ../DanangLeafy6/prepare_danangleafy6_dataset.py. `path` is an absolute
    # path (outside DATA_ROOT, which only holds the 3 datasets above), so
    # these 2 entries work regardless of DATA_ROOT's value. All 3 tasks:
    # classification (6-way), regression (6 of the 19 pesticide
    # concentrations, picked for this benchmark; -1 = absent = MISSING_VALUE),
    # quality (learned head, "safe_label" = An toan/Vuot nguong precomputed
    # in prepare_danangleafy6_dataset.py from the *original* category + MRLs
    # over just these 6 substances -- a quality_rules dict doesn't fit here,
    # since it treats a missing/NaN target as undefined rather than
    # trivially-safe-for-that-substance).
    # "log_targets": these concentrations are heavy-tailed (CV~3-4, max
    # 100-900x the median -- additional/DanangLeafy6/figs/*/SUMMARY.csv),
    # same reason additional/dataset/regression_dataset.py and every other
    # DanangLeafy6 regression engine log1p's the target before z-scoring;
    # train_fold() does the same here when this flag is set.
    "DanangLeafy6_FLAMENIR": {
        "path":        "/home/phannhat/NIRS-PROCESSING/additional/DanangLeafy6/data/FLAMENIR/ALL_synthetic.csv",
        "targets":     ['Thiamethoxam', 'Permethrin', 'Azoxystrobin',
                        'Difenoconazole', 'Cypermethrin', 'Chlothianidin'],
        "group_col":   "_group_id",
        "quality_col": "safe_label",
        "log_targets": True,
    },
    "DanangLeafy6_OCEANFX": {
        "path":        "/home/phannhat/NIRS-PROCESSING/additional/DanangLeafy6/data/OCEANFX/ALL_synthetic.csv",
        "targets":     ['Thiamethoxam', 'Permethrin', 'Azoxystrobin',
                        'Difenoconazole', 'Cypermethrin', 'Chlothianidin'],
        "group_col":   "_group_id",
        "log_targets": True,
        "quality_col": "safe_label",
    },
}
DATASET_NAMES = list(DATASETS)
QUALITY_CLASSES = ["FAIL", "PASS"]   # index 0 / 1 for rule-derived quality


def derive_quality(
    y_reg: np.ndarray,          # (N, T) targets in original units, NaN = missing
    class_idx: np.ndarray,      # (N,)   class index per sample (true or predicted)
    food_classes: list,
    target_names: list,
    rules: dict,
) -> np.ndarray:
    """PASS (1.0) / FAIL (0.0) per sample; NaN when a required target is missing."""
    out = np.full(len(y_reg), np.nan, dtype=np.float32)
    for c, cname in enumerate(food_classes):
        rule = rules.get(cname, rules.get("_default"))
        sel = class_idx == c
        if rule is None or not sel.any():
            continue
        ok = np.ones(sel.sum(), dtype=bool)
        known = np.ones(sel.sum(), dtype=bool)
        for t, (kind, thr) in rule.items():
            v = y_reg[sel, target_names.index(t)]
            known &= ~np.isnan(v)
            ok &= (v >= thr) if kind == "min" else (v <= thr)
        out[sel] = np.where(known, ok.astype(np.float32), np.nan)
    return out


# ─── Spectral utilities ───────────────────────────────────────────────────────

def _spectral_cols(df: pd.DataFrame) -> list:
    return sorted(
        (c for c in df.columns if c.startswith("w_")),
        key=lambda c: int(c.split("_")[1]),
    )


def resample_spectra(X: np.ndarray, target_len: int) -> np.ndarray:
    """Linear interpolation: (N, src_len) → (N, target_len)."""
    src_len = X.shape[1]
    if src_len == target_len:
        return X.astype(np.float32)
    src_grid = np.linspace(0.0, 1.0, src_len)
    dst_grid = np.linspace(0.0, 1.0, target_len)
    return interp1d(src_grid, X, kind="linear", axis=1)(dst_grid).astype(np.float32)


def normalize_spectra(X: np.ndarray) -> np.ndarray:
    """Per-sample min-max normalisation → [0, 1]."""
    xmin = X.min(axis=1, keepdims=True)
    xmax = X.max(axis=1, keepdims=True)
    return ((X - xmin) / (xmax - xmin + 1e-8)).astype(np.float32)


def load_benchmark(
    name: str,
    data_root: Optional[str | Path] = None,
    verbose: bool = True,
) -> dict:
    """
    Load one benchmark dataset.

    Returns a dict:
        X            : (N, seq_len) float32 — resampled, min-max normalised spectra
        y_food       : (N,)         int64   — encoded class labels
        y_reg        : (N, T)       float32 — regression targets, NaN where missing
        y_aux        : (N,) or None int64   — quality-head labels (Rapeseed N-/N+)
        aux_classes  : list[str]            — quality-head class names ([] if none)
        quality_rules: dict or None         — post-hoc PASS/FAIL thresholds
        groups       : (N,) or None         — CV group ids (Mango: fruit id)
        food_classes : list[str]
        target_names : list[str]
        seq_len      : int
        n_bands      : int — original band count
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {DATASET_NAMES}")
    spec = DATASETS[name]
    path = Path(data_root or DEFAULT_DATA_ROOT) / spec["path"]
    if not path.exists():
        raise FileNotFoundError(f"[{name}] CSV not found: {path}")

    df = pd.read_csv(path)
    missing = [c for c in [FOOD_COL, *spec["targets"]] if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing columns: {missing}")

    spec_cols = _spectral_cols(df)
    n_bands = len(spec_cols)
    seq_len = n_bands - n_bands % 4
    X = resample_spectra(df[spec_cols].values.astype(np.float32), seq_len)
    X = normalize_spectra(X)

    y_reg = df[spec["targets"]].values.astype(np.float32)
    y_reg[y_reg == MISSING_VALUE] = np.nan

    le = LabelEncoder().fit(df[FOOD_COL].astype(str).values)
    y_food = le.transform(df[FOOD_COL].astype(str).values).astype(np.int64)

    groups = df[spec["group_col"]].values if spec["group_col"] else None

    y_aux, aux_classes = None, []
    if spec.get("quality_col"):
        le_aux = LabelEncoder().fit(df[spec["quality_col"]].astype(str).values)
        y_aux = le_aux.transform(df[spec["quality_col"]].astype(str).values).astype(np.int64)
        aux_classes = list(le_aux.classes_)
    quality_rules = spec.get("quality_rules")

    if verbose:
        n_missing = np.isnan(y_reg).sum(axis=0)
        counts = np.bincount(y_food)
        print(f"[{name}] {len(df):,} samples | {n_bands} bands → seq_len={seq_len} "
              f"(H_out={seq_len // 4}) | {len(le.classes_)} classes "
              f"(min/max per class: {counts.min()}/{counts.max()})")
        for t, nm in zip(spec["targets"], n_missing):
            col = y_reg[:, spec["targets"].index(t)]
            print(f"  target {t:<12} mean={np.nanmean(col):.3f} std={np.nanstd(col):.3f} "
                  f"missing={nm}")
        if groups is not None:
            print(f"  CV groups: '{spec['group_col']}' ({len(np.unique(groups)):,} unique)")
        if y_aux is not None:
            print(f"  quality head: '{spec['quality_col']}' "
                  f"{dict(zip(aux_classes, np.bincount(y_aux).tolist()))}")
        if quality_rules:
            q = derive_quality(y_reg, y_food, list(le.classes_), spec["targets"], quality_rules)
            print(f"  quality (derived): PASS={int(np.nansum(q))} "
                  f"FAIL={int((q == 0).sum())} undefined={int(np.isnan(q).sum())}")

    return {
        "X":            X,
        "y_food":       y_food,
        "y_reg":        y_reg,
        "y_aux":        y_aux,
        "aux_classes":  aux_classes,
        "quality_rules": quality_rules,
        "groups":       groups,
        "food_classes": list(le.classes_),
        "target_names": list(spec["targets"]),
        "seq_len":      seq_len,
        "n_bands":      n_bands,
    }
