"""Shared helpers for additional_experiments/{Dataset}/benchmark/*.csv --
these compare methods (SMART-NIR, GuidedDCNet, EBAR, NirMACNet, XSpecMamba,
...) side by side and are hand-aligned (fixed-width columns, not a plain
pandas-written CSV), so appending a new method's row needs to recompute
column widths and re-pad every row rather than just writing one line.
"""

import os

import pandas as pd


def _read_rows(csv_path: str):
    if not os.path.exists(csv_path):
        return None, None
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].str.strip()
    return df, list(df.columns)


def _write_aligned(csv_path: str, df: pd.DataFrame, text_cols: list, float_cols: dict):
    """float_cols: {col_name: n_decimals}. text_cols are left-justified,
    numeric columns right-justified, matching the existing hand-aligned style."""
    formatted = pd.DataFrame(index=df.index)
    widths = {}
    for c in df.columns:
        if c in float_cols:
            formatted[c] = df[c].map(lambda v: f"{float(v):.{float_cols[c]}f}")
        else:
            formatted[c] = df[c].astype(str)
        widths[c] = max(len(c), formatted[c].map(len).max())

    lines = []
    header = ", ".join(c.ljust(widths[c]) for c in df.columns)
    lines.append(header)
    for _, row in formatted.iterrows():
        cells = []
        for c in df.columns:
            if c in text_cols:
                cells.append(row[c].ljust(widths[c]))
            else:
                cells.append(row[c].rjust(widths[c]))
        lines.append(", ".join(cells))

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def append_classification_row(csv_path: str, method: str, acc_mean, acc_std,
                                prec_mean, prec_std, rec_mean, rec_std, f1_mean, f1_std):
    df, cols = _read_rows(csv_path)
    new_row = {
        "method": method, "accuracy_mean": acc_mean, "accuracy_std": acc_std,
        "precision_mean": prec_mean, "precision_std": prec_std,
        "recall_mean": rec_mean, "recall_std": rec_std,
        "f1_mean": f1_mean, "f1_std": f1_std,
    }
    if df is None:
        df = pd.DataFrame([new_row])
    else:
        df = df[df["method"] != method]  # replace if re-run
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    float_cols = {c: 2 for c in df.columns if c != "method"}
    _write_aligned(csv_path, df, text_cols=["method"], float_cols=float_cols)


def append_regression_row(csv_path: str, substance: str, method: str,
                           mse_mean, mse_std, mae_mean, mae_std, rmse_mean, rmse_std,
                           r2_mean, r2_std):
    df, cols = _read_rows(csv_path)
    new_row = {
        "substance": substance, "method": method,
        "mse_mean": mse_mean, "mse_std": mse_std, "mae_mean": mae_mean, "mae_std": mae_std,
        "rmse_mean": rmse_mean, "rmse_std": rmse_std, "r2_mean": r2_mean, "r2_std": r2_std,
    }
    if df is None:
        df = pd.DataFrame([new_row])
    else:
        df = df[~((df["substance"] == substance) & (df["method"] == method))]
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    float_cols = {
        "mse_mean": 4, "mse_std": 4, "mae_mean": 4, "mae_std": 4,
        "rmse_mean": 4, "rmse_std": 4, "r2_mean": 2, "r2_std": 2,
    }
    _write_aligned(csv_path, df, text_cols=["substance", "method"], float_cols=float_cols)
