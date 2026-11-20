"""Inference-efficiency benchmark (params, MACs, single-sample latency,
batch throughput) for every method already in additional/DanangLeafy6/
benchmark/{classification,regression,quality}/{FLAMENIR,OCEANFX}.csv --
see additional/efficiency_utils.py's docstring for why no checkpoint
needs loading except EBAR, and additional/measure_efficiency.py for the
Grainit/Mango/Rapeseed counterpart (identical model configs; this file
only differs in machine/BIN handling and the real synthetic data path).

Writes BOTH machines into one file (the "machine" column distinguishes
them): additional/DanangLeafy6/benchmark/efficiency.csv.

Usage: python3 danangleafy6_measure_efficiency.py [FLAMENIR OCEANFX]
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))  # additional/DanangLeafy6
_ADD = os.path.dirname(_HERE)  # additional/
_ROOT = os.path.dirname(_ADD)  # repo root
sys.path.insert(0, os.path.join(_ROOT, "main"))
sys.path.insert(0, _ADD)  # additional/: merges additional/model|dataset into the namespace packages
sys.path.insert(0, os.path.join(_ADD, "LensNIR"))

import joblib
import pandas as pd
import torch

from benchmark_utils import append_efficiency_row
from efficiency_utils import measure_sklearn_model, measure_torch_model

from model.classification_model import SMARTNIRClassifier, SmartNIRClassificationConfig
from model.regression_model import SMARTNIRRegressor, SmartNIRRegressionConfig
from model.guideddcnet_model import GuidedDCNet, GuidedDCNetConfig
from model.nirmacnet_model import NirMACNet, NirMACNetConfig
from model.xspecmamba_model import XSpecMamba, XSpecMambaConfig

import train as lensnir_train  # additional/LensNIR/train.py: build_model()
from dataset_benchmark import load_benchmark

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SUBSTANCES = ['Thiamethoxam', 'Permethrin', 'Azoxystrobin', 'Difenoconazole', 'Cypermethrin', 'Chlothianidin']
CSV_PATH = f"{_HERE}/benchmark/efficiency.csv"


def bin_wavelengths_count(n_bands: int, bin_factor: int) -> int:
    """Matches every DanangLeafy6 engine's bin_wavelengths(): average-pool
    by bin_factor, then truncate to a multiple of 8."""
    if bin_factor <= 1:
        return n_bands
    return (n_bands // bin_factor) // 8 * 8


def _lensnir_args(n_food, n_targets, n_aux, seq_len):
    class A:
        pass
    a = A()
    a.model_type = "lensnir"
    a.kernels = lensnir_train.DEFAULT_KERNELS
    a.scale_select = "none"
    a.gate_init_prob = 0.832
    a.gate_struct = "hier"
    a.stem = "deep"
    a.sim = "summary"
    a.token_pool = 2
    a.readout = "flat"
    a.readout_rank = 32
    a.task_linear = "reg"
    a.cond_reg = "detach"
    a.fusion = "task"
    a.n_food = n_food
    a.n_targets = n_targets
    a.n_aux = n_aux
    a.c_out = 64
    a.n_layers = 4
    a.n_heads = 4
    a.h_hidden = 128
    a.seq_len = seq_len
    a.use_kan = False
    a.dropout = 0.1
    a.ffn_type = "local_gated"
    a.d_model = 128
    return a


def _find_ebar_checkpoint(machine: str) -> str | None:
    for substance in SUBSTANCES:
        p = f"{_HERE}/checkpoint/substance_regression/stage2_ebar/{machine}/{substance}/{substance}_fold_1.joblib"
        if os.path.exists(p):
            return p
    return None


def measure_machine(machine: str):
    bin_factor = 8 if machine == "OCEANFX" else 1
    full_path = f"{_HERE}/data/{machine}/ALL_synthetic.csv"
    df_head = pd.read_csv(full_path, nrows=0)
    raw_n_bands = len([c for c in df_head.columns if c.startswith("w_")])
    n_bands = bin_wavelengths_count(raw_n_bands, bin_factor)
    n_classes = int(pd.read_csv(full_path, usecols=["category"])["category"].nunique())
    n_targets = len(SUBSTANCES)
    print(f"[{machine}] raw_n_bands={raw_n_bands} BIN={bin_factor} -> n_bands={n_bands} "
          f"n_classes={n_classes} n_targets={n_targets} device={DEVICE}")

    def report(task, method, metrics, notes=""):
        append_efficiency_row(CSV_PATH, machine, task, method, metrics["params_m"], metrics["macs_m"],
                               metrics["latency_ms_mean"], metrics["latency_ms_std"],
                               metrics["throughput_sps"], metrics["batch_size"], metrics["device"], notes)
        print(f"  {task:<14} {method:<12} params={metrics['params_m']:.3f}M macs={metrics['macs_m']:.2f}M "
              f"lat={metrics['latency_ms_mean']:.3f}±{metrics['latency_ms_std']:.3f}ms "
              f"thr={metrics['throughput_sps']:.1f}/s ({metrics['device']}) {notes}")

    # --- classification: SMART-NIR, GuidedDCNet (6-way) ---
    cfg = SmartNIRClassificationConfig(signal_len=n_bands, out_ch_per_branch=64, d_model=128,
                                        depth=3, n_heads=4, classifier="kan", num_classes=n_classes)
    report("classification", "SMART-NIR", measure_torch_model(SMARTNIRClassifier(cfg), (n_bands,), DEVICE))

    gdc = GuidedDCNet(GuidedDCNetConfig(num_classes=n_classes))
    report("classification", "GuidedDCNet",
           measure_torch_model(gdc, (n_bands,), DEVICE, call_fn=lambda m, x: m.reverse_sample(x)),
           notes="full T=100-step reverse diffusion (real inference path)")

    # --- quality: SMART-NIR, GuidedDCNet (binary safe_label) ---
    cfg_q = SmartNIRClassificationConfig(signal_len=n_bands, out_ch_per_branch=64, d_model=128,
                                          depth=3, n_heads=4, classifier="kan", num_classes=2)
    report("quality", "SMART-NIR", measure_torch_model(SMARTNIRClassifier(cfg_q), (n_bands,), DEVICE))
    gdc_q = GuidedDCNet(GuidedDCNetConfig(num_classes=2))
    report("quality", "GuidedDCNet",
           measure_torch_model(gdc_q, (n_bands,), DEVICE, call_fn=lambda m, x: m.reverse_sample(x)),
           notes="full T=100-step reverse diffusion (real inference path)")

    # --- regression: SMART-NIR (stage2), EBAR, NirMACNet, XSpecMamba (joint) ---
    cfg_r = SmartNIRRegressionConfig(signal_len=n_bands, out_ch_per_branch=64, d_model=128,
                                      depth=3, n_heads=4, classifier="kan", num_targets=1)
    report("regression", "SMART-NIR", measure_torch_model(SMARTNIRRegressor(cfg_r), (n_bands,), DEVICE),
           notes="one model/substance; num_targets=1 architecture (identical across substances)")

    ebar_ckpt = _find_ebar_checkpoint(machine)
    if ebar_ckpt:
        loaded = joblib.load(ebar_ckpt)
        fitted = loaded["model"] if isinstance(loaded, dict) else loaded
        n_feat = getattr(fitted, "n_features_in_", n_bands)
        report("regression", "EBAR", measure_sklearn_model(fitted, n_feat),
               notes=f"sklearn StackingRegressor; CPU only; no MACs; timed from {os.path.basename(ebar_ckpt)}")
    else:
        print(f"  regression     EBAR         SKIPPED -- no fold_1 checkpoint retained on disk for {machine}")

    report("regression", "NirMACNet", measure_torch_model(NirMACNet(NirMACNetConfig(num_targets=1)), (n_bands,), DEVICE),
           notes="one model/substance; num_targets=1 architecture (identical across substances)")

    img_shape = (3, 64, 64)
    xsm = XSpecMamba(XSpecMambaConfig(img_size=64, in_chans=3, emb_dim=128, depth=1, n_dirs=4, n_outputs=n_targets))
    report("regression", "XSpecMamba", measure_torch_model(xsm, img_shape, DEVICE),
           notes="joint: one model predicts all 6 substances at once")

    # --- LensNIR: one joint model covers all 3 tasks in a single forward pass ---
    bd = load_benchmark(f"DanangLeafy6_{machine}", verbose=False)
    n_aux = len(bd["aux_classes"])
    args = _lensnir_args(n_food=len(bd["food_classes"]), n_targets=len(bd["target_names"]),
                          n_aux=n_aux, seq_len=bd["seq_len"])
    lens = lensnir_train.build_model(args)
    metrics = measure_torch_model(lens, (bd["seq_len"],), DEVICE, call_fn=lambda m, x: m(x))
    for task in ("classification", "regression", "quality"):
        report(task, "LensNIR", metrics, notes="joint model: 1 forward pass produces all 3 tasks' outputs")

    print(f"[{machine}] -> {CSV_PATH}\n")


if __name__ == "__main__":
    machines = sys.argv[1:] or ["FLAMENIR", "OCEANFX"]
    for m in machines:
        measure_machine(m)
