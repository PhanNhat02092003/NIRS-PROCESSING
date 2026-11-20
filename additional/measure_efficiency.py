"""Inference-efficiency benchmark (params, MACs, single-sample latency,
batch throughput) for every method already in additional/{Dataset}/
benchmark/{classification,regression,quality}/*.csv -- Grainit, Mango,
Rapeseed. See efficiency_utils.py's docstring for why no checkpoint needs
loading except EBAR, and DanangLeafy6_measure_efficiency.py for the
DanangLeafy6 (per-machine, BIN-aware) counterpart.

Writes one row per (task, method) to additional/{Dataset}/benchmark/
efficiency.csv (machine column = the dataset name itself, so this shares
its schema/append function with DanangLeafy6's two-machine file).

Usage: python3 measure_efficiency.py Grainit [Mango Rapeseed ...]
"""
import glob
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))  # additional/
_ROOT = os.path.dirname(_HERE)  # repo root
sys.path.insert(0, os.path.join(_ROOT, "main"))
sys.path.insert(0, _HERE)  # additional/: merges additional/model|dataset into the namespace packages
sys.path.insert(0, os.path.join(_HERE, "LensNIR"))

import joblib
import torch

from benchmark_utils import append_efficiency_row
from efficiency_utils import measure_sklearn_model, measure_torch_model

from model.classification_model import SMARTNIRClassifier, SmartNIRClassificationConfig
from model.regression_model import SMARTNIRRegressor, SmartNIRRegressionConfig
from model.guideddcnet_model import GuidedDCNet, GuidedDCNetConfig
from model.nirmacnet_model import NirMACNet, NirMACNetConfig
from model.xspecmamba_model import XSpecMamba, XSpecMambaConfig

# dataset_benchmark.py's DEFAULT_DATA_ROOT (computed at import time from
# __file__, not repo-root-relative) doesn't match where this machine's
# Grainit/Mango/Rapeseed data actually lives -- must be set before the
# import below, matching whatever DATA_ROOT was set to when those
# datasets' LensNIR "main" checkpoints were originally trained.
os.environ.setdefault("DATA_ROOT", "/home/phannhat/all-dataset")

import train as lensnir_train  # additional/LensNIR/train.py: build_model()
from dataset_benchmark import load_benchmark

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOTS = {
    "Grainit":  os.path.join(_ROOT, "..", "all-dataset", "Zenodo-Grainit", "Grainit", "ALL.csv"),
    "Mango":    os.path.join(_ROOT, "..", "all-dataset", "Mendeley-Mango", "Mango", "ALL.csv"),
    "Rapeseed": os.path.join(_ROOT, "..", "all-dataset", "RechercheDataGouv-Rapeseed", "Rapeseed", "ALL.csv"),
}


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


def _find_ebar_checkpoint(dataset: str) -> str | None:
    pattern = f"{_HERE}/{dataset}/checkpoint/substance_regression/stage2_ebar/{dataset}/*/*_fold_1.joblib"
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def measure_dataset(dataset: str):
    csv_path = f"{_HERE}/{dataset}/benchmark/efficiency.csv"
    full_path = ROOTS[dataset]
    n_classes = int(__import__("pandas").read_csv(full_path, usecols=["category"])["category"].nunique())
    n_bands = len([c for c in __import__("pandas").read_csv(full_path, nrows=0).columns if c.startswith("w_")])
    targets = {
        "Grainit": ["Moisture", "Protein"],
        "Mango": ["dry_matter"],
        "Rapeseed": ["N_content", "C_content"],
    }[dataset]
    n_targets = len(targets)
    print(f"[{dataset}] n_classes={n_classes} n_bands={n_bands} n_targets={n_targets} device={DEVICE}")

    def report(task, method, metrics, notes=""):
        append_efficiency_row(csv_path, dataset, task, method, metrics["params_m"], metrics["macs_m"],
                               metrics["latency_ms_mean"], metrics["latency_ms_std"],
                               metrics["throughput_sps"], metrics["batch_size"], metrics["device"], notes)
        print(f"  {task:<14} {method:<12} params={metrics['params_m']:.3f}M macs={metrics['macs_m']:.2f}M "
              f"lat={metrics['latency_ms_mean']:.3f}±{metrics['latency_ms_std']:.3f}ms "
              f"thr={metrics['throughput_sps']:.1f}/s ({metrics['device']}) {notes}")

    # --- classification: SMART-NIR, GuidedDCNet ---
    cfg = SmartNIRClassificationConfig(signal_len=n_bands, out_ch_per_branch=64, d_model=128,
                                        depth=3, n_heads=4, classifier="kan", num_classes=n_classes)
    report("classification", "SMART-NIR", measure_torch_model(SMARTNIRClassifier(cfg), (n_bands,), DEVICE))

    gdc = GuidedDCNet(GuidedDCNetConfig(num_classes=n_classes))
    report("classification", "GuidedDCNet",
           measure_torch_model(gdc, (n_bands,), DEVICE, call_fn=lambda m, x: m.reverse_sample(x)),
           notes="full T=100-step reverse diffusion (real inference path)")

    # --- quality: SMART-NIR, GuidedDCNet (always binary PASS/FAIL or regime) ---
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

    ebar_ckpt = _find_ebar_checkpoint(dataset)
    if ebar_ckpt:
        loaded = joblib.load(ebar_ckpt)
        fitted = loaded["model"] if isinstance(loaded, dict) else loaded
        n_feat = getattr(fitted, "n_features_in_", n_bands)
        report("regression", "EBAR", measure_sklearn_model(fitted, n_feat),
               notes=f"sklearn StackingRegressor; CPU only; no MACs; timed from {os.path.basename(ebar_ckpt)}")
    else:
        print(f"  regression     EBAR         SKIPPED -- no fold_1 checkpoint retained on disk for {dataset}")

    report("regression", "NirMACNet", measure_torch_model(NirMACNet(NirMACNetConfig(num_targets=1)), (n_bands,), DEVICE),
           notes="one model/substance; num_targets=1 architecture (identical across substances)")

    img_shape = (3, 64, 64)
    xsm = XSpecMamba(XSpecMambaConfig(img_size=64, in_chans=3, emb_dim=128, depth=1, n_dirs=4, n_outputs=n_targets))
    report("regression", "XSpecMamba", measure_torch_model(xsm, img_shape, DEVICE),
           notes="joint: one model predicts all targets at once")

    # --- LensNIR: one joint model covers all 3 tasks in a single forward pass ---
    bd = load_benchmark(dataset, verbose=False)
    n_aux = len(bd["aux_classes"])
    args = _lensnir_args(n_food=len(bd["food_classes"]), n_targets=len(bd["target_names"]),
                          n_aux=n_aux, seq_len=bd["seq_len"])
    lens = lensnir_train.build_model(args)
    metrics = measure_torch_model(lens, (bd["seq_len"],), DEVICE,
                                   call_fn=lambda m, x: m(x))
    for task in ("classification", "regression", "quality"):
        report(task, "LensNIR", metrics, notes="joint model: 1 forward pass produces all 3 tasks' outputs")

    print(f"[{dataset}] -> {csv_path}\n")


if __name__ == "__main__":
    datasets = sys.argv[1:] or ["Grainit", "Mango", "Rapeseed"]
    for ds in datasets:
        measure_dataset(ds)
