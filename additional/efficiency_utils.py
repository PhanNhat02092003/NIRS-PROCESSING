"""Shared inference-efficiency measurement for additional/{Dataset}/benchmark/
efficiency*.csv -- params, MACs, single-sample latency and batch throughput,
alongside the accuracy/R2/quality numbers already in benchmark/{classification,
regression,quality}/*.csv.

These are architecture-level numbers: MACs/params/latency/throughput depend
only on the model's shape and the device it runs on, not on its learned
weight *values* -- so every torch.nn.Module here is measured freshly
constructed (no checkpoint load needed, and no risk of accidentally loading
a checkpoint from an incompatible fork/branch). The one exception is EBAR
(additional/model/ebar_model.py's sklearn StackingRegressor): a classical-ML
ensemble's predict() latency depends on what actually got fitted (e.g. an
SVR's number of support vectors), and it has no MACs/FLOPs in the usual
sense, so measure_sklearn_model requires a real fitted (joblib-loaded) model
and always returns NaN for params/macs.
"""

import time

import numpy as np
import torch
import torch.nn as nn


def count_params(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def _macs_via_thop(model, call_fn, inputs):
    """call_fn(model, x) must trigger the real inference path (plain
    forward(), or a custom method like GuidedDCNet's reverse_sample()) --
    `model` is kept as a real registered submodule (not just closed over)
    so thop's module-tree walk actually finds its Conv/Linear/etc. layers
    and fires their hooks; a bare lambda/closure wrapper has no submodules
    and silently reports 0 MACs."""
    import thop

    class _Wrapper(nn.Module):
        def __init__(self, model, fn):
            super().__init__()
            self.model = model
            self._fn = fn

        def forward(self, *a):
            return self._fn(self.model, *a)

    wrapper = _Wrapper(model, call_fn)
    macs, _ = thop.profile(wrapper, inputs=inputs, verbose=False)
    return macs / 1e6  # -> millions of MACs


@torch.no_grad()
def measure_torch_model(model: nn.Module, input_shape: tuple, device: str,
                         call_fn=None, n_warmup: int = 10, n_iters: int = 50,
                         throughput_batch: int = 64) -> dict:
    """input_shape: shape of ONE sample's input tensor, e.g. (seq_len,) or
    (3, img_size, img_size) -- batch dim is prepended here.
    call_fn(model, x) defaults to model(x); pass e.g.
    lambda m, x: m.reverse_sample(x) for GuidedDCNet."""
    model = model.to(device).eval()
    call_fn = call_fn or (lambda m, x: m(x))

    params_m = count_params(model)

    x1 = torch.randn((1, *input_shape), device=device)
    try:
        macs_m = _macs_via_thop(model, call_fn, (x1,))
    except Exception as e:
        macs_m = float("nan")
        print(f"  [warn] thop MACs counting failed ({e}); macs_M=NaN")
    model.eval()  # thop's internal wrapper.train(prev_status) can flip submodules
    # (e.g. BatchNorm1d) back to train mode as a side effect of profiling.

    # Latency: batch=1, wall-clock per call, GPU-synchronized.
    for _ in range(n_warmup):
        call_fn(model, x1)
    if device == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        call_fn(model, x1)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    latency_ms_mean, latency_ms_std = float(np.mean(times)), float(np.std(times))

    # Throughput: larger batch, samples/sec.
    xb = torch.randn((throughput_batch, *input_shape), device=device)
    for _ in range(max(3, n_warmup // 2)):
        call_fn(model, xb)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    n_rep = max(5, n_iters // 5)
    for _ in range(n_rep):
        call_fn(model, xb)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    throughput_sps = (throughput_batch * n_rep) / elapsed

    return dict(params_m=params_m, macs_m=macs_m, latency_ms_mean=latency_ms_mean,
                latency_ms_std=latency_ms_std, throughput_sps=throughput_sps,
                batch_size=throughput_batch, device=device)


def measure_sklearn_model(fitted_model, n_features: int, n_warmup: int = 5,
                           n_iters: int = 30, throughput_batch: int = 256) -> dict:
    """fitted_model: an already-fit()/joblib-loaded estimator with .predict().
    CPU only (sklearn has no GPU path here)."""
    rng = np.random.default_rng(0)
    x1 = rng.standard_normal((1, n_features)).astype(np.float32)
    for _ in range(n_warmup):
        fitted_model.predict(x1)
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fitted_model.predict(x1)
        times.append((time.perf_counter() - t0) * 1000.0)
    latency_ms_mean, latency_ms_std = float(np.mean(times)), float(np.std(times))

    xb = rng.standard_normal((throughput_batch, n_features)).astype(np.float32)
    for _ in range(3):
        fitted_model.predict(xb)
    t0 = time.perf_counter()
    n_rep = 10
    for _ in range(n_rep):
        fitted_model.predict(xb)
    elapsed = time.perf_counter() - t0
    throughput_sps = (throughput_batch * n_rep) / elapsed

    return dict(params_m=float("nan"), macs_m=float("nan"),
                latency_ms_mean=latency_ms_mean, latency_ms_std=latency_ms_std,
                throughput_sps=throughput_sps, batch_size=throughput_batch, device="cpu")
