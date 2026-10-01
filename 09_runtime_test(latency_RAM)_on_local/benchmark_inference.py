"""
benchmark_inference.py
-----------------------
Standalone latency + RAM benchmark for the CubeSat SOH TFLite model.

Runs entirely in Python (no MATLAB/pyenv overhead), feeding random-but-valid
feature vectors through CubeSatSOHPredict.predict_soh() so you get a clean
read on:
  - pure TFLite interpreter.invoke() latency
  - full predict_soh() latency (preprocessing + scaling + inference)
  - process RAM (RSS) before load, after load, and during steady-state inference

Usage:
    pip install psutil          # if not already installed
    python benchmark_inference.py
"""

import os
import time
import numpy as np

try:
    import psutil
except ImportError:
    raise SystemExit("Please install psutil first:  pip install psutil")

from lstm_core_6_tflite import CubeSatSOHPredict

N_WARMUP_CYCLES = 5        # extra calls to fill the sliding window before timing
N_BENCH_CYCLES = 500       # number of timed inference calls


def make_fake_feature_dict(features, cycle):
    """
    Generates a plausible feature dict for benchmarking purposes only.
    Values don't need to be physically realistic — we're timing the
    scaler + interpreter, not validating model accuracy.
    """
    rng = np.random.default_rng(cycle)
    d = {}
    for f in features:
        if f == "cycle":
            d[f] = float(cycle)
        elif "flag" in f:
            d[f] = float(rng.integers(0, 2))
        else:
            d[f] = float(rng.normal(loc=0.0, scale=1.0))
    return d


def main():
    proc = psutil.Process(os.getpid())

    ram_before_load = proc.memory_info().rss / (1024 ** 2)
    print(f"RAM before loading model : {ram_before_load:.2f} MB")

    t_load0 = time.perf_counter()
    predictor = CubeSatSOHPredict()
    t_load1 = time.perf_counter()

    ram_after_load = proc.memory_info().rss / (1024 ** 2)
    print(f"RAM after loading model  : {ram_after_load:.2f} MB "
          f"(+{ram_after_load - ram_before_load:.2f} MB)")
    print(f"Model load time          : {(t_load1 - t_load0) * 1000:.2f} ms\n")

    n_total_calls = N_WARMUP_CYCLES + predictor.seq_len + N_BENCH_CYCLES

    print(f"Running {n_total_calls} predict_soh() calls "
          f"({predictor.seq_len} to fill the window, "
          f"then {N_BENCH_CYCLES} timed inferences)...\n")

    for cycle in range(1, n_total_calls + 1):
        feat = make_fake_feature_dict(predictor.features, cycle)
        predictor.predict_soh(feat)

    predictor.print_performance_report(exclude_warmup=3)

    ram_end = proc.memory_info().rss / (1024 ** 2)
    print(f"RAM at end of benchmark  : {ram_end:.2f} MB")


if __name__ == "__main__":
    main()
