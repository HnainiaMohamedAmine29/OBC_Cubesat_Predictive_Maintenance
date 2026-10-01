"""
Standalone latency + RAM benchmark (no MATLAB overhead).
Fixes vs benchmark_inference.py: imports the v8 predictor (which has the report
methods), feeds REAL feature rows when raw_features.csv is available (so int8
clipping / data-dependent paths behave as in the test bench), prints percentiles.

python benchmark_inference_fixed.py [--model cubesat_soh_lstm_v3_int8.tflite] [--n 500]
"""
import argparse, os, numpy as np
from lstm_core_8_tflite import CubeSatSOHPredict

p = argparse.ArgumentParser()
p.add_argument('--model', default='cubesat_soh_lstm_v3_int8.tflite')
p.add_argument('--n', type=int, default=500); p.add_argument('--raw', default='raw_features.csv')
a = p.parse_args()

pred = CubeSatSOHPredict(model_path=a.model)
if os.path.exists(a.raw):
    import pandas as pd
    rows = pd.read_csv(a.raw)[pred.features].to_dict('records'); src = a.raw
else:
    rng = np.random.default_rng(0); src = 'synthetic'
    rows = [{f: float(rng.normal()) for f in pred.features} for _ in range(a.n + 60)]
print(f'feeding {src}')
for k in range(pred.seq_len + a.n):
    pred.predict_soh(rows[k % len(rows)])
pred.print_performance_report(exclude_warmup=5)
