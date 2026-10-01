"""
CubeSatSOHPredict v8  -  PC test-bench predictor with latency / RAM tracking.

= lstm_core_7_tflite.py (perf tracking, same MATLAB-facing API)
+ int8 / float32 .tflite support (reads scale / zero-point from the tensors)
+ optional 2-number output correction (soh_output_fix.json) from fit_output_fix.py
+ NO offset, NO EMA, NO monotone clamp, NO warm-up gate: raw model output.

API used by cubesat_main*.m (unchanged):
    predict_soh(dict) -> float | None   (None until the 30-row window is full,
                                         unless pad_start=True)
    print_performance_report(), get_performance_report(),
    get_latency_history() -> (total_ms, infer_ms, ram_mb), reset_performance_stats()
"""
import os, time, json
import numpy as np
try:
    from ai_edge_litert.interpreter import Interpreter as _Interpreter
except ImportError:
    import tensorflow as tf
    _Interpreter = tf.lite.Interpreter
import joblib
try:
    import psutil
except ImportError:
    psutil = None


def estimate_model_memory(interpreter):
    """Static estimate from the graph (not a measurement): weights bytes and the
    peak size of live activation tensors assuming ideal memory reuse. The real
    TFLite-Micro arena is larger (planner overhead, per-node bookkeeping), so
    treat the activation figure as a LOWER bound."""
    try:
        T = {t['index']: t for t in interpreter.get_tensor_details()}
        ops = interpreter._get_ops_details()
        produced = {int(o) for op in ops for o in op['outputs']}
        inputs = {int(d['index']) for d in interpreter.get_input_details()}
        size = lambda i: int(np.prod(T[i]['shape'])) * np.dtype(T[i]['dtype']).itemsize
        first, last = {}, {}
        for k, op in enumerate(ops):
            for o in op['outputs']:
                o = int(o); first.setdefault(o, k); last[o] = max(last.get(o, k), k)
            for i in op['inputs']:
                i = int(i)
                if i in produced or i in inputs:
                    last[i] = max(last.get(i, k), k)
        for i in inputs: first[i] = 0
        live = np.zeros(len(ops) + 1)
        for i in first:
            if i in T and (i in produced or i in inputs):
                live[first[i]:last.get(i, first[i]) + 1] += size(i)
        weights = sum(size(i) for i in T if i not in produced and i not in inputs
                      and len(T[i]['shape']) > 0)
        return {'n_ops': len(ops), 'n_tensors': len(T),
                'weights_kb': weights / 1024, 'peak_activations_kb_lower_bound': float(live.max()) / 1024}
    except Exception as e:                      # private API - never break the run
        return {'error': str(e)}


class CubeSatSOHPredict:
    def __init__(self, model_path='cubesat_soh_lstm__int8.tflite',
                 scaler_X_path='scaler_X.pkl', features_path='features_used.pkl',
                 fix_path='soh_output_fix.json', pad_start=False):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"model_path='{model_path}' not found in {os.getcwd()}")
        self._proc = psutil.Process(os.getpid()) if psutil else None
        self.ram_before_load_mb = self._proc.memory_info().rss / 2**20 if self._proc else None

        self.interp = _Interpreter(model_path=model_path)
        self.interp.allocate_tensors()
        self.interp.reset_all_variables()
        self.i = self.interp.get_input_details()[0]
        self.o = self.interp.get_output_details()[0]
        self.seq_len, self.n_features = int(self.i['shape'][1]), int(self.i['shape'][2])

        self.scaler = joblib.load(scaler_X_path)
        self.features = list(joblib.load(features_path))
        if len(self.features) != self.n_features or len(self.scaler.mean_) != self.n_features:
            raise ValueError("features_used.pkl / scaler_X.pkl do not match the model's feature count")
        self._mean = self.scaler.mean_.astype(np.float32)
        self._scale = self.scaler.scale_.astype(np.float32)

        self.out_min, self.a, self.b = 0.0, 1.0, 0.0     # defaults = no correction
        if fix_path and os.path.exists(fix_path):
            fx = json.load(open(fix_path))
            self.out_min, self.a, self.b = fx['out_min'], fx['a'], fx['b']

        self.pad_start = pad_start
        self.buffer = []
        self.last_raw = None
        self.model_size_kb = os.path.getsize(model_path) / 1024
        self.ram_after_load_mb = self._proc.memory_info().rss / 2**20 if self._proc else None
        self.static_mem = estimate_model_memory(self.interp)
        self._t_total, self._t_infer, self._ram = [], [], []

        print(f"LSTM loaded | window={self.seq_len} features={self.n_features} | "
              f"in={self.i['dtype'].__name__} out={self.o['dtype'].__name__} | {self.model_size_kb:.1f} KB | "
              f"fix={'on' if (self.a, self.b, self.out_min) != (1.0, 0.0, 0.0) else 'off'}")

    def reset(self):
        self.buffer, self.last_raw = [], None

    def predict_soh(self, feature_dict):
        t0 = time.perf_counter()
        missing = [f for f in self.features if f not in feature_dict]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        x = np.array([feature_dict[f] for f in self.features], dtype=np.float32)
        z = (x - self._mean) / self._scale

        if not self.buffer and self.pad_start:
            self.buffer = [z.copy() for _ in range(self.seq_len)]
        else:
            self.buffer.append(z)
            if len(self.buffer) > self.seq_len:
                self.buffer.pop(0)
        if len(self.buffer) < self.seq_len:
            return None

        X = np.asarray(self.buffer, dtype=np.float32).reshape(1, self.seq_len, self.n_features)
        if self.i['dtype'] == np.int8:
            s, zp = self.i['quantization']
            X = np.clip(np.round(X / s + zp), -128, 127).astype(np.int8)

        t_i0 = time.perf_counter()
        self.interp.set_tensor(self.i['index'], X)
        self.interp.invoke()
        out = self.interp.get_tensor(self.o['index'])
        t_i1 = time.perf_counter()

        v = float(out.ravel()[0])
        if self.o['dtype'] == np.int8:
            s, zp = self.o['quantization']
            v = (v - zp) * s
        self.last_raw = v + self.out_min
        pred = self.a * self.last_raw + self.b

        self._t_total.append((time.perf_counter() - t0) * 1e3)
        self._t_infer.append((t_i1 - t_i0) * 1e3)
        if self._proc: self._ram.append(self._proc.memory_info().rss / 2**20)
        return pred

    # ---------------- reporting ----------------
    @staticmethod
    def _stats(a):
        a = np.asarray(a)
        return {'mean_ms': float(a.mean()), 'std_ms': float(a.std()), 'min_ms': float(a.min()),
                'p50_ms': float(np.percentile(a, 50)), 'p95_ms': float(np.percentile(a, 95)),
                'p99_ms': float(np.percentile(a, 99)), 'max_ms': float(a.max())}

    def get_performance_report(self, exclude_warmup=3):
        n = len(self._t_total)
        if n == 0:
            return {'n_inferences': 0}
        r = {'n_inferences': n, 'model_size_kb': self.model_size_kb,
             'total_latency_all': self._stats(self._t_total),
             'pure_inference_all': self._stats(self._t_infer)}
        if n > exclude_warmup:
            r['total_latency_steady'] = self._stats(self._t_total[exclude_warmup:])
            r['pure_inference_steady'] = self._stats(self._t_infer[exclude_warmup:])
        if self._ram:
            r['ram_mb'] = {'before_load': self.ram_before_load_mb, 'after_load': self.ram_after_load_mb,
                           'model_load_delta': self.ram_after_load_mb - self.ram_before_load_mb,
                           'mean': float(np.mean(self._ram)), 'peak': float(np.max(self._ram)),
                           'growth_during_run': float(self._ram[-1] - self._ram[0])}
        r['static_memory_estimate'] = self.static_mem
        return r

    def print_performance_report(self, exclude_warmup=3):
        r = self.get_performance_report(exclude_warmup)
        if r['n_inferences'] == 0:
            print('No inferences recorded.'); return
        k = 'steady' if 'total_latency_steady' in r else 'all'
        t, p = r[f'total_latency_{k}'], r[f'pure_inference_{k}']
        line = '=' * 64
        print(f"\n{line}\n  LSTM SOH - PC TEST-BENCH PERFORMANCE ({r['n_inferences']} inferences, "
              f"{'warm-up excluded' if k == 'steady' else 'all calls'})\n{line}")
        print(f"  model file            : {r['model_size_kb']:.1f} KB")
        print(f"  total  (pre+infer)    : mean {t['mean_ms']:.3f} | p50 {t['p50_ms']:.3f} | p95 {t['p95_ms']:.3f} | p99 {t['p99_ms']:.3f} | max {t['max_ms']:.3f} ms")
        print(f"  pure interpreter.invoke: mean {p['mean_ms']:.3f} | p50 {p['p50_ms']:.3f} | p95 {p['p95_ms']:.3f} | p99 {p['p99_ms']:.3f} | max {p['max_ms']:.3f} ms")
        if 'ram_mb' in r:
            m = r['ram_mb']
            print(f"  Python process RSS    : before load {m['before_load']:.1f} | after load {m['after_load']:.1f} "
                  f"(+{m['model_load_delta']:.1f}) | peak {m['peak']:.1f} MB | growth during run {m['growth_during_run']:+.2f} MB")
            print("    (RSS includes TensorFlow/Python itself - it is NOT the MCU RAM need)")
        s = r['static_memory_estimate']
        if 'error' not in s:
            print(f"  graph estimate (MCU)  : weights {s['weights_kb']:.0f} KB (flash) | peak activations >= {s['peak_activations_kb_lower_bound']:.0f} KB "
                  f"(RAM, lower bound) | {s['n_ops']} ops")
        print("  PC latency does not transfer to the STM32 - measure there with the DWT cycle counter.\n" + line + "\n")

    def get_latency_history(self):
        return list(self._t_total), list(self._t_infer), list(self._ram)

    def reset_performance_stats(self):
        self._t_total.clear(); self._t_infer.clear(); self._ram.clear()


if __name__ == '__main__':
    print('Predictor module OK')
