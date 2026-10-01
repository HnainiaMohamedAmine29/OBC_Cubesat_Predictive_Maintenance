import os
import time
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


class CubeSatSOHPredict:
    def __init__(self,
                 model_path='soh_model_SIMPLE.tflite',
                 scaler_X_path='scaler_X.pkl',
                 features_path='features_used.pkl'):

        # --- Load TFLite model via the Interpreter (NOT tf.keras.models.load_model,
        #     which only understands SavedModel/H5, not the .tflite FlatBuffer) ---
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.interpreter.reset_all_variables()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # This model's real signature is (1, 30, 20) -> (1, 1)
        in_shape = self.input_details[0]['shape']
        self.seq_len = int(in_shape[1])
        self.n_features = int(in_shape[2])

        self.scaler_X = joblib.load(scaler_X_path)
        self.features = joblib.load(features_path)

        if len(self.features) != self.n_features:
            raise ValueError(
                f"❌ features_used.pkl has {len(self.features)} features but "
                f"model expects {self.n_features}."
            )

        self.buffer = []

        # ==================================================================
        # PERFORMANCE / RESOURCE TRACKING
        # ==================================================================
        self.model_size_mb = os.path.getsize(model_path) / (1024 ** 2)

        if _HAS_PSUTIL:
            self._proc = psutil.Process(os.getpid())
        else:
            self._proc = None
            print("⚠️  psutil not installed — RAM tracking disabled. "
                  "Install with: pip install psutil")

        # Per-call records (only appended once the sliding window is full,
        # i.e. once an actual model invocation happens)
        self._t_total_ms = []       # preprocessing + scaling + inference
        self._t_infer_ms = []       # pure interpreter.invoke() only
        self._ram_mb = []           # process RSS sampled right after invoke

        print(f"✅ TFLite LSTM loaded | Window = {self.seq_len} | Features = {self.n_features}")
        print(f"   Input:  {self.input_details[0]['shape']} ({self.input_details[0]['dtype']})")
        print(f"   Output: {self.output_details[0]['shape']} ({self.output_details[0]['dtype']})")
        print(f"   Model file size: {self.model_size_mb:.3f} MB")

        print(type(self.scaler_X))
        print(self.scaler_X.mean_[:5])
        print(self.scaler_X.scale_[:5])

    def predict_soh(self, feature_dict):

        t0 = time.perf_counter()

        # 1. Check features
        missing = [f for f in self.features if f not in feature_dict]
        if missing:
            raise ValueError(f"❌ Missing features: {missing}")

        # 2. Build dataframe in correct order
        df = pd.DataFrame([[feature_dict[f] for f in self.features]],
                          columns=self.features).astype(np.float32)

        # 3. Scale
        feat_scaled = self.scaler_X.transform(df)[0]

        # 4. Buffer
        self.buffer.append(feat_scaled)

        if len(self.buffer) > self.seq_len:
            self.buffer.pop(0)

        if len(self.buffer) < self.seq_len:
            return None

        # 5. Sequence
        X_seq = np.array(self.buffer, dtype=np.float32).reshape(1, self.seq_len, self.n_features)

        # 6. Predict via TFLite Interpreter (this is the part we call "pure inference")
        t_inf0 = time.perf_counter()
        self.interpreter.set_tensor(self.input_details[0]['index'], X_seq)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_details[0]['index'])
        t_inf1 = time.perf_counter()

        pred = float(out[0][0])
        pred = float(np.clip(pred, 0.0, 1.05))

        t1 = time.perf_counter()

        # ---- record metrics for this call ----
        self._t_total_ms.append((t1 - t0) * 1000.0)
        self._t_infer_ms.append((t_inf1 - t_inf0) * 1000.0)
        if self._proc is not None:
            self._ram_mb.append(self._proc.memory_info().rss / (1024 ** 2))

        return pred

    # ======================================================================
    # PERFORMANCE REPORTING
    # ======================================================================
    def get_performance_report(self, exclude_warmup=1):
        """
        Returns a dict summarizing latency (ms) and RAM usage (MB) across
        all calls that actually ran the TFLite interpreter.

        exclude_warmup: number of initial calls to exclude from the
                        "steady_state" stats (first call(s) after allocate_tensors
                        are typically slower due to lazy init / caching).
        """
        n = len(self._t_total_ms)
        if n == 0:
            return {"n_inferences": 0, "message": "No inferences recorded yet."}

        total = np.array(self._t_total_ms)
        infer = np.array(self._t_infer_ms)

        def _stats(arr):
            return {
                "mean_ms": float(np.mean(arr)),
                "std_ms": float(np.std(arr)),
                "min_ms": float(np.min(arr)),
                "max_ms": float(np.max(arr)),
                "p50_ms": float(np.percentile(arr, 50)),
                "p95_ms": float(np.percentile(arr, 95)),
                "p99_ms": float(np.percentile(arr, 99)),
            }

        report = {
            "n_inferences": n,
            "model_size_mb": self.model_size_mb,
            "total_latency_all_calls": _stats(total),   # preprocessing + inference
            "pure_inference_all_calls": _stats(infer),  # interpreter.invoke() only
        }

        if n > exclude_warmup:
            report["total_latency_steady_state"] = _stats(total[exclude_warmup:])
            report["pure_inference_steady_state"] = _stats(infer[exclude_warmup:])

        if self._ram_mb:
            ram = np.array(self._ram_mb)
            report["ram_mb"] = {
                "mean": float(np.mean(ram)),
                "min": float(np.min(ram)),
                "max_peak": float(np.max(ram)),
                "current": float(ram[-1]),
            }
        else:
            report["ram_mb"] = None
            report["ram_note"] = "psutil not available — install psutil for RAM tracking."

        return report

    def print_performance_report(self, exclude_warmup=1):
        r = self.get_performance_report(exclude_warmup=exclude_warmup)
        if r.get("n_inferences", 0) == 0:
            print(r.get("message", "No data."))
            return

        print("\n" + "=" * 60)
        print("  LSTM SOH MODEL — INFERENCE PERFORMANCE REPORT")
        print("=" * 60)
        print(f"  Model file size        : {r['model_size_mb']:.3f} MB")
        print(f"  Number of inferences   : {r['n_inferences']}")

        t = r["total_latency_all_calls"]
        i = r["pure_inference_all_calls"]
        print("\n  -- Latency (ALL calls, incl. warm-up) --")
        print(f"  Total (preproc+infer)  : mean {t['mean_ms']:.3f} ms | "
              f"p95 {t['p95_ms']:.3f} ms | max {t['max_ms']:.3f} ms")
        print(f"  Pure interpreter.invoke: mean {i['mean_ms']:.3f} ms | "
              f"p95 {i['p95_ms']:.3f} ms | max {i['max_ms']:.3f} ms")

        if "total_latency_steady_state" in r:
            ts = r["total_latency_steady_state"]
            is_ = r["pure_inference_steady_state"]
            print("\n  -- Latency (steady-state, warm-up excluded) --")
            print(f"  Total (preproc+infer)  : mean {ts['mean_ms']:.3f} ms | "
                  f"std {ts['std_ms']:.3f} | p95 {ts['p95_ms']:.3f} ms")
            print(f"  Pure interpreter.invoke: mean {is_['mean_ms']:.3f} ms | "
                  f"std {is_['std_ms']:.3f} | p95 {is_['p95_ms']:.3f} ms")

        if r["ram_mb"] is not None:
            ram = r["ram_mb"]
            print("\n  -- Process RAM (RSS, whole Python process) --")
            print(f"  Mean   : {ram['mean']:.2f} MB")
            print(f"  Peak   : {ram['max_peak']:.2f} MB")
            print(f"  Current: {ram['current']:.2f} MB")
        else:
            print("\n  -- RAM tracking disabled (psutil not installed) --")
        print("=" * 60 + "\n")

    def get_latency_history(self):
        """Returns (total_ms_list, pure_infer_ms_list, ram_mb_list) for plotting."""
        return list(self._t_total_ms), list(self._t_infer_ms), list(self._ram_mb)

    def reset_performance_stats(self):
        self._t_total_ms.clear()
        self._t_infer_ms.clear()
        self._ram_mb.clear()


if __name__ == "__main__":
    pred = CubeSatSOHPredict()
    print("✅ Predictor READY")
