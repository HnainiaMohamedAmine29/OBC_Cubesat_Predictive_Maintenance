import os
import json
import numpy as np
try:
    from ai_edge_litert.interpreter import Interpreter as _Interpreter
except ImportError:
    import tensorflow as tf
    _Interpreter = tf.lite.Interpreter
import joblib
import pandas as pd


class CubeSatSOHPredict:
    def __init__(self,
                 model_path='cubesat_soh_lstm_v3_22.tflite',
                 scaler_X_path='scaler_X.pkl',
                 features_path='features_used.pkl',
                 out_min=None,
                 out_max=None,
                 warmup_cycles=30,     # output nominal SOH (1.0) until model has real history
                 startup_neutral=10,   # first N rows: OOD features set to training mean (z=0)
                 ema_alpha=0.02,       # slow EMA on the raw model output (0..1, 1 = off)
                 monotonic=True):      # SOH cannot increase -> clamp to running minimum
        # out_min/out_max MUST match --out_min/--out_max used in
        # optimize_v3_int8.py when this .tflite was exported - loading the
        # wrong values here silently produces wrong SOH predictions (the
        # .tflite still allocates and runs fine either way).
        #
        # To remove that failure mode: if optimize_v3_int8.py wrote a
        # "<model>_meta.json" sidecar next to model_path, read out_min/out_max
        # from THAT instead of trusting a hardcoded default here. Only fall
        # back to the out_min/out_max arguments (or the 0.60/1.05 defaults)
        # if no sidecar is found, and say so loudly - silently guessing here
        # is exactly what caused the last bad result.
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"❌ model_path='{model_path}' does not exist in {os.getcwd()}. "
                f"This is checked explicitly so a stale/wrong filename fails loudly "
                f"here, instead of silently loading a different .tflite you didn't "
                f"intend, or a cryptic TFLite error later.")
        print(f"Loading model from: {os.path.abspath(model_path)}")

        meta_path = os.path.splitext(model_path)[0] + '_meta.json'
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            self.out_min = meta['out_min']
            self.out_max = meta['out_max']
            print(f"✅ Loaded {meta_path}: out_min={self.out_min}, out_max={self.out_max} "
                  f"(these came from the export run, not a hardcoded default)")
            if out_min is not None and out_min != self.out_min:
                print(f"⚠️  You passed out_min={out_min} but the sidecar says {self.out_min} - "
                      f"using the sidecar's value, since it's what this .tflite was ACTUALLY built with.")
            if out_max is not None and out_max != self.out_max:
                print(f"⚠️  You passed out_max={out_max} but the sidecar says {self.out_max} - "
                      f"using the sidecar's value, since it's what this .tflite was ACTUALLY built with.")
        else:
            self.out_min = 0.60 if out_min is None else out_min
            self.out_max = 1.05 if out_max is None else out_max
            print(f"⚠️  No {meta_path} found next to {model_path} - falling back to "
                  f"out_min={self.out_min}, out_max={self.out_max}. If this .tflite was exported "
                  f"with different --out_min/--out_max values, predictions will be silently wrong. "
                  f"Re-export with the current optimize_v3_int8.py to get the sidecar file.")

        # --- Load TFLite model via the Interpreter (NOT tf.keras.models.load_model,
        #     which only understands SavedModel/H5, not the .tflite FlatBuffer) ---
        self.interpreter = _Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.interpreter.reset_all_variables()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # This model's real signature is (1, 30, 20) -> (1, 1)
        in_shape = self.input_details[0]['shape']
        self.seq_len = int(in_shape[1])
        self.n_features = int(in_shape[2])

        # --- INT8 quantization params (model was exported with
        #     inference_input_type = inference_output_type = tf.int8, see
        #     optimize_v3_int8.py). set_tensor()/get_tensor() work in the
        #     quantized int8 domain, so every call must convert float <-> int8
        #     using these two (scale, zero_point) pairs. ---
        self.input_is_int8 = (self.input_details[0]['dtype'] == np.int8)
        self.output_is_int8 = (self.output_details[0]['dtype'] == np.int8)
        self.in_scale, self.in_zp = self.input_details[0]['quantization']
        self.out_scale, self.out_zp = self.output_details[0]['quantization']

        self.scaler_X = joblib.load(scaler_X_path)
        self.features = joblib.load(features_path)

        if len(self.features) != self.n_features:
            raise ValueError(
                f"❌ features_used.pkl has {len(self.features)} features but "
                f"model expects {self.n_features}."
            )

        self.buffer = []
        self.warmup_cycles = warmup_cycles
        self.startup_neutral = startup_neutral
        self.ema_alpha = ema_alpha
        self.monotonic = monotonic
        self._n = 0            # rows seen
        self._ema = None
        self._out = 1.0        # last reported (filtered) SOH
        self.last_raw = None   # unfiltered model output, for debugging

        print(f"✅ TFLite LSTM loaded | Window = {self.seq_len} | Features = {self.n_features}")
        print(f"   Input:  {self.input_details[0]['shape']} ({self.input_details[0]['dtype']})"
              + (f"  scale={self.in_scale:.8f} zp={self.in_zp}" if self.input_is_int8 else ""))
        print(f"   Output: {self.output_details[0]['shape']} ({self.output_details[0]['dtype']})"
              + (f"  scale={self.out_scale:.8f} zp={self.out_zp}" if self.output_is_int8 else ""))

        print(type(self.scaler_X))
        print(self.scaler_X.mean_[:5])
        print(self.scaler_X.scale_[:5])

    def predict_soh(self, feature_dict):

        # 1. Check features
        missing = [f for f in self.features if f not in feature_dict]
        if missing:
            raise ValueError(f"❌ Missing features: {missing}")

        # 2. Build dataframe in correct order
        df = pd.DataFrame([[feature_dict[f] for f in self.features]],
                          columns=self.features).astype(np.float32)

        # 3. Scale
        feat_scaled = self.scaler_X.transform(df)[0]

        # Startup fix: these three features are out-of-distribution in the first
        # rolling window (V_mean_lag_1=0 -> z=-57, expanding Tavg_rolling -> z=-59,
        # QD_diff=0 -> z=+4.7). Training never contained those rows, so use z=0.
        if self._n < self.startup_neutral:
            for name in ('V_mean_lag_1', 'Tavg_rolling', 'QD_diff'):
                feat_scaled[self.features.index(name)] = 0.0
        self._n += 1

        # 4. Buffer - seeded with copies of the FIRST real scaled row.
        # This is exactly what build_window.m does on the MATLAB/STM32 path
        # (repmat(scaled_row, window_len, 1) on the first call). The previous
        # version padded with rows of zeros (= the dataset mean, i.e. a
        # mid-life battery), which made the first ~30 predictions dip to ~0.87
        # even though the battery is at ~1.0.
        if not self.buffer:
            self.buffer = [feat_scaled.astype(np.float32).copy()
                           for _ in range(self.seq_len)]
        else:
            self.buffer.append(feat_scaled.astype(np.float32))
            self.buffer.pop(0)   # slide: drop oldest

        # 5. Sequence
        X_seq = np.array(self.buffer, dtype=np.float32).reshape(1, self.seq_len, self.n_features)

        # 6. Predict via TFLite Interpreter
        if self.input_is_int8:
            x_q = np.round(X_seq / self.in_scale + self.in_zp)
            x_q = np.clip(x_q, -128, 127).astype(np.int8)
            self.interpreter.set_tensor(self.input_details[0]['index'], x_q)
        else:
            self.interpreter.set_tensor(self.input_details[0]['index'], X_seq)

        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_details[0]['index'])

        if self.output_is_int8:
            pred = (float(out[0][0]) - self.out_zp) * self.out_scale + self.out_min
        else:
            pred = float(out[0][0]) + self.out_min

        # Clip to the SAME [out_min, out_max] band the model was calibrated
        # against - not the old, wider [0.0, 1.05]. A value that needs this
        # clip to fire now means out_min/out_max here disagree with what
        # optimize_v3_int8.py actually used to export the model - that's a
        # real bug to go find, not something to silently paper over with a
        # wide safety net.
        raw = float(np.clip(pred, self.out_min, self.out_max))
        self.last_raw = raw

        # Warm-up gate: a fresh window is built from repeated synthetic rows,
        # so the model is not trustworthy yet. Report nominal SOH.
        if self._n <= self.warmup_cycles:
            return self._out

        # Slow EMA (SOH moves ~3e-4 per 30 cycles, so lag is harmless) then
        # monotone clamp (battery capacity does not recover).
        self._ema = raw if self._ema is None else (1 - self.ema_alpha) * self._ema + self.ema_alpha * raw
        self._out = min(self._out, self._ema) if self.monotonic else self._ema
        return self._out


if __name__ == "__main__":
    pred = CubeSatSOHPredict()
    print("✅ Predictor READY")