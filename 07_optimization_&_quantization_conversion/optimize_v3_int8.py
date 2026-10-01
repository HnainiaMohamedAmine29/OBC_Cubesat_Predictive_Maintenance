"""
Post-training INT8 quantization of the v3 SOH model, calibrated and
validated on normal-condition data.

Produces cubesat_soh_lstm_v3_int8.tflite (FIXED BATCH SIZE 1, for TFLite Micro on the
STM32), then validates it against the
original float32 .keras model so you can see exactly what accuracy
quantization costs before deploying.

Requires:
    - cubesat_soh_lstm_v3_unroll_True.keras   (the float32 model)
    - scaler_X.pkl                            (the actual fitted StandardScaler)
    - features_used.pkl                       (the actual feature name/order list)
    - raw_features.csv                        (normal-condition training data,
                                                 used for calibration + validation)

Usage:
    python optimize_v3_int8.py \
        --keras_model cubesat_soh_lstm_v3_unroll_True.keras \
        --scaler scaler_X.pkl \
        --features features_used.pkl \
        --raw_features raw_features.csv \
        --output cubesat_soh_lstm_v3_int8.tflite \
        --n_calib 300
"""
import argparse
import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers

WINDOW = 30
TRAIN_FRAC = 0.80  # used to carve out a validation slice for reporting


@keras.saving.register_keras_serializable(package='cubesat_soh')
class AttentionContext(layers.Layer):
    """Sum-pools the softmax-weighted timestep vectors into a context vector.
    Must be registered before load_model() can deserialize the .keras file."""
    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])


def build_seqs(df, scaler, feature_cols):
    scaled = scaler.transform(df[feature_cols]).astype(np.float32)
    y = df['SOH'].values.astype(np.float32)
    X, Y = [], []
    for start in range(0, len(df) - WINDOW):
        X.append(scaled[start:start + WINDOW])
        Y.append(y[start + WINDOW])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


# Ops registered in the firmware's resolver (App: TFLM/tflm_c_api.cpp)
FIRMWARE_OPS = {
    'RESHAPE', 'MEAN', 'NEG', 'SQUARED_DIFFERENCE', 'ADD', 'RSQRT', 'MUL',
    'FULLY_CONNECTED', 'TRANSPOSE', 'SOFTMAX', 'ELU', 'LOGISTIC', 'SPLIT',
    'TANH', 'QUANTIZE', 'DEQUANTIZE', 'PACK', 'UNPACK', 'SHAPE',
    'STRIDED_SLICE', 'FILL', 'SUM', 'GATHER', 'CONCATENATION',
}
# Ops that only show up when the batch dimension was dynamic. TFLite Micro
# cannot run them (FILL needs constant dims, REDUCE_PROD has no kernel).
DYNAMIC_SHAPE_OPS = {'SHAPE', 'GATHER', 'REDUCE_PROD', 'FILL', 'STRIDED_SLICE'}


def audit_ops(tflite_path):
    """Print which ops the converted model uses and flag anything the STM32
    firmware cannot run."""
    try:
        interp = tf.lite.Interpreter(model_path=tflite_path)
        names = [d['op_name'] for d in interp._get_ops_details()]
    except Exception as e:  # private API - don't fail the export over it
        print(f'(op audit skipped: {e})')
        return
    used = sorted(set(names))
    print(f'ops used ({len(names)} nodes): {used}')
    unsupported = [o for o in used if o not in FIRMWARE_OPS]
    dynamic = [o for o in used if o in DYNAMIC_SHAPE_OPS]
    if unsupported:
        print(f'  !! NOT registered in the firmware: {unsupported}')
    if dynamic:
        print(f'  !! dynamic-shape ops present: {dynamic} - TFLite Micro will fail '
              f'to prepare these. The model was not exported with a fixed batch size.')
    if not unsupported and not dynamic:
        print('  OK: every op is registered in the firmware and no dynamic-shape ops remain.')


def load_raw_features(path, feature_cols):
    """Read raw_features.csv defensively: only the columns we need, and fall
    back to more forgiving parsers if the file has stray quote characters or
    malformed lines (a stray '"' makes pandas treat the rest of the file as one
    huge field, which shows up as 'C error: out of memory')."""
    import csv
    print(f'reading {path} ({os.path.getsize(path) / 1e6:.1f} MB)')
    needed = set(feature_cols) | {'cycle', 'SOH'}
    attempts = [
        ('default parser', dict()),
        ('quotes disabled', dict(quoting=csv.QUOTE_NONE)),
        ('python parser, bad lines skipped',
         dict(engine='python', quoting=csv.QUOTE_NONE, on_bad_lines='skip')),
    ]
    last_err = None
    for name, extra in attempts:
        try:
            df = pd.read_csv(path, usecols=lambda c: c.strip() in needed, **extra)
            df.columns = [c.strip() for c in df.columns]
            missing = needed - set(df.columns)
            if missing:
                raise ValueError(f'columns missing from {path}: {sorted(missing)}')
            print(f'  loaded with {name}: {df.shape[0]} rows x {df.shape[1]} cols')
            return df
        except (pd.errors.ParserError, MemoryError) as e:
            print(f'  {name} failed: {str(e)[:120]}')
            last_err = e
    raise last_err


def quantize(args):
    # --- Use the ACTUAL scaler + feature order from training, not a reconstruction ---
    scaler_X = joblib.load(args.scaler)
    feature_cols = joblib.load(args.features)
    print(f'Loaded real scaler_X.pkl and features_used.pkl ({len(feature_cols)} features)')

    raw = load_raw_features(args.raw_features, feature_cols).sort_values('cycle').reset_index(drop=True)
    n_train = int(len(raw) * TRAIN_FRAC)
    val_df = raw.iloc[n_train:].reset_index(drop=True)

    X_normal_full, _ = build_seqs(raw, scaler_X, feature_cols)
    print(f'normal pool: {X_normal_full.shape}')

    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_normal_full), args.n_calib, replace=False)
    calib_seqs = X_normal_full[idx]
    print(f'calibration set: {calib_seqs.shape}')

    def representative_dataset():
        for i in range(calib_seqs.shape[0]):
            yield [calib_seqs[i:i + 1]]

    # --- Load the float32 model ---
    model = tf.keras.models.load_model(
        args.keras_model, compile=False,
        custom_objects={'AttentionContext': AttentionContext}
    )
    fp32_size = os.path.getsize(args.keras_model) / 1024
    print(f'Original float32 .keras size: {fp32_size:.1f} KB')

    # --- Full INT8 conversion: int8 weights, activations, AND input/output ---
    # --- Fix the batch size to 1 ---
    # Converting the Keras model as-is leaves the batch dimension dynamic
    # (shape [None, 30, 20]). The converter then emits SHAPE / GATHER /
    # REDUCE_PROD / PACK / FILL / STRIDED_SLICE nodes to compute the LSTM
    # initial state and the attention Tensordot sizes at runtime. TFLite Micro
    # plans all memory at init, so it rejects these (REDUCE_PROD has no kernel,
    # FILL requires constant dims). Rebuilding the model on a fixed
    # batch_shape=(1, WINDOW, n_features) input lets the converter fold them all
    # away. The weights are shared - this does not change the model.
    n_features = len(feature_cols)
    fixed_in = keras.Input(batch_shape=(1, WINDOW, n_features))
    model_static = keras.Model(fixed_in, model(fixed_in))

    converter = tf.lite.TFLiteConverter.from_keras_model(model_static)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    with open(args.output, 'wb') as f:
        f.write(tflite_model)
    int8_size = len(tflite_model) / 1024
    print(f'Quantized int8 .tflite size: {int8_size:.1f} KB '
          f'({fp32_size / int8_size:.2f}x smaller)')
    audit_ops(args.output)

    return model, scaler_X, feature_cols, val_df


def validate(args, model, scaler_X, feature_cols, val_df):
    X_val, y_val = build_seqs(val_df, scaler_X, feature_cols)

    pred_fp32 = model.predict(X_val, verbose=0).flatten()

    interp = tf.lite.Interpreter(model_path=args.output)
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    in_scale, in_zp = in_d['quantization']
    out_scale, out_zp = out_d['quantization']

    def run_int8(X):
        preds = []
        for i in range(X.shape[0]):
            x_q = np.round(X[i:i + 1] / in_scale + in_zp).astype(np.int8)
            interp.set_tensor(in_d['index'], x_q)
            interp.invoke()
            out_q = interp.get_tensor(out_d['index'])
            preds.append((out_q.astype(np.float32) - out_zp) * out_scale)
        return np.array(preds, dtype=np.float32).flatten()

    pred_int8 = run_int8(X_val)

    print('--- normal validation ---')
    print(f'  MAE fp32 vs true: {np.mean(np.abs(pred_fp32 - y_val)):.5f}')
    print(f'  MAE int8 vs true: {np.mean(np.abs(pred_int8 - y_val)):.5f}')
    print(f'  quantization-only drift (int8 vs fp32): '
          f'{np.mean(np.abs(pred_int8 - pred_fp32)):.5f}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--keras_model', required=True)
    p.add_argument('--scaler', required=True, help='path to scaler_X.pkl')
    p.add_argument('--features', required=True, help='path to features_used.pkl')
    p.add_argument('--raw_features', required=True)
    p.add_argument('--output', default='cubesat_soh_lstm_v3_int8.tflite')
    p.add_argument('--n_calib', type=int, default=300)
    args = p.parse_args()

    model, scaler_X, feature_cols, val_df = quantize(args)
    validate(args, model, scaler_X, feature_cols, val_df)