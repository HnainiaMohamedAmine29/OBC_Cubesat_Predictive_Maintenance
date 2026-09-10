"""
Build representative_data.npz for the full_int8 TFLite conversion,
from the same raw_features.csv used to train the SOH model.

IMPORTANT ASSUMPTION: this uses the CSV columns exactly as-is (no extra
scaling/normalization), because the Keras model appears to consume the
engineered features directly. If your training pipeline applied any
additional scaler/normalizer (e.g. a sklearn StandardScaler saved as a
.pkl, or a tf.keras.layers.Normalization layer with adapted stats) BEFORE
feeding windows into the model, you must apply that exact same transform
here too, or calibration will compute the wrong int8 scale/zero-point for
every activation in the network and accuracy will suffer. If you don't
recall such a step, this script is likely already correct.
"""
import numpy as np
import pandas as pd

SEQ_LEN = 30          # SOH_WINDOW_LEN in firmware
N_FEATURES = 20        # SOH_N_FEATURES in firmware
LABEL_COL = "SOH"
STRIDE = 15            # overlap between windows; ~450 windows from 6807 rows
MAX_SAMPLES = 200       # matches rep_dataset() cap in the converter script

df = pd.read_csv("raw_features.csv")
assert LABEL_COL in df.columns, f"{LABEL_COL} column not found"
feature_cols = [c for c in df.columns if c != LABEL_COL]
assert len(feature_cols) == N_FEATURES, (
    f"Expected {N_FEATURES} feature columns, got {len(feature_cols)}: {feature_cols}"
)

data = df[feature_cols].to_numpy(dtype="float32")
n_rows = data.shape[0]

windows = []
for start in range(0, n_rows - SEQ_LEN + 1, STRIDE):
    windows.append(data[start:start + SEQ_LEN])

windows = np.stack(windows, axis=0)  # (N, 30, 20)
print(f"Built {windows.shape[0]} windows of shape {windows.shape[1:]} "
      f"from {n_rows} rows (stride={STRIDE})")

# Subsample evenly across the run (covers early/mid/late-life behavior)
# rather than just taking the first MAX_SAMPLES, which would all come
# from the start of cycle life.
if len(windows) > MAX_SAMPLES:
    idx = np.linspace(0, len(windows) - 1, MAX_SAMPLES).astype(int)
    windows = windows[idx]
    print(f"Subsampled evenly down to {len(windows)} windows")

np.savez("representative_data.npz", X=windows)
print(f"Saved representative_data.npz  shape={windows.shape}  dtype={windows.dtype}")
print("Feature column order (must match model's training-time column order):")
for i, c in enumerate(feature_cols):
    print(f"  [{i:2d}] {c}")
