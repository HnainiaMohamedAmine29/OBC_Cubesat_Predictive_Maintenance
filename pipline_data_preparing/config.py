# Python Configuration File for OBC CubeSat Battery SOH Prediction Pipeline
# ============================================================================
# Adjust these settings for your specific dataset and mission parameters
# ============================================================================

import os

# ============================================================================
# 1. DATA PATHS
# ============================================================================

# Input: Raw battery dataset from MATLAB simulation (battery_model_Normal_conditions)
DATA_PATH = "battery_dataset_normal.csv"

# Output directory for processed data
OUTPUT_DIR = "processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# 2. DATASET COLUMN NAMES
# ============================================================================

ID_COL = "battery_id"           # Battery identifier
CYCLE_COL = "cycle"             # Orbital cycle number (time ordering)
TARGET_COL = "SOH"              # State of Health — PREDICTION TARGET

# ============================================================================
# 3. PHYSICAL PARAMETERS (from battery_model_Normal_conditions)
# ============================================================================

# Battery pack parameters (from cubesat_params.m)
Q_NOM = 4.035                   # Ah — nominal capacity at BOL
NOMINAL_VOLTAGE_V = 7.40        # V — midpoint operating voltage (2S pack)
COULOMBIC_EFF = 0.98            # — charging efficiency

# Thermal thresholds
COLD_THRESHOLD_C = -10          # °C — eclipse minimum temperature
ECLIPSE_THRESHOLD_K = 263.15    # K — temperature threshold for eclipse detection

# Temperature for Arrhenius model
T_REF_K = 298.15                # K — reference (25°C)
T_REF_C = 25.0                  # °C

# ============================================================================
# 4. FEATURE ENGINEERING PARAMETERS
# ============================================================================

# Rolling window for moving average calculations
# Aligned with prediction_soh_lstm_target(_SOH).ipynb (ROLLING_W = 10)
ROLLING_WINDOW = 10             # cycles — window size for smoothing trends

# Threshold used by cold_cycle_count in feature_engineering.py.
# The notebook flags a cycle as "cold" when Tavg_C < 5°C (not Tmin_C < COLD_THRESHOLD_C).
COLD_CYCLE_THRESHOLD_C = 5

# Aging rate baseline (from cubesat_params.m)
ALPHA = 0.02085                 # — base aging rate coefficient
K_R = 1.899                     # — resistance growth factor with capacity fade
EA = 32000                      # J/mol — activation energy
R_GAS = 8.314                   # J/(mol·K) — gas constant

# ============================================================================
# 5. TRAIN / TEST SPLITTING
# ============================================================================

# Split method: "temporal" (recommended for time-series) or "random"
# Temporal: Earlier cycles → training, Later cycles → testing
SPLIT_METHOD = "temporal"

# Proportion of data for training
# Aligned with prediction_soh_lstm_target(_SOH).ipynb (TRAIN_FRAC = 0.80)
TRAIN_RATIO = 0.80

# Random state for reproducibility (if using random split)
RANDOM_STATE = 42

# ============================================================================
# 6. NORMALIZATION
# ============================================================================

# Scaler type: "standard" (zero mean, unit variance) or "minmax" (0-1 range)
SCALER_TYPE = "standard"        # Recommended: "standard" for LSTM

# Columns to exclude from normalization (identifiers, cycle count, flags)
COLS_NO_SCALE = ["battery_id", "cycle", "eclipse_flag"]

# Save scaler for later inference (embedded deployment)
# (actual path is defined once, below, as FILE_SCALER / SCALER_PATH alias)

# ============================================================================
# 7. LSTM SEQUENCE CONSTRUCTION
# ============================================================================

# Sliding window size (in cycles)
# ~2 days of orbital history (1 day ≈ 15 orbits, so 30 cycles ≈ 2 days)
# These three values are the single source of truth for the whole pipeline —
# sequences_construction.py and model.py both import them from here instead
# of hardcoding their own copies (which is what caused the previous drift
# vs. prediction_soh_lstm_target(_SOH).ipynb).
WINDOW_SIZE = 30                # cycles — lookback window for prediction

# Step size for sliding window (1 = generate all possible windows)
# Aligned with the notebook (STEP_SIZE = 3): denser windows overlap too much
# and slow training down without adding much new information.
STEP_SIZE = 3

# Prediction horizon (1 = predict SOH at next cycle t+1)
HORIZON = 1

# Number of features per cycle (from feature_engineering.py)
NUM_FEATURES = 20

# ============================================================================
# 8. MODEL ARCHITECTURE & TRAINING
#    Mirrors build_model_v2() in prediction_soh_lstm_target(_SOH).ipynb
#    (LSTM 128 -> LSTM 64 -> Self-Attention -> Dense 64 -> Dense 1)
# ============================================================================

# LSTM stack
LSTM_UNITS_1 = 128               # First LSTM layer (return_sequences=True)
LSTM_UNITS_2 = 64                # Second LSTM layer (return_sequences=True)
LSTM_DROPOUT = 0.2               # Dropout applied after each LSTM block

# Self-attention block
ATTENTION_HEADS = 4
ATTENTION_KEY_DIM = 16           # 4 heads * 16 = 64, matches LSTM_UNITS_2
ATTENTION_DROPOUT = 0.1

# Dense head
DENSE_UNITS = 64
DENSE_ACTIVATION = "elu"
DENSE_DROPOUT = 0.1

# Output activation.
# NOTE: the notebook's docstring argues for a *linear* output (sigmoid's
# gradient near SOH=1.0 is too flat and causes an upward bias), but the
# actual notebook code still builds the layer with activation="sigmoid".
# We keep "sigmoid" here to match the code that was actually run and
# evaluated; switch to "linear" if you want to test the documented fix.
OUTPUT_ACTIVATION = "sigmoid"

# Regularization / optimizer
L2_REG = 2e-4                    # L2 penalty on LSTM/Dense kernels + AdamW weight decay
LEARNING_RATE = 3e-4
CLIP_NORM = 1.0                  # Gradient norm clipping (AdamW)
HUBER_DELTA = 0.01               # Huber loss crossover, ~1% SOH error

# Training loop
BATCH_SIZE = 128                 # Used by model.fit (matches notebook's model.fit call)
SEQ_BUILD_BATCH_SIZE = 64        # Unused batch constant kept for parity with notebook (BATCH_SIZE var before fit override)
EPOCHS = 400                     # Maximum epochs (MAX_EPOCHS in the notebook)
EARLY_STOPPING_PATIENCE = 25
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.3
REDUCE_LR_MIN = 1e-6
RANDOM_SEED = 42

# Optimizer: adamw (per notebook). Kept for reference / documentation.
OPTIMIZER = "adamw"

# Loss function: Huber (robust to noisy cycles), metric: MAE
LOSS = "huber"
METRIC = "mae"

# Where model.py saves its artifacts
MODEL_PATH = os.path.join(OUTPUT_DIR, "cubesat_soh_lstm.keras")
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics.csv")
HISTORY_PATH = os.path.join(OUTPUT_DIR, "training_history.csv")

# ============================================================================
# 9. VALIDATION TARGETS (Reference from battery_model_Normal_conditions)
# ============================================================================

# Expected SOH values (from simulation)
SOH_INITIAL = 0.99              # 99% — beginning of life
SOH_EOL = 0.70                  # 70% — end of life threshold (IEC 62660-1 standard)

# Expected cycle life
CYCLES_TO_70 = 7000             # cycles — normal operation until EOL

# Expected resistance
IR_INITIAL_mOhm = 15            # mΩ — initial resistance at 25°C
IR_AT_EOL_mOhm = 100            # mΩ — approx resistance at 70% SOH

# Performance targets
RMSE_TARGET = 0.02              # 2% absolute error on SOH prediction
MAE_TARGET = 0.015              # 1.5% absolute error
R2_TARGET = 0.95                # Coefficient of determination

# ============================================================================
# 10. LOGGING & OUTPUT
# ============================================================================

# Verbosity level: "debug", "info", "warning"
LOG_LEVEL = "info"

# Save intermediate results
SAVE_FEATURES = True            # Save features.csv after feature engineering
SAVE_TRAIN_TEST = True          # Save train.csv and test.csv
SAVE_SEQUENCES = True           # Save sequences in .npy format

# Feature / split / normalization output paths
# NOTE: these are the names actually imported downstream
# (feature_engineering.py, split.py, normalize.py, sequences_construction.py).
FILE_FEATURES = os.path.join(OUTPUT_DIR, "features.csv")
FILE_TRAIN = os.path.join(OUTPUT_DIR, "train.csv")
FILE_TEST = os.path.join(OUTPUT_DIR, "test.csv")
FILE_TRAIN_NORM = os.path.join(OUTPUT_DIR, "train_normalized.csv")
FILE_TEST_NORM = os.path.join(OUTPUT_DIR, "test_normalized.csv")

# Scaler artifact (fit on train features only — matches scaler_X in the notebook)
FILE_SCALER = os.path.join(OUTPUT_DIR, "scaler_X.pkl")
FILE_SCALER_COLS = os.path.join(OUTPUT_DIR, "scaler_columns.txt")

# LSTM sequence arrays (produced by sequences_construction.py, consumed by model.py)
FILE_X_TRAIN = os.path.join(OUTPUT_DIR, "X_train.npy")
FILE_Y_TRAIN = os.path.join(OUTPUT_DIR, "y_train.npy")
FILE_X_TEST = os.path.join(OUTPUT_DIR, "X_test.npy")
FILE_Y_TEST = os.path.join(OUTPUT_DIR, "y_test.npy")
FILE_SEQ_INFO = os.path.join(OUTPUT_DIR, "sequences_info.txt")

# Keep the old, more descriptive names as aliases for anything that still
# references them (e.g. custom notebooks or notes).
FEATURES_PATH = FILE_FEATURES
TRAIN_PATH = FILE_TRAIN
TEST_PATH = FILE_TEST
TRAIN_NORMALIZED_PATH = FILE_TRAIN_NORM
TEST_NORMALIZED_PATH = FILE_TEST_NORM
SCALER_PATH = FILE_SCALER

# =============================================================================
# VERIFICATION (Auto-check on import)
# =============================================================================

def verify_config():
    """Verify configuration integrity"""
    assert SPLIT_METHOD in ["temporal", "random"], "SPLIT_METHOD must be 'temporal' or 'random'"
    assert SCALER_TYPE in ["standard", "minmax"], "SCALER_TYPE must be 'standard' or 'minmax'"
    assert 0 < TRAIN_RATIO < 1, "TRAIN_RATIO must be between 0 and 1"
    assert WINDOW_SIZE > 0, "WINDOW_SIZE must be positive"
    assert HORIZON > 0, "HORIZON must be positive"
    print("✓ Configuration validated successfully")

if __name__ == "__main__":
    verify_config()
    print(f"✓ Config loaded: {DATA_PATH}")
    print(f"  - Split method: {SPLIT_METHOD} ({TRAIN_RATIO*100:.0f}% train)")
    print(f"  - Window size: {WINDOW_SIZE} cycles ({WINDOW_SIZE//15:.1f} days)")
    print(f"  - Output directory: {OUTPUT_DIR}/")
