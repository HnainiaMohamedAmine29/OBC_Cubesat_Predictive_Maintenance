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
ROLLING_WINDOW = 5              # cycles — window size for smoothing trends

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

# Proportion of data for training (0.7 = 70% train, 30% test)
TRAIN_RATIO = 0.70

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
SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler.pkl")

# ============================================================================
# 7. LSTM SEQUENCE CONSTRUCTION
# ============================================================================

# Sliding window size (in cycles)
# ~2 days of orbital history (1 day ≈ 15 orbits, so 30 cycles ≈ 2 days)
WINDOW_SIZE = 30                # cycles — lookback window for prediction

# Step size for sliding window (1 = generate all possible windows)
STEP_SIZE = 1

# Prediction horizon (1 = predict SOH at next cycle t+1)
HORIZON = 1

# Number of features per cycle (from feature_engineering.py)
NUM_FEATURES = 20

# ============================================================================
# 8. MODEL TRAINING (Optional, for Jupyter notebooks)
# ============================================================================

# LSTM architecture
LSTM_UNITS = 64                 # Hidden units in LSTM layer
DROPOUT_RATE = 0.2              # Dropout for regularization
BATCH_SIZE = 32                 # Training batch size
EPOCHS = 100                    # Maximum epochs
EARLY_STOPPING_PATIENCE = 10    # Stop if no improvement for N epochs
VALIDATION_SPLIT = 0.2          # Fraction of training for validation

# Learning rate
LEARNING_RATE = 0.001

# Optimizer: adam, sgd, rmsprop
OPTIMIZER = "adam"

# Loss function: mse (Mean Squared Error) for regression
LOSS = "mse"

# Metric: mae (Mean Absolute Error)
METRIC = "mae"

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

# Feature output file paths
FEATURES_PATH = os.path.join(OUTPUT_DIR, "features.csv")
TRAIN_PATH = os.path.join(OUTPUT_DIR, "train.csv")
TEST_PATH = os.path.join(OUTPUT_DIR, "test.csv")
TRAIN_NORMALIZED_PATH = os.path.join(OUTPUT_DIR, "train_normalized.csv")
TEST_NORMALIZED_PATH = os.path.join(OUTPUT_DIR, "test_normalized.csv")
TRAIN_SEQ_PATH = os.path.join(OUTPUT_DIR, "train_sequences.npy")
TEST_SEQ_PATH = os.path.join(OUTPUT_DIR, "test_sequences.npy")
TRAIN_TARGETS_PATH = os.path.join(OUTPUT_DIR, "train_targets.npy")
TEST_TARGETS_PATH = os.path.join(OUTPUT_DIR, "test_targets.npy")

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
