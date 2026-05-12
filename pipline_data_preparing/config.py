# =============================================================================
#  config.py — Configuration centrale du pipeline CubeSat EPS
#  Modifiez CE fichier uniquement. Les scripts feature_engineering.py,
#  split.py et normalize.py restent inchangés quelle que soit la mission.
# =============================================================================

import os

# -----------------------------------------------------------------------------
# 1. DATASET
# -----------------------------------------------------------------------------
DATA_PATH   = "processed/features.csv"   # chemin vers le CSV brut
ID_COL      = "battery_id"                   # colonne identifiant batterie
CYCLE_COL   = "cycle"                        # colonne numéro de cycle
TARGET_COL  = "SOH"                          # variable cible à prédire

# -----------------------------------------------------------------------------
# 2. PARAMÈTRES FEATURE ENGINEERING
# -----------------------------------------------------------------------------

# Fenêtre rolling (cycles)
ROLLING_WINDOW = 10          # rolling mean/std sur tension, température, QD

# Seuils physiques
COLD_THRESHOLD_C    = 0.0    # en dessous → cycle froid (lithium plating LEO)
ECLIPSE_THRESHOLD_K = 270.0  # T_amb < seuil → phase éclipse orbitale

# Tension nominale de cellule (V) — pour détecter les déviations
NOMINAL_VOLTAGE_V = 7.4      # 2 cellules en série NCR1865B (2 × 3.7 V)

# Capacité nominale initiale (Ah) — lue automatiquement depuis le cycle 1
# Laisser None pour détection automatique
NOMINAL_CAPACITY_AH = None

# -----------------------------------------------------------------------------
# 3. SPLIT TRAIN / TEST
# -----------------------------------------------------------------------------

# Méthode de split : "temporal" (recommandé pour séries temporelles)
#                    "random"   (pour tests comparatifs uniquement)
SPLIT_METHOD = "temporal"

# Proportion d'entraînement (ex: 0.8 = 80 % des cycles les plus anciens)
TRAIN_RATIO  = 0.8

# Graine aléatoire (utilisée seulement si SPLIT_METHOD = "random")
RANDOM_STATE = 42

# -----------------------------------------------------------------------------
# 4. NORMALISATION
# -----------------------------------------------------------------------------

# Type de scaler : "standard"  → StandardScaler  (μ=0, σ=1)
#                  "minmax"    → MinMaxScaler     (plage [0, 1])
SCALER_TYPE = "standard"

# Colonnes binaires / ordinales → PAS de normalisation
COLS_NO_SCALE = [
    "eclipse_flag",   # binaire 0/1
    "cycle",          # indice ordinal croissant
]

# -----------------------------------------------------------------------------
# 5. SORTIES
# -----------------------------------------------------------------------------

OUTPUT_DIR = "processed"     # dossier de sortie (créé automatiquement)

# Noms des fichiers produits (modifiables si multi-missions)
FILE_FEATURES   = os.path.join(OUTPUT_DIR, "features.csv")
FILE_TRAIN      = os.path.join(OUTPUT_DIR, "train.csv")
FILE_TEST       = os.path.join(OUTPUT_DIR, "test.csv")
FILE_TRAIN_NORM = os.path.join(OUTPUT_DIR, "train_normalized.csv")
FILE_TEST_NORM  = os.path.join(OUTPUT_DIR, "test_normalized.csv")
FILE_SCALER     = os.path.join(OUTPUT_DIR, "scaler.joblib")
FILE_SCALER_COLS= os.path.join(OUTPUT_DIR, "scaled_columns.txt")
