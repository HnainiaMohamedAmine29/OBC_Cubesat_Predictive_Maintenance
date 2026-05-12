# =============================================================================
#  build_sequences.py — Construction des séquences temporelles pour le LSTM
#  CubeSat EPS Battery SOH Estimation
#  Version SANS battery_id : une seule série temporelle globale
# =============================================================================

import os
import numpy as np
import pandas as pd
from config import (
    CYCLE_COL, TARGET_COL,
    FILE_TRAIN_NORM, FILE_TEST_NORM,
    OUTPUT_DIR
)
from feature_engineering import FEATURE_COLS

# =============================================================================
# PARAMÈTRES LSTM
# =============================================================================
WINDOW_SIZE    = 30    # Longueur de séquence (cycles passés utilisés)
STEP_SIZE      = 1     # Pas du sliding window (1 = dense, N = décimé)
HORIZON        = 1     # Prédire SOH à t+HORIZON (1 = prochain cycle)

FILE_X_TRAIN   = os.path.join(OUTPUT_DIR, "X_train.npy")
FILE_Y_TRAIN   = os.path.join(OUTPUT_DIR, "y_train.npy")
FILE_X_TEST    = os.path.join(OUTPUT_DIR, "X_test.npy")
FILE_Y_TEST    = os.path.join(OUTPUT_DIR, "y_test.npy")
FILE_SEQ_INFO  = os.path.join(OUTPUT_DIR, "sequences_info.txt")


# =============================================================================
# FONCTION DE CONSTRUCTION DES SÉQUENCES (GLOBALE, SANS GROUPE)
# =============================================================================

def build_sequences(df: pd.DataFrame, window: int, step: int, horizon: int,
                    feature_cols: list, target_col: str):
   
    X_list, y_list = [], []

    features = df[feature_cols].values   # (n_total, n_features)
    targets  = df[target_col].values      # (n_total,)

    n_total = len(df)
    max_start = n_total - window - horizon + 1

    if max_start <= 0:
        raise ValueError(f"Pas assez de données ({n_total} lignes) "
                         f"pour window={window} + horizon={horizon}.")

    for start in range(0, max_start, step):
        end    = start + window
        target_idx = start + window + horizon - 1
        X_list.append(features[start:end])
        y_list.append(targets[target_idx])

    print(f"  Total : {n_total} lignes → {len(X_list)} séquences construites")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def run():
    print("=" * 60)
    print("  BUILD SEQUENCES — LSTM CubeSat EPS Battery")
    print(f"  Window={WINDOW_SIZE} | Step={STEP_SIZE} | Horizon={HORIZON}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Sélection des features (correction du doublon cycle/cycle.1)
    # ------------------------------------------------------------------
    feature_cols = [c for c in FEATURE_COLS if c in
                    pd.read_csv(FILE_TRAIN_NORM, nrows=0).columns]

    # ------------------------------------------------------------------
    # Chargement – on conserve l'ordre des fichiers tel quel
    # (ils sont supposés être déjà ordonnés temporellement)
    # ------------------------------------------------------------------
    print(f"\n[1/4] Chargement des datasets normalisés...")
    train = pd.read_csv(FILE_TRAIN_NORM)
    test  = pd.read_csv(FILE_TEST_NORM)

    print(f"      Train : {len(train):,} lignes  |  Test : {len(test):,} lignes")
    print(f"      Features utilisées : {len(feature_cols)}")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    print(f"\n[2/4] Construction des séquences train...")
    X_train, y_train = build_sequences(
        train, WINDOW_SIZE, STEP_SIZE, HORIZON, feature_cols, TARGET_COL
    )

    print(f"\n      Construction des séquences test...")
    X_test, y_test = build_sequences(
        test, WINDOW_SIZE, STEP_SIZE, HORIZON, feature_cols, TARGET_COL
    )

    # ------------------------------------------------------------------
    # Vérifications
    # ------------------------------------------------------------------
    print(f"\n[3/4] Vérifications...")
    assert X_train.ndim == 3, "X_train doit être (N, window, features)"
    assert X_train.shape[1] == WINDOW_SIZE
    assert X_train.shape[2] == len(feature_cols)
    assert not np.any(np.isnan(X_train)), "NaN détectés dans X_train !"
    assert not np.any(np.isnan(X_test)),  "NaN détectés dans X_test !"

    print(f"      X_train : {X_train.shape}  y_train : {y_train.shape}")
    print(f"      X_test  : {X_test.shape}   y_test  : {y_test.shape}")
    print(f"      SOH train → min={y_train.min():.4f} max={y_train.max():.4f} "
          f"moy={y_train.mean():.4f}")
    print(f"      SOH test  → min={y_test.min():.4f}  max={y_test.max():.4f}  "
          f"moy={y_test.mean():.4f}")

    # ------------------------------------------------------------------
    # Sauvegarde
    # ------------------------------------------------------------------
    print(f"\n[4/4] Sauvegarde...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(FILE_X_TRAIN, X_train)
    np.save(FILE_Y_TRAIN, y_train)
    np.save(FILE_X_TEST,  X_test)
    np.save(FILE_Y_TEST,  y_test)

    # Fichier d'info
    info = (
        f"window_size   = {WINDOW_SIZE}\n"
        f"step_size     = {STEP_SIZE}\n"
        f"horizon       = {HORIZON}\n"
        f"n_features    = {len(feature_cols)}\n"
        f"features      = {feature_cols}\n"
        f"X_train_shape = {list(X_train.shape)}\n"
        f"X_test_shape  = {list(X_test.shape)}\n"
    )
    with open(FILE_SEQ_INFO, "w") as f:
        f.write(info)

    print(f"      → {FILE_X_TRAIN}  {X_train.nbytes/1e6:.1f} MB")
    print(f"      → {FILE_Y_TRAIN}")
    print(f"      → {FILE_X_TEST}   {X_test.nbytes/1e6:.1f} MB")
    print(f"      → {FILE_Y_TEST}")
    print(f"      → {FILE_SEQ_INFO}")

    print(f"\n{'=' * 60}")
    print(f"  ✅ Séquences prêtes pour le LSTM")
    print(f"  Input shape  : ({WINDOW_SIZE}, {len(feature_cols)})")
    print(f"  N train seq  : {len(X_train):,}")
    print(f"  N test seq   : {len(X_test):,}")
    print(f"{'=' * 60}\n")

    return X_train, y_train, X_test, y_test, feature_cols


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================
if __name__ == "__main__":
    X_train, y_train, X_test, y_test, feature_cols = run()