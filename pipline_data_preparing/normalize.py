# =============================================================================
#  normalize.py — Normalisation du dataset CubeSat EPS Battery
#  Ce script ne change pas. Modifiez config.py pour adapter les paramètres.
#
#  Règle fondamentale : le scaler est fitté UNIQUEMENT sur le train
#  puis appliqué au test → évite toute fuite d'information future.
#
#  Stratégie par type de feature :
#    - Features continues (tension, temp, capacité...) → StandardScaler ou MinMaxScaler
#    - Features binaires / ordinales (eclipse_flag, cycle) → PAS de normalisation
#    - Colonnes d'identifiant (battery_id, cycle_raw) → PAS de normalisation
# =============================================================================

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from config import (
    ID_COL, CYCLE_COL, TARGET_COL,
    SCALER_TYPE, COLS_NO_SCALE,
    FILE_TRAIN, FILE_TEST,
    FILE_TRAIN_NORM, FILE_TEST_NORM,
    FILE_SCALER, FILE_SCALER_COLS,
    OUTPUT_DIR
)
from feature_engineering import FEATURE_COLS


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def get_scaler(scaler_type: str):
    """Instancie le scaler choisi dans config.py."""
    scalers = {
        "standard": StandardScaler(),
        "minmax":   MinMaxScaler(feature_range=(0, 1)),
    }
    if scaler_type not in scalers:
        raise ValueError(f"SCALER_TYPE inconnu : '{scaler_type}'. "
                         f"Valeurs acceptées : {list(scalers.keys())}")
    return scalers[scaler_type]


def get_cols_to_scale(feature_cols: list, no_scale_cols: list) -> list:
    """
    Retourne la liste des colonnes à normaliser :
    toutes les features sauf celles listées dans COLS_NO_SCALE.
    """
    return [c for c in feature_cols if c not in no_scale_cols]


def log_scaler_stats(scaler, cols_scaled: list):
    """Affiche les paramètres du scaler pour inspection."""
    if isinstance(scaler, StandardScaler):
        stats = pd.DataFrame({
            "feature": cols_scaled,
            "mean":    scaler.mean_.round(6),
            "std":     np.sqrt(scaler.var_).round(6),
        })
        print("\n      Paramètres StandardScaler (fitté sur train) :")
    else:
        stats = pd.DataFrame({
            "feature": cols_scaled,
            "min":     scaler.data_min_.round(6),
            "max":     scaler.data_max_.round(6),
        })
        print("\n      Paramètres MinMaxScaler (fitté sur train) :")

    print(stats.to_string(index=False))


# =============================================================================
# NORMALISATION PRINCIPALE
# =============================================================================

def run(train_path: str = FILE_TRAIN, test_path: str = FILE_TEST):
    print("=" * 60)
    print("  NORMALISATION — CubeSat EPS Battery")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Chargement
    # ------------------------------------------------------------------
    print(f"\n[1/5] Chargement des données splittées...")
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    print(f"      Train : {len(train):,} lignes  |  Test : {len(test):,} lignes")

    # ------------------------------------------------------------------
    # Identification des colonnes à normaliser
    # ------------------------------------------------------------------
    cols_scaled    = get_cols_to_scale(FEATURE_COLS, COLS_NO_SCALE)
    cols_not_scaled = [c for c in FEATURE_COLS if c not in cols_scaled]

    print(f"\n[2/5] Colonnes à normaliser ({len(cols_scaled)}) :")
    print(f"      {cols_scaled}")
    print(f"\n      Colonnes NON normalisées ({len(cols_not_scaled)}) :")
    print(f"      {cols_not_scaled}  ← binaires / ordinales")

    # ------------------------------------------------------------------
    # Fit du scaler sur le TRAIN uniquement
    # ------------------------------------------------------------------
    print(f"\n[3/5] Fit du scaler ({SCALER_TYPE.upper()}) sur X_train uniquement...")
    scaler = get_scaler(SCALER_TYPE)
    scaler.fit(train[cols_scaled])
    log_scaler_stats(scaler, cols_scaled)

    # ------------------------------------------------------------------
    # Transformation train + test
    # ------------------------------------------------------------------
    print(f"\n[4/5] Transformation train et test...")

    def transform_dataset(df: pd.DataFrame, name: str) -> pd.DataFrame:
        df_norm = df.copy()
        df_norm[cols_scaled] = scaler.transform(df[cols_scaled])
        return df_norm

    train_norm = transform_dataset(train, "Train")
    test_norm  = transform_dataset(test,  "Test")

    # Vérification : mean et std du train normalisé doivent être ≈ 0 et 1
    if SCALER_TYPE == "standard":
        train_means = train_norm[cols_scaled].mean().abs().max()
        train_stds  = train_norm[cols_scaled].std().max()
        print(f"      ✓ Vérification train normalisé : "
              f"max|μ| = {train_means:.2e}  max(σ) = {train_stds:.4f}  "
              f"(attendu ≈ 0 et ≈ 1)")

    # Les colonnes non normalisées et la cible restent inchangées
    print(f"      ✓ Colonnes non normalisées préservées : {cols_not_scaled}")
    print(f"      ✓ Cible '{TARGET_COL}' non normalisée (valeurs brutes conservées)")

    # ------------------------------------------------------------------
    # Sauvegarde
    # ------------------------------------------------------------------
    print(f"\n[5/5] Sauvegarde...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_norm.to_csv(FILE_TRAIN_NORM, index=False)
    test_norm.to_csv(FILE_TEST_NORM,  index=False)

    # Sauvegarde du scaler pour réutilisation (inférence en vol)
    joblib.dump(scaler, FILE_SCALER)

    # Sauvegarde de la liste des colonnes normalisées (reproductibilité)
    with open(FILE_SCALER_COLS, "w") as f:
        f.write("\n".join(cols_scaled))

    print(f"      → {FILE_TRAIN_NORM}")
    print(f"      → {FILE_TEST_NORM}")
    print(f"      → {FILE_SCALER}         ← scaler sauvegardé (joblib)")
    print(f"      → {FILE_SCALER_COLS}  ← liste colonnes normalisées")

    print(f"\n{'=' * 60}")
    print(f"  ✅ Normalisation terminée")
    print(f"  Features normalisées  : {len(cols_scaled)}")
    print(f"  Features brutes/binary: {len(cols_not_scaled)}")
    print(f"  Cible '{TARGET_COL}'  : non normalisée")
    print(f"{'=' * 60}\n")

    return train_norm, test_norm, scaler, cols_scaled


# =============================================================================
# UTILITAIRE : recharger le scaler sauvegardé
# =============================================================================

def load_scaler():
    """
    Charge le scaler sauvegardé pour normaliser de nouvelles données
    (utile pour l'inférence embarquée ou un nouveau dataset de même mission).
    """
    scaler = joblib.load(FILE_SCALER)
    with open(FILE_SCALER_COLS, "r") as f:
        cols = f.read().splitlines()
    print(f"Scaler chargé depuis {FILE_SCALER} ({len(cols)} colonnes)")
    return scaler, cols


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================
if __name__ == "__main__":
    train_norm, test_norm, scaler, cols_scaled = run()

    print("Aperçu train normalisé (3 premières lignes) :")
    print(train_norm[cols_scaled[:5]].head(3).round(4).to_string())

    print("\nAperçu test normalisé (3 premières lignes) :")
    print(test_norm[cols_scaled[:5]].head(3).round(4).to_string())
