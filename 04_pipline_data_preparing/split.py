# =============================================================================
#  split.py — Découpage Train / Test du dataset CubeSat EPS
#  Adapté pour un dataset sans colonne battery_id.
#  Split temporel global : les cycles les plus anciens → train, les plus récents → test.
# =============================================================================

import os
import pandas as pd
from config import (
    CYCLE_COL, TARGET_COL,
    SPLIT_METHOD, TRAIN_RATIO, RANDOM_STATE,
    FILE_FEATURES, FILE_TRAIN, FILE_TEST, OUTPUT_DIR
)
from feature_engineering import FEATURE_COLS


# =============================================================================
# FONCTIONS DE SPLIT
# =============================================================================

def temporal_split(df: pd.DataFrame, train_ratio: float):
    """
    Split temporel global : les premiers train_ratio% des cycles → train, le reste → test.
    Le DataFrame est trié par CYCLE_COL croissant.
    """
    df_sorted = df.sort_values(CYCLE_COL).reset_index(drop=True)
    n_total = len(df_sorted)
    n_train = int(n_total * train_ratio)

    train = df_sorted.iloc[:n_train]
    test  = df_sorted.iloc[n_train:]

    print(f"      Split temporel global : train = {n_train} lignes (cycles ≤ {train[CYCLE_COL].max()}) "
          f"| test = {n_total - n_train} lignes (cycles ≥ {test[CYCLE_COL].min()})")

    return train, test


def random_split(df: pd.DataFrame, train_ratio: float, random_state: int):
    """
    Split aléatoire (usage comparatif uniquement — risque de data leakage).
    """
    train = df.sample(frac=train_ratio, random_state=random_state)
    test  = df.drop(train.index)
    return train.sort_index(), test.sort_index()


# =============================================================================
# SPLIT PRINCIPAL
# =============================================================================

def run(features_path: str = FILE_FEATURES):
    print("=" * 60)
    print("  SPLIT TRAIN / TEST — CubeSat EPS Battery")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Chargement
    # ------------------------------------------------------------------
    print(f"\n[1/3] Chargement des features : {features_path}")
    df = pd.read_csv(features_path)
    print(f"      {len(df):,} lignes  |  {df.shape[1]} colonnes")

    # ------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------
    print(f"\n[2/3] Méthode : {SPLIT_METHOD.upper()}  |  Ratio train : {TRAIN_RATIO:.0%}")

    if SPLIT_METHOD == "temporal":
        train, test = temporal_split(df, TRAIN_RATIO)
    elif SPLIT_METHOD == "random":
        print(f"      ⚠️  Random split : risque de data leakage sur séries temporelles")
        train, test = random_split(df, TRAIN_RATIO, RANDOM_STATE)
    else:
        raise ValueError(f"SPLIT_METHOD inconnu : '{SPLIT_METHOD}'. "
                         f"Valeurs acceptées : 'temporal', 'random'")

    train = train.reset_index(drop=True)
    test  = test.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Résumé des distributions SOH
    # ------------------------------------------------------------------
    print(f"\n      Répartition :")
    print(f"      Train : {len(train):>6,} lignes  |  "
          f"SOH min={train[TARGET_COL].min():.4f}  max={train[TARGET_COL].max():.4f}  "
          f"moy={train[TARGET_COL].mean():.4f}")
    print(f"      Test  : {len(test):>6,} lignes  |  "
          f"SOH min={test[TARGET_COL].min():.4f}  max={test[TARGET_COL].max():.4f}  "
          f"moy={test[TARGET_COL].mean():.4f}")

    # Vérification simple de non-chevauchement temporel
    if SPLIT_METHOD == "temporal" and not train.empty and not test.empty:
        max_train_cycle = train[CYCLE_COL].max()
        min_test_cycle  = test[CYCLE_COL].min()
        if min_test_cycle <= max_train_cycle:
            print(f"      ⚠️  Chevauchement de cycles détecté ! (max train={max_train_cycle}, min test={min_test_cycle})")
        else:
            print(f"      ✓  Pas de chevauchement — split propre (max train={max_train_cycle} < min test={min_test_cycle})")

    # ------------------------------------------------------------------
    # Sauvegarde
    # ------------------------------------------------------------------
    print(f"\n[3/3] Sauvegarde...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train.to_csv(FILE_TRAIN, index=False)
    test.to_csv(FILE_TEST,  index=False)
    print(f"      → {FILE_TRAIN}")
    print(f"      → {FILE_TEST}")

    print(f"\n{'=' * 60}")
    print(f"  ✅ Split terminé")
    print(f"  Train : {len(train):,} lignes ({len(train)/len(df):.1%})")
    print(f"  Test  : {len(test):,} lignes ({len(test)/len(df):.1%})")
    print(f"{'=' * 60}\n")

    return train, test


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================
if __name__ == "__main__":
    train, test = run()
    print(f"Colonnes disponibles : {list(train.columns)}")