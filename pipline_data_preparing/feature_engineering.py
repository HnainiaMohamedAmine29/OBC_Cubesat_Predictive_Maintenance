# =============================================================================
#  feature_engineering.py — Construction des 20 features capteurs CubeSat EPS
#  Ce script ne change pas. Modifiez config.py pour adapter les paramètres.
# =============================================================================

import os
import pandas as pd
import numpy as np
from config import (
    DATA_PATH, ID_COL, CYCLE_COL, TARGET_COL,
    ROLLING_WINDOW, COLD_THRESHOLD_C, ECLIPSE_THRESHOLD_K,
    OUTPUT_DIR, FILE_FEATURES
)

# =============================================================================
# FONCTIONS PAR GROUPE DE CAPTEUR
# =============================================================================

def build_voltage_features(df: pd.DataFrame, grp) -> pd.DataFrame:
    """
    Capteur tension (ADC EPS) → 5 features
    Sources : V_mean_V, V_min_V, V_max_V
    """
    # 1. Tension moyenne brute (capteur direct)
    df["V_mean_V"] = df["V_mean_V"]

    # 2. Fenêtre de tension utilisable par cycle
    df["V_spread"] = df["V_max_V"] - df["V_min_V"]

    # 3. Tendance lissée de la tension sur ROLLING_WINDOW cycles
    df["V_mean_rolling"] = grp["V_mean_V"].transform(
        lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).mean()
    )

    # 4. Tension moyenne du cycle précédent (auto-régressif)
    df["V_mean_lag_1"] = grp["V_mean_V"].transform(lambda x: x.shift(1))

    # 5. Glissement de la tension minimale de fin de décharge
    #    (différence par rapport au cycle 1 de chaque batterie)
    df["V_min_fade"] = grp["V_min_V"].transform(lambda x: x - x.iloc[0])

    return df


def build_temperature_features(df: pd.DataFrame, grp) -> pd.DataFrame:
    """
    Thermistance NTC batterie + capteur structure → 6 features
    Sources : Tavg_C, Tmin_C, Tmax_C, T_amb_K
    """
    # 6. Température moyenne brute (capteur direct)
    df["Tavg_C"] = df["Tavg_C"]

    # 7. Amplitude thermique intra-cycle (stress mécanique + SEI)
    df["thermal_range"] = df["Tmax_C"] - df["Tmin_C"]

    # 8. Auto-échauffement au-dessus de l'ambiant (proxy IR interne)
    df["delta_T_ambient"] = df["Tavg_C"] - (df["T_amb_K"] - 273.15)

    # 9. Compteur cumulé de cycles froids (risque lithium plating LEO)
    df["cold_cycle_count"] = grp["Tmin_C"].transform(
        lambda x: (x < COLD_THRESHOLD_C).cumsum()
    )

    # 10. Régime thermique moyen lissé
    df["Tavg_rolling"] = grp["Tavg_C"].transform(
        lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).mean()
    )

    # 11. Flag phase orbitale : éclipse (1) vs soleil (0)
    #     T_amb alterne 263 K (éclipse) / 271 K (soleil) sur orbite LEO
    df["eclipse_flag"] = (df["T_amb_K"] < ECLIPSE_THRESHOLD_K).astype(int)

    return df


def build_current_features(df: pd.DataFrame, grp) -> pd.DataFrame:
    """
    Shunt résistif + intégrateur EPS → 5 features
    Sources : QD_Ah, QC_Ah, discharge_time_min
    """
    # 12. Capacité déchargée brute (mesure la plus directe du SOH)
    df["QD_Ah"] = df["QD_Ah"]

    # 13. Efficacité coulombique par cycle
    df["coulombic_eff"] = df["QD_Ah"] / df["QC_Ah"].replace(0, np.nan)

    # 14. C-rate de décharge (régime de sollicitation)
    discharge_h = df["discharge_time_min"] / 60.0
    df["discharge_C_rate"] = df["QD_Ah"] / discharge_h.replace(0, np.nan)

    # 15. Rétention de capacité normalisée depuis le cycle 1
    df["capacity_retention"] = grp["QD_Ah"].transform(
        lambda x: x / x.iloc[0]
    )

    # 16. Tendance lissée de la capacité déchargée
    df["QD_rolling"] = grp["QD_Ah"].transform(
        lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).mean()
    )

    return df


def build_obc_features(df: pd.DataFrame, grp) -> pd.DataFrame:
    """
    Calculé par l'OBC (On-Board Computer) → 4 features
    Sources : QD_Ah (cumul), SOH (lag), cycle
    """
    # 17. Indice de cycle (position temporelle absolue)
    df["cycle"] = df[CYCLE_COL]

    # 18. Énergie totale cyclée depuis le lancement
    df["cumul_Ah"] = grp["QD_Ah"].transform(lambda x: x.cumsum())

    # 19. SOH estimé au cycle précédent (feature auto-régressive)
    df["SOH_lag_1"] = grp[TARGET_COL].transform(lambda x: x.shift(1))

    # 20. Variation de capacité cycle à cycle (détecte les accélérations)
    df["QD_diff"] = grp["QD_Ah"].transform(lambda x: x.diff())

    return df


# =============================================================================
# FEATURE ENGINEERING PRINCIPAL
# =============================================================================

FEATURE_COLS = [
    # Tension (5)
    "V_mean_V", "V_spread", "V_mean_rolling", "V_mean_lag_1", "V_min_fade",
    # Température (6)
    "Tavg_C", "thermal_range", "delta_T_ambient",
    "cold_cycle_count", "Tavg_rolling", "eclipse_flag",
    # Courant / capacité (5)
    "QD_Ah", "coulombic_eff", "discharge_C_rate",
    "capacity_retention", "QD_rolling",
    # OBC calculé (4)
    "cycle", "cumul_Ah", "SOH_lag_1", "QD_diff",
]


def run(data_path: str = DATA_PATH) -> pd.DataFrame:
    print("=" * 60)
    print("  FEATURE ENGINEERING — CubeSat EPS Battery")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Chargement
    # ------------------------------------------------------------------
    print(f"\n[1/4] Chargement du dataset : {data_path}")
    df = pd.read_csv(data_path)
    print(f"      {len(df):,} lignes  |  {df.shape[1]} colonnes brutes")
    print(f"      Batteries : {df[ID_COL].unique().tolist()}")

    # ------------------------------------------------------------------
    # Construction des features par groupe batterie
    # ------------------------------------------------------------------
    print(f"\n[2/4] Construction des 20 features capteurs...")
    df = df.sort_values([ID_COL, CYCLE_COL]).reset_index(drop=True)
    grp = df.groupby(ID_COL)

    df = build_voltage_features(df, grp)
    df = build_temperature_features(df, grp)
    df = build_current_features(df, grp)
    df = build_obc_features(df, grp)

    print(f"      ✓ 5 features tension")
    print(f"      ✓ 6 features température")
    print(f"      ✓ 5 features courant / capacité")
    print(f"      ✓ 4 features OBC calculé")

    # ------------------------------------------------------------------
    # Sélection colonnes finales + cible
    # ------------------------------------------------------------------
    print(f"\n[3/4] Sélection des colonnes finales...")
    keep_cols = [ID_COL, CYCLE_COL] + FEATURE_COLS + [TARGET_COL]
    df_out = df[keep_cols].copy()

    # Supprimer les lignes avec NaN (lag/diff génèrent des NaN au cycle 1)
    n_before = len(df_out)
    df_out = df_out.dropna().reset_index(drop=True)
    n_dropped = n_before - len(df_out)
    print(f"      {n_dropped} lignes supprimées (NaN lag/diff cycle 1)")
    print(f"      Dataset final : {len(df_out):,} lignes × {len(FEATURE_COLS)} features + cible")

    # ------------------------------------------------------------------
    # Sauvegarde
    # ------------------------------------------------------------------
    print(f"\n[4/4] Sauvegarde → {FILE_FEATURES}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_out.to_csv(FILE_FEATURES, index=False)

    print(f"\n{'=' * 60}")
    print(f"  ✅ Features sauvegardées : {FILE_FEATURES}")
    print(f"  Dimensions : {df_out.shape[0]:,} lignes × {len(FEATURE_COLS)} features")
    print(f"{'=' * 60}\n")

    return df_out


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================
if __name__ == "__main__":
    df_features = run()
    print("Aperçu des 3 premières lignes :")
    print(df_features.head(3).to_string())
    print(f"\nStatistiques des features :")
    print(df_features[FEATURE_COLS].describe().round(4).to_string())
