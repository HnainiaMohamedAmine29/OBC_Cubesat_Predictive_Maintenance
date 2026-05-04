"""
Feature Engineering pour l'estimation du SOH
Basé sur l'analyse de la matrice de corrélation (Doc1_ang.pdf)
+ Features temporelles (rolling mean, rolling std, lags)
"""

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# 1. CHARGEMENT DU DATASET
# ─────────────────────────────────────────────

df = pd.read_csv("data_cleaned/cleaned_dataset.csv")
df = df.sort_values(['battery_id', 'cycle']).reset_index(drop=True)

print("Dataset original :")
print(f"  Lignes    : {df.shape[0]}")
print(f"  Colonnes  : {df.shape[1]}")
print()

# ─────────────────────────────────────────────
# 2. FEATURES DE BASE (depuis le document PDF)
#    Créées AVANT suppression pour utiliser toutes les colonnes
# ─────────────────────────────────────────────

# Différence de SOC  →  SOC_end - SOC_start
df['SOC_diff']   = df['SOC_end'] - df['SOC_start']

# Efficacité énergétique  →  QD_Ah / QC_Ah
df['efficiency'] = df['QD_Ah'] / df['QC_Ah']

# Plage de tension  →  V_max_V - V_min_V
df['V_range']    = df['V_max_V'] - df['V_min_V']

print("Features de base creees : SOC_diff, efficiency, V_range")

# ─────────────────────────────────────────────
# 3. SUPPRESSION DES VARIABLES REDONDANTES
#    QC_Ah     → r = 1.00 avec QD_Ah
#    SOC_start → r = 0.94 avec V_mean_V
# ─────────────────────────────────────────────

df = df.drop(columns=['QC_Ah', 'SOC_start'])

print("Colonnes redondantes supprimees : QC_Ah, SOC_start")
print()

# ─────────────────────────────────────────────
# 4. FEATURES TEMPORELLES
#
#    Variables choisies : liées physiquement au SOH
#      - SOH      : signal cible
#      - IR_ohm   : résistance interne (meilleur indicateur de vieillissement)
#      - QD_Ah    : capacité de décharge
#      - V_mean_V : tension moyenne
#      - Tavg_C   : température moyenne
#
#    Fenêtres rolling : [5, 10, 20, 50] cycles
#    Lags             : [1, 2, 5, 10]   cycles
# ─────────────────────────────────────────────

ROLLING_VARS    = ['SOH', 'IR_ohm', 'QD_Ah', 'V_mean_V', 'Tavg_C']
ROLLING_WINDOWS = [5, 10, 20, 50]

LAG_VARS  = ['SOH', 'IR_ohm', 'QD_Ah']
LAG_STEPS = [1, 2, 5, 10]

# ── 4.1 ROLLING MEAN ──────────────────────────
# Moyenne glissante sur les N derniers cycles
# → dénoise le signal et révèle la tendance locale

print("Calcul des rolling mean...")
for var in ROLLING_VARS:
    for w in ROLLING_WINDOWS:
        df[f'{var}_roll_mean_{w}'] = (
            df.groupby('battery_id')[var]
            .transform(lambda x: x.rolling(window=w, min_periods=1).mean())
        )
print(f"  -> {len(ROLLING_VARS) * len(ROLLING_WINDOWS)} features creees")

# ── 4.2 ROLLING STD ───────────────────────────
# Écart-type glissant sur les N derniers cycles
# → mesure la variabilité récente du signal
# → une std croissante sur SOH ou IR signale l'approche du knee point

print("Calcul des rolling std...")
for var in ROLLING_VARS:
    for w in ROLLING_WINDOWS:
        df[f'{var}_roll_std_{w}'] = (
            df.groupby('battery_id')[var]
            .transform(lambda x: x.rolling(window=w, min_periods=1).std().fillna(0))
        )
print(f"  -> {len(ROLLING_VARS) * len(ROLLING_WINDOWS)} features creees")

# ── 4.3 LAGS ──────────────────────────────────
# Valeur de la variable N cycles en arrière
# → introduit la mémoire du système dans le modèle
# → justifié par l'ACF qui montre une forte autocorrélation

print("Calcul des lags...")
for var in LAG_VARS:
    for lag in LAG_STEPS:
        df[f'{var}_lag{lag}'] = (
            df.groupby('battery_id')[var]
            .shift(lag)
            .bfill()
        )
print(f"  -> {len(LAG_VARS) * len(LAG_STEPS)} features creees")
print()

# ─────────────────────────────────────────────
# 5. BILAN FINAL
# ─────────────────────────────────────────────

n_rolling_mean = len(ROLLING_VARS) * len(ROLLING_WINDOWS)
n_rolling_std  = len(ROLLING_VARS) * len(ROLLING_WINDOWS)
n_lags         = len(LAG_VARS) * len(LAG_STEPS)
n_base         = 3

print("=" * 45)
print("BILAN DES FEATURES")
print("=" * 45)
print(f"  Features de base (PDF)  : {n_base}")
print(f"  Rolling mean            : {n_rolling_mean}")
print(f"  Rolling std             : {n_rolling_std}")
print(f"  Lags                    : {n_lags}")
print(f"  ---------------------------------")
print(f"  Total features ajoutees : {n_base + n_rolling_mean + n_rolling_std + n_lags}")
print(f"  Colonnes totales        : {df.shape[1]}")
print("=" * 45)
print()

nulls = df.isnull().sum().sum()
print(f"Valeurs nulles restantes : {nulls}")
print()

# ─────────────────────────────────────────────
# 6. SAUVEGARDE
# ─────────────────────────────────────────────

output_path = "dataset_features_created/final_dataset_4.csv"
df.to_csv(output_path, index=False)

print(f"Fichier sauvegarde : {output_path}")