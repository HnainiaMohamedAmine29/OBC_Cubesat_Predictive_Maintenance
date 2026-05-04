# Feature Engineering — SOH Estimation

## Fichiers

| Fichier | Description |
|---|---|
| `feature_engineering_soh.py` | Script Python principal |
| `feature_engineering_soh.csv` | Dataset final avec toutes les features |

---

## Lancer le script

```bash
python feature_engineering_soh.py
```

> Le script lit `outputs/preprocessing/battery_dataset_cleaned.csv` et sauvegarde le résultat dans `outputs/feature_engineering_soh.csv`.

---

## Ce que fait le script

### Étape 1 — Features de base

Créées avant la suppression pour pouvoir utiliser toutes les colonnes.

| Feature | Formule | Signification |
|---|---|---|
| `SOC_diff` | `SOC_end - SOC_start` | Profondeur réelle de décharge utilisée |
| `efficiency` | `QD_Ah / QC_Ah` | Efficacité coulombique (≈1.0 en bonne santé) |
| `V_range` | `V_max_V - V_min_V` | Amplitude de tension par cycle |

### Étape 2 — Suppression des colonnes redondantes

| Colonne supprimée | Raison |
|---|---|
| `QC_Ah` | Corrélation r = 1.00 avec `QD_Ah` |
| `SOC_start` | Corrélation r = 0.94 avec `V_mean_V` |

### Étape 3 — Features temporelles

**Variables utilisées** (physiquement liées au SOH) :

```
SOH, IR_ohm, QD_Ah, V_mean_V, Tavg_C
```

**Rolling Mean** — moyenne glissante sur les N derniers cycles :

```
fenêtres : [5, 10, 20, 50]
→ 5 variables × 4 fenêtres = 20 features
ex : SOH_roll_mean_10, IR_ohm_roll_mean_20 ...
```

**Rolling Std** — écart-type glissant sur les N derniers cycles :

```
fenêtres : [5, 10, 20, 50]
→ 5 variables × 4 fenêtres = 20 features
ex : SOH_roll_std_5, IR_ohm_roll_std_10 ...
```

**Lags** — valeur de la variable N cycles en arrière :

```
lags    : [1, 2, 5, 10]
variables: SOH, IR_ohm, QD_Ah
→ 3 variables × 4 lags = 12 features
ex : SOH_lag1, IR_ohm_lag5, QD_Ah_lag10 ...
```

---

## Bilan

| Groupe | Nombre |
|---|---|
| Features de base | 3 |
| Rolling mean | 20 |
| Rolling std | 20 |
| Lags | 12 |
| **Total ajouté** | **55** |
| **Colonnes finales** | **71** |
