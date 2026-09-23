# Technical Documentation — Data Preparation Pipeline (LSTM-based SOH Estimation)

**Document scope:** `04_pipline_data_preparing`
**Files analyzed:** `config.py`, `feature_engineering.py`, `split.py`, `normalize.py`, `sequences_construction.py`, `README.md`

This document complements the documentation of modules 02 (physical simulation model) and 03 (visualization): it describes the **data preparation pipeline** that transforms the raw dataset `battery_dataset_normal.csv` (output of module 02) into supervised tensors ready to train a recurrent neural network (LSTM) for battery State of Health (SOH) estimation.

---

## 1. Purpose and place in the processing chain

The pipeline addresses a classic **embedded prognostics and health management (PHM)** need: from a CubeSat battery pack's recent history of sensor measurements, estimate its current state of health (and, eventually, predict its degradation trajectory) via a sequential learning model.

```
battery_dataset_normal.csv  (module 02 output — 1 row = 1 orbital cycle)
        │
        ▼
[1] feature_engineering.py   → processed/features.csv        (20 features + SOH target)
        │
        ▼
[2] split.py                 → processed/train.csv / test.csv (80/20 temporal split)
        │
        ▼
[3] normalize.py             → processed/train_normalized.csv / test_normalized.csv
        │                       + scaler_X.pkl (persisted for embedded inference)
        ▼
[4] sequences_construction.py → X_train.npy / y_train.npy / X_test.npy / y_test.npy
        │                        (tensors (N, WINDOW_SIZE, 20) ready for the LSTM)
        ▼
[5] model.py (mentioned in the README, not provided in this archive)
                              → cubesat_soh_lstm.keras, metrics.csv, training_history.csv
```

### 1.1 Guiding principle: centralized configuration

The `README.md` explicitly states the repository's philosophy: **"configure once, run everywhere"**. Only `config.py` is meant to be modified from one mission to another; the four processing scripts (`feature_engineering.py`, `split.py`, `normalize.py`, `sequences_construction.py`) **all import their parameters** from this central file rather than hard-coding them. This is a solid software-engineering principle for a pipeline intended to be reused across several datasets (normal / fault / a different cell type, see module 03) without risk of divergence between scripts.

### 1.2 Documented correction history

The `README.md` documents an **explicit correction history**, a rare and noteworthy traceability practice:

1. **Fixed blocking bug**: the scripts were importing variable names (`FILE_FEATURES`, `FILE_TRAIN`, etc.) that did not exist in an earlier version of `config.py` (which defined `FEATURES_PATH`, `TRAIN_PATH`, etc.) — the pipeline simply could not run. The new names are now the source of truth, with the old ones kept as **compatibility aliases** (`FEATURES_PATH = FILE_FEATURES`, etc.).
2. **Hyperparameter realignment** against a reference notebook (`prediction_soh_lstm_target(_SOH).ipynb`): `ROLLING_WINDOW` 5→10, `TRAIN_RATIO` 0.70→0.80, `STEP_SIZE` 1→3.
3. **Feature formula realignment** (`coulombic_eff` inverted, `discharge_C_rate`, `cold_cycle_count`, `delta_T_ambient` redefined) — detailed in section 2.
4. **Addition of `model.py`** carrying the notebook's architecture — **not present in the archive provided here** (only the 5 files listed at the top of the document are delivered). Section 6 of this document nevertheless reconstructs the intended architecture from the hyperparameters exposed in `config.py`.
5. **Documented and accepted inconsistency on the output activation function** (`sigmoid` vs. `linear`) — see section 6.3.

---

## 2. Step 1 — Feature engineering (`feature_engineering.py`)

### 2.1 General principle

The script builds **20 features** organized into four groups, each corresponding to a **family of real physical sensors/sources** of a CubeSat EPS (Electrical Power System): voltage (ADC), temperature (NTC thermistor), current (shunt resistor + integrator), and quantities computed onboard by the OBC (On-Board Computer). This grouping by sensor is not merely a coding convention: it **implicitly documents the real acquisition chain** this pipeline is meant to reflect for an embedded deployment (each feature has a traceable hardware origin), a good practice for later systems-engineering use (sensor criticality analysis, fault tolerance).

Features are computed **per battery group** (`df.groupby(ID_COL)`), guaranteeing that no rolling-window, shift, or diff operation mistakenly mixes the histories of two distinct batteries — an important point of rigor should the pipeline ever be run on a multi-battery dataset (`n_batt > 1` in module 02).

### 2.2 Group 1 — Voltage (5 features)

| # | Feature | Formula | Technical interpretation |
|---|---|---|---|
| 1 | `V_mean_V` | direct measurement | raw average voltage of the cycle |
| 2 | `V_spread` | `V_max_V − V_min_V` | intra-cycle voltage amplitude, an indirect proxy for the ohmic drop (hence for internal resistance) |
| 3 | `V_mean_rolling` | moving average of `V_mean_V` over `ROLLING_WINDOW=10` cycles | smoothed trend, filters cycle-to-cycle noise (random orbital variability identified in module 03) |
| 4 | `V_mean_lag_1` | `V_mean_V` shifted by one cycle (`shift(1)`) | autoregressive feature, gives the model the previous cycle's value as a reference |
| 5 | `V_min_fade` | `V_min_V(t) − V_min_V(cycle 1)` | absolute drift of the minimum discharge voltage relative to the beginning of life — a direct electrical-fade indicator |

### 2.3 Group 2 — Temperature (6 features)

| # | Feature | Formula | Technical interpretation |
|---|---|---|---|
| 6 | `Tavg_C` | direct measurement | raw average temperature of the cycle |
| 7 | `thermal_range` | `Tmax_C − Tmin_C` | intra-cycle thermal amplitude — a proxy for cyclic mechanical stress (expansion/contraction) and stress on the SEI interphase |
| 8 | `delta_T_ambient` | `diff(T_amb_K)` from one cycle to the next | structural temperature variation from one cycle to the next — captures the eclipse↔sunlight orbital transition |
| 9 | `cold_cycle_count` | cumulative count of `(Tavg_C < 5 °C)` | cumulative counter of "cold" cycles, a proxy for lithium plating risk, an accelerated degradation mechanism in a cold LEO environment |
| 10 | `Tavg_rolling` | moving average of `Tavg_C` over 10 cycles | smoothed average thermal regime |
| 11 | `eclipse_flag` | `1` if `T_amb_K < 263.15 K`, else `0` | binary indicator of the orbital phase (eclipse vs. sunlight) |

**Point explicitly flagged in the code** (lines 56–62): the `delta_T_ambient` feature changed definition between an earlier version of the script and the current one, to stay faithful to the reference notebook. The old formula computed a **self-heating gap** (`Tavg_C − (T_amb_K − 273.15)`, a proxy for internal Joule dissipation / resistance), while the current formula computes a **pure ambient variation** (`diff(T_amb_K)`, a proxy for orbital transition). These are two physically distinct quantities that carried the same historical name — a risk of confusion correctly documented in a code comment, but which would benefit from an explicit rename (e.g., `delta_T_ambient_diff` vs. `self_heating_proxy`) to eliminate any future ambiguity.

### 2.4 Group 3 — Current / capacity (5 features)

| # | Feature | Formula | Technical interpretation |
|---|---|---|---|
| 12 | `QD_Ah` | direct measurement | discharged capacity — the measurement most directly correlated with SOH (see correlation ≈ −0.96 documented in module 03) |
| 13 | `coulombic_eff` | `QC_Ah / QD_Ah` | coulombic efficiency of the cycle (recovered charge / delivered discharge); protected against division by zero via `.replace(0, np.nan)` |
| 14 | `discharge_C_rate` | `DoD / (discharge_time_min/60)` (falls back to `QD_Ah/Q_NOM` if `DoD` is absent from the CSV) | discharge stress regime (effective C-rate), useful for distinguishing aging accelerated by heavy stress |
| 15 | `capacity_retention` | `QD_Ah(t) / QD_Ah(cycle 1)` | capacity retention normalized since the beginning of life — a normalized SOH variant built solely from the capacity measurement |
| 16 | `QD_rolling` | moving average of `QD_Ah` over 10 cycles | smoothed trend of discharged capacity |

The code comment highlights an **important correction point** for `coulombic_eff`: the formula was inverted relative to an earlier version (`QD_Ah/QC_Ah` → `QC_Ah/QD_Ah`) to match the usual definition of coulombic efficiency (useful output / supplied input... here, rather, charge recovered in sunlight / discharge delivered in eclipse, consistent with the direction of energy flow orbit by orbit). An inversion of this kind changes the feature's direction of variation (a value > 1 becomes < 1 and vice versa) — a particularly dangerous silent error for a learning model, which would then learn an inverted relationship with no error ever being raised.

### 2.5 Group 4 — Computed by the OBC (4 features)

| # | Feature | Formula | Technical interpretation |
|---|---|---|---|
| 17 | `cycle` | `CYCLE_COL` | absolute temporal position (orbital cycle number) |
| 18 | `cumul_Ah` | cumulative sum (`cumsum`) of `QD_Ah` | total energy cycled since launch — a cumulative-throughput proxy, correlated with cycling aging (see the `dSOH/dt ∝ I` model in module 02) |
| 19 | `SOH_lag_1` | `SOH` shifted by one cycle | autoregressive feature: the model receives the previous cycle's SOH as direct information |
| 20 | `QD_diff` | `diff(QD_Ah)` from one cycle to the next | cycle-to-cycle capacity variation — detects sudden accelerations/drops in degradation (particularly relevant for distinguishing the discontinuous-step `fault_state` scenario identified in module 03) |

**Important engineering note on `SOH_lag_1`:** this feature directly injects the **one-step-lagged target** as an input variable. This is a standard time-series forecasting practice ("autoregressive feature"), but it makes the model **heavily dependent on the availability of the previous cycle's SOH at inference time**. In real embedded operation, SOH is never directly measured (it is precisely the quantity the model is trying to estimate): `SOH_lag_1` would then need to be filled in by the **model's own prediction from the previous step** (recursive inference), a practice that introduces a risk of cumulative error (drift) not addressed by this data preparation pipeline — to be anticipated in the design of `model.py` / the embedded inference loop.

### 2.6 Final cleanup and export

```python
df_out = df_out.dropna().reset_index(drop=True)
```

Rows containing `NaN` — inevitably produced by the `shift(1)` and `diff()` operations at the first cycle of each battery (no previous cycle available) — are removed. The script logs the number of rows removed, a good traceability practice for detecting an abnormally high data loss (which would, for example, signal a sorting or battery-grouping issue).

The final file `processed/features.csv` contains: `battery_id`, `cycle`, the 20 features, then the `SOH` target.

---

## 3. Step 2 — Train/test split (`split.py`)

### 3.1 Two splitting strategies

| Method | Principle | Associated risk |
|---|---|---|
| `"temporal"` (recommended, default) | The first `TRAIN_RATIO` (80%) of cycles → train; the remaining cycles → test, after global sorting by `CYCLE_COL` | No risk of temporal leakage if correctly implemented |
| `"random"` | Random draw (`df.sample`) of a fraction of the rows | **Risk of data leakage** explicitly flagged by the script itself (`⚠️ Random split: risk of data leakage on time series`): future cycles could end up in training while past cycles (direct, highly correlated temporal neighbors) end up in test |

The choice of temporal split as the default method is the methodologically correct choice for a time-series prognostics task: it simulates the real operating situation, where the model only has access to the past to predict the future.

### 3.2 Non-overlap check

```python
if min_test_cycle <= max_train_cycle:
    print("⚠️ Cycle overlap detected!")
else:
    print("✓ No overlap — clean split")
```

This post-hoc check is a **good automated-validation practice**: it numerically confirms, on every run, that the train/test temporal boundary is respected — useful in particular should the pipeline someday be changed to a per-battery split rather than a global one.

### 3.3 Identified limitation: no per-battery split

The script header carries the comment *"Adapted for a dataset without a battery_id column. Global temporal split"*. With `n_batt=1` (module 02's current configuration), this assumption has no consequence: there is only a single time trajectory, so a global temporal split is strictly equivalent to a per-battery temporal split.

**However, if the pipeline were run on a multi-battery dataset** (`n_batt > 1`), the **global** temporal split (based solely on `CYCLE_COL`, without considering `battery_id`) would mix the cycles of all batteries into the same **relative** chronological order (battery A's cycle 1 with battery B's cycle 1, etc.), rather than guaranteeing that **each battery individually** is cut at 80% of its own lifetime. Depending on how battery identifiers are distributed in the sorted file, this could in some cases result in late cycles of a battery B (i.e., degraded health states) ending up in training while early cycles of a battery A (healthy state) end up in test — a subtle form of mixing that would partially break the intended strict temporal separation guarantee. This point should be fixed (splitting by `battery_id` group, with a `TRAIN_RATIO` applied **within each group**) before any extension to a multi-battery dataset.

---

## 4. Step 3 — Normalization (`normalize.py`)

### 4.1 Fundamental rule: fit exclusively on the train set

```python
scaler.fit(train[cols_scaled])
train_norm[cols_scaled] = scaler.transform(train[cols_scaled])
test_norm[cols_scaled]  = scaler.transform(test[cols_scaled])
```

The scaler (`StandardScaler` or `MinMaxScaler`, depending on `config.SCALER_TYPE`) is **fit only on the training data**, then applied (transform) identically to both train and test. This is the essential golden rule for avoiding any information leakage from test into train (the test set must not influence any statistic used during training) — correctly implemented here, with an explicit comment at the top of the file recalling this rule.

### 4.2 Columns excluded from normalization

```python
COLS_NO_SCALE = ["battery_id", "cycle", "eclipse_flag"]
```

- `cycle`: kept as a raw value (an ordinal quantity, its magnitude carries directly usable positional/temporal information);
- `eclipse_flag`: a binary variable (0/1), normalizing a variable already bounded on an interpretable scale would add nothing and would complicate its reading;
- `battery_id`: a categorical identifier, actually absent from `FEATURE_COLS` (it is only used for `groupby` in `feature_engineering.py`) — its presence in `COLS_NO_SCALE` is therefore **redundant but harmless**.

The `SOH` target is **never normalized** — a choice consistent with the goal of obtaining a model output that is directly interpretable as a health percentage (0.70–1.00), with no denormalization step required after inference.

### 4.3 Post-normalization check

```python
train_means = train_norm[cols_scaled].mean().abs().max()
train_stds  = train_norm[cols_scaled].std().max()
```

Automatic check that the mean and standard deviation of the normalized train set are indeed close to 0 and 1 respectively (`StandardScaler` case) — a simple but effective consistency test for detecting an implementation error (for example a column offset).

### 4.4 Persisting the scaler for embedded inference

```python
joblib.dump(scaler, FILE_SCALER)
with open(FILE_SCALER_COLS, "w") as f:
    f.write("\n".join(cols_scaled))
```

The fitted scaler is serialized (`scaler_X.pkl`) together with the ordered list of normalized columns (`scaler_columns.txt`), and a `load_scaler()` function is provided to reload it. This is an essential step for a real inference deployment (embedded onboard the CubeSat, or ground-based): new measurements will need to be normalized with **exactly the same parameters** (mean/standard deviation or min/max) as those learned during training, or risk a distribution shift (*covariate shift*) that would invalidate the model's predictions.

---

## 5. Step 4 — LSTM sequence construction (`sequences_construction.py`)

### 5.1 Sliding-window principle

```python
for start in range(0, max_start, step):
    end = start + window
    target_idx = start + window + horizon - 1
    X_list.append(features[start:end])
    y_list.append(targets[target_idx])
```

Each training sample is built by taking a **window of `WINDOW_SIZE` consecutive cycles of features** as input, and the SOH at cycle `WINDOW_SIZE + HORIZON` (counted from the start of the window) as the target — a classic **"one-step-ahead forecasting"** task in supervised time-series learning.

### 5.2 Chosen parameters and justification (documented in the README)

| Parameter | Value | Summarized justification |
|---|---|---|
| `WINDOW_SIZE` | 30 cycles | ≈ 2 days of orbital history (≈ 15 orbits/day in LEO); a trade-off between enough temporal context to capture short-term aging trends and a tensor size compatible with the memory constraints of a hardened embedded microcontroller (30 × 20 = 600 values per sample) |
| `STEP_SIZE` | 3 | the sweep step — a value aligned with the reference notebook; a step > 1 reduces overlap between successive windows (less redundancy, faster training) at the cost of a total number of sequences reduced by a factor of ≈ 3 relative to `STEP_SIZE=1` |
| `HORIZON` | 1 | predicting the SOH at the cycle immediately following the window — a near-real-time tracking/prognostics task |

The `README.md` provides a detailed physical justification for the `WINDOW_SIZE=30` choice (an empirical rule "dominant period × number of observations per period", embedded memory constraint, preliminary tests not included in the archive) — a level of design-choice documentation rarely found in a data preparation pipeline, and a good practice worth noting.

**Consistency note with module 03:** the ACF/PACF analysis in module 03 showed that the **real, structurally dominant period** in the thermal variables is **2 cycles** (the programmed alternation of the eclipse structural temperature), not 20 or 30. The justification for `WINDOW_SIZE=30` put forward in the README ("capture the dominant period") therefore relies more on an operational time scale (2 days of orbital context) than on the strict statistical periodicity measured in module 03; the two justifications are not contradictory, but would benefit from being explicitly connected in the project's documentation.

### 5.3 Single source of truth for hyperparameters

The comment at the top of the file specifies that `WINDOW_SIZE`, `STEP_SIZE` and `HORIZON` are now imported from `config.py` rather than redefined locally — a correction explicitly motivated by a history of **value drift** between this script and the reference notebook. This is the same centralized-configuration logic described in section 1.1.

### 5.4 Safeguards and checks

```python
assert X_train.ndim == 3
assert X_train.shape[1] == WINDOW_SIZE
assert X_train.shape[2] == len(feature_cols)
assert not np.any(np.isnan(X_train))
```

Four assertions validate the shape and integrity of the produced tensors before saving — a simple but effective safeguard against a silent construction error (for example a desynchronization between `feature_cols` and the columns actually present in the normalized CSV).

One dynamic column-selection line deserves to be noted:

```python
feature_cols = [c for c in FEATURE_COLS if c in
                pd.read_csv(FILE_TRAIN_NORM, nrows=0).columns]
```

The associated comment ("fix for the cycle/cycle.1 duplicate") indicates that this step corrects an earlier problem where a duplicated `cycle` column (probably from an unfortunate merge or reindexing in an upstream step) could have been selected twice or incorrectly referenced. Reading only the headers (`nrows=0`) is an efficient trick to validate column presence without loading the entire file.

### 5.5 Produced outputs

- `X_train.npy`, `X_test.npy`: tensors of shape `(N, WINDOW_SIZE=30, 20)`.
- `y_train.npy`, `y_test.npy`: target vectors of shape `(N,)`.
- `sequences_info.txt`: a traceability file summarizing the parameters and shapes of the tensors produced — useful for auditing a training run after the fact.

With `STEP_SIZE=3`, the number of sequences produced is approximately `(n_rows − WINDOW_SIZE − HORIZON + 1) / 3`, to be applied separately to the train and test partitions (each processed independently by `build_sequences`, guaranteeing that no window crosses the train/test boundary).

---

## 6. Step 5 — Model (`model.py`, referenced but not provided in this archive)

The `README.md` describes a `model.py` script that is absent from the files delivered in `04_pipline_data_preparing`. The full hyperparameters of this model are nevertheless defined in `config.py` (the *MODEL ARCHITECTURE & TRAINING* section), which makes it possible to reconstruct the intended architecture for documentation purposes, provided the actual script faithfully matches this configuration.

### 6.1 Architecture (LSTM + self-attention)

Based on `config.py` and the README, `build_model_v2()` stacks:

```
Input (WINDOW_SIZE=30, NUM_FEATURES=20)
   → LSTM(128, return_sequences=True) + Dropout(0.2)
   → LSTM(64,  return_sequences=True) + Dropout(0.2)
   → Multi-head self-attention (4 heads × 16 dim = 64, consistent with the 2nd LSTM's output) + Dropout(0.1)
   → Dense(64, activation='elu')     + Dropout(0.1)
   → Dense(1, activation=OUTPUT_ACTIVATION)
```

Adding a **self-attention** block after the LSTM layers is a relevant architecture for this type of task: it allows the model to dynamically weight the relative importance of each cycle in the 30-step window (for example, giving more weight to a cycle showing an abnormal voltage drop) rather than relying solely on the sequential compression of the final LSTM hidden state.

### 6.2 Regularization and optimization

| Parameter | Value | Role |
|---|---|---|
| `L2_REG` | 2×10⁻⁴ | L2 penalty on the LSTM/Dense kernels, also used as AdamW *weight decay* |
| `LEARNING_RATE` | 3×10⁻⁴ | initial learning rate |
| `CLIP_NORM` | 1.0 | gradient-norm clipping — protection against gradient explosion, relevant for a deep LSTM |
| `HUBER_DELTA` | 0.01 | Huber loss transition threshold (quadratic below, linear above) — ≈ 1% SOH error, makes training robust to outlier cycles without being as sensitive as a pure MSE loss |
| `BATCH_SIZE` | 128 | batch size |
| `EPOCHS` (max) | 400, with `EARLY_STOPPING_PATIENCE=25` | early stopping if no improvement for 25 epochs |
| `REDUCE_LR_PATIENCE` / `REDUCE_LR_FACTOR` | 5 / 0.3 | learning-rate reduction (×0.3) after 5 epochs with no improvement — standard *ReduceLROnPlateau* strategy |
| `OPTIMIZER` | AdamW | Adam with decoupled weight decay, a choice consistent with the explicit use of L2/weight decay above |
| `LOSS` / `METRIC` | Huber / MAE | robust training loss, a tracking metric directly interpretable in SOH units |

### 6.3 Documented inconsistency: output activation function

The `README.md` flags a noteworthy point of **methodological transparency**: the reference notebook contains a comment recommending a `linear` output activation (justification: the `sigmoid` function's gradient becomes too flat near SOH=1.0, which would bias learning upward in early life), **but the code of the notebook actually run nevertheless uses `activation='sigmoid'`**. The pipeline reproduces this choice (`OUTPUT_ACTIVATION = "sigmoid"`) to stay faithful to the results actually obtained and evaluated, while explicitly documenting the fix to be pursued in future work.

This decision to **document a known contradiction rather than silently fixing it** is a sound engineering practice: it prevents a future user of the pipeline from discovering the `sigmoid` output without understanding whether it was a deliberate choice or an oversight, and leaves a record of the trade-off made between fidelity to already-validated results and theoretical correctness.

**Technical analysis of the trade-off:** a `sigmoid` output naturally bounds the prediction to `[0,1]`, which is consistent with the physical nature of SOH (never negative, rarely > 1). The flagged risk (near-zero gradient close to 1.0) is real for early-life samples (SOH close to 0.99), where the model will receive a very weak learning signal and could therefore converge more slowly or in a biased way on that range — a point specifically worth monitoring in the error curves per SOH range (comparable to the SOH-conditioned analysis already performed in module 03), once `model.py` is available and has been run.

### 6.4 Declared performance targets

```python
RMSE_TARGET = 0.02   # 2% absolute error on SOH
MAE_TARGET  = 0.015  # 1.5%
R2_TARGET   = 0.95
```

These thresholds, defined in `config.py`, constitute **a priori acceptance criteria** for judging model quality once trained — a good practice of setting performance objectives before experimentation rather than after the fact.

---

## 7. Summary table of the 20 features

| Group | Feature | Type | Normalized? |
|---|---|---|---|
| Voltage | `V_mean_V` | continuous | yes |
| Voltage | `V_spread` | continuous | yes |
| Voltage | `V_mean_rolling` | continuous | yes |
| Voltage | `V_mean_lag_1` | continuous | yes |
| Voltage | `V_min_fade` | continuous | yes |
| Temperature | `Tavg_C` | continuous | yes |
| Temperature | `thermal_range` | continuous | yes |
| Temperature | `delta_T_ambient` | continuous | yes |
| Temperature | `cold_cycle_count` | cumulative integer | yes |
| Temperature | `Tavg_rolling` | continuous | yes |
| Temperature | `eclipse_flag` | binary | **no** |
| Current/capacity | `QD_Ah` | continuous | yes |
| Current/capacity | `coulombic_eff` | continuous | yes |
| Current/capacity | `discharge_C_rate` | continuous | yes |
| Current/capacity | `capacity_retention` | continuous | yes |
| Current/capacity | `QD_rolling` | continuous | yes |
| OBC | `cycle` | ordinal | **no** |
| OBC | `cumul_Ah` | cumulative continuous | yes |
| OBC | `SOH_lag_1` | continuous (autoregressive) | yes |
| OBC | `QD_diff` | continuous | yes |

---

## 8. Limitations and possible improvements

1. **Non-robust multi-battery split** (detailed in 3.3): the global temporal split does not guarantee a clean per-battery separation if `n_batt > 1` — to be fixed before any move to the Monte Carlo scaling mentioned in module 03.
2. **`SOH_lag_1` at real inference time** (detailed in 2.5): this autoregressive feature assumes the availability of the real previous cycle's SOH, available during training (known labels) but not in real embedded operation — the recursive inference strategy (using the model's own prediction) and its cumulative-drift risk are not addressed by this data preparation pipeline.
3. **Historical ambiguity of `delta_T_ambient`** (detailed in 2.3): two physically different definitions have carried the same column name across different versions of the code; an explicit rename would eliminate the risk of confusion for a future contributor who does not read the comments in detail.
4. **`model.py` absent from the archive**: its architecture and hyperparameters are documented in `config.py` and the `README.md`, but the script itself could not be directly analyzed; the analysis in section 6 therefore relies on the declared configuration, not on a reading of the actual execution code.
5. **Accepted but unresolved sigmoid/linear inconsistency** (detailed in 6.3): a comparative test of the two output activations, with error analysis per SOH range, would allow this to be settled objectively rather than defaulting to the original notebook's choice.
6. **`STEP_SIZE=3` reduces the volume of sequences by a factor of ≈ 3** relative to an exhaustive sweep (`STEP_SIZE=1`): an accepted, documented trade-off (training speed vs. window redundancy), but one that would benefit from being quantified (number of sequences obtained, comparison of validation performance with `STEP_SIZE=1`) to confirm it does not cause significant information loss, particularly if the final dataset turns out to be modest in size after EOL filtering.

---

## 9. Conclusion

This pipeline constitutes a **rigorous, well-tracked data preparation chain**, structured around a centralized-configuration principle that limits the risk of divergence between scripts. It correctly applies the fundamental rules of data preparation for time-series learning: feature construction grouped by entity (battery), a strictly temporal split with automatic non-overlap verification, normalization fit exclusively on training data with the scaler persisted for inference, and sliding-sequence construction sized according to an explicit physical justification (orbital context, embedded memory constraint). The associated documentation (`README.md`, comments at the top of each script) stands out for its transparency regarding the correction history and at least one known, unresolved inconsistency (output activation) — a traceability practice worth commending and continuing as the pipeline evolves, particularly once `model.py` is actually integrated and a possible move to a multi-battery dataset takes place.
