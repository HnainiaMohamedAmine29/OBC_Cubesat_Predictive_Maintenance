# Technical Documentation — LSTM Model for SOH Estimation (Module 05)

**Document scope:** `05_Model_`
**Files analyzed:**
- `Reference_Model/prediction_soh_lstm_target(_SOH).ipynb` + `cubesat_soh_lstm_unrolled.keras` (**v2** architecture)
- `optimized_model/prediction_soh_lstm_target__SOH__v3_optimized.ipynb` + `cubesat_soh_lstm_v3_unroll_True.keras` (**v3** architecture)

This document complements modules 02 (physical simulation), 03 (visualization) and 04 (data preparation pipeline). It documents the **deep learning model** responsible for estimating battery State of Health (SOH) from the feature sequences built upstream, as well as its evolution from a reference architecture (v2) to an architecture optimized for embedded use (v3).

---

## 1. Archive contents and positioning

The archive delivers **two standalone Google Colab notebooks**, each accompanied by its corresponding trained Keras model:

| Folder | Notebook | Saved model | Role |
|---|---|---|---|
| `Reference_Model/` | `prediction_soh_lstm_target(_SOH).ipynb` | `cubesat_soh_lstm_unrolled.keras` (1.86 MB) | Reference architecture (**v2**), LSTM + multi-head self-attention |
| `optimized_model/` | `prediction_soh_lstm_target__SOH__v3_optimized.ipynb` | `cubesat_soh_lstm_v3_unroll_True.keras` (505 KB) | **v3** architecture, reduced for an embedded microcontroller target |

Both notebooks share an identical 15-section structure (imports, paths, loading, feature engineering, split, normalization, sequence construction, architecture, training, prediction, evaluation, visualizations, saving) and differ significantly only in **section 10 (model architecture)**. It is this architecture that evolved from v2 to v3; the rest of the data pipeline is reused identically between the two notebooks.

### 1.1 Important architectural point: duplication of the module 04 pipeline

The two notebooks **do not consume** the artifacts produced by the module 04 pipeline (`train_normalized.csv`, `X_train.npy`, etc.). They start again from the raw file `battery_dataset_normal.csv` (direct output of module 02) and a `features.pkl` file containing only the **list of feature column names** (not the data itself), then **fully reimplement**, within the notebook body itself:

- the `engineer_features()` function (section 5), logically identical to module 04's `feature_engineering.py`;
- the chronological train/validation split (section 7), logically identical to `split.py`;
- fit-train-only `StandardScaler` normalization (section 8), logically identical to `normalize.py`;
- the `build_sequences()` function (section 9), logically identical to `sequences_construction.py`.

This duplication is consistent with the context given by module 04's `README.md` ("*this pipeline was updated to stay consistent with the Colab notebook that contains the model architecture*"): the Colab notebook is actually the **original reference version**, and module 04's pipeline is a **later replication** of it in the form of modular scripts. Both implementations have been verified to be aligned formula by formula (see module 04, "Adaptation to the notebook" section). Nevertheless, a **two-location maintenance risk** remains: any future correction to the feature-engineering, split, or normalization logic would need to be manually carried over to both codebases to avoid renewed drift — exactly the kind of problem module 04 documents having already experienced and fixed once (desynchronized `ROLLING_WINDOW`, `TRAIN_RATIO`, `STEP_SIZE` parameters).

---

## 2. Recap of the data pipeline internal to the notebook (sections 1 to 9)

### 2.1 Reproducibility

```python
os.environ['PYTHONHASHSEED']       = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(42); np.random.seed(42); tf.random.set_seed(42)
```

All sources of randomness (Python hashing, TensorFlow operations, NumPy/Python generators) are fixed at the top of the notebook. This is an essential good practice for **experimental reproducibility** of a model intended to be compared across versions (v2 vs. v3) and audited.

### 2.2 Loading and feature engineering

Loading (`pd.read_csv(DATA_PATH)`) explicitly drops the `battery_id` column (`errors='ignore'`), consistent with the fact that the dataset contains only a single battery (`n_batt=1` in module 02) and that sequence construction is subsequently treated as **a single global time series** — logic identical to that documented in `sequences_construction.py` from module 04 ("*version WITHOUT battery_id*").

The `engineer_features()` function rebuilds the 20 features conditionally on their presence in `FEATURE_COLS` (loaded from `features.pkl`), with the same formulas already validated in module 04 — notably `coulombic_eff = QC_Ah/QD_Ah`, `discharge_C_rate = DoD/(discharge_time_min/60)`, `cold_cycle_count` on the `Tavg_C < 5°C` threshold, `delta_T_ambient = diff(T_amb_K)`. The fact that these formulas are **identical character for character** between this notebook and `feature_engineering.py` confirms that the alignment described in module 04 was indeed successfully achieved.

### 2.3 Chronological split and a note on extrapolation

```python
n_train = int(n_total * TRAIN_FRAC)   # TRAIN_FRAC = 0.80
train_df = raw.iloc[:n_train]
val_df   = raw.iloc[n_train:]
```

The notebook explicitly displays an educational warning after the split:

> ⚠ *Note: the val set covers a SOH range LOWER than the train set's. The model must extrapolate slightly (SOH 0.78 → 0.70). This is normal for a temporal split over a monotonic degradation.*

This is an important, correctly anticipated technical observation: since SOH decreases monotonically over time (see module 02), a purely chronological split **necessarily** places the lowest SOH values (end of life) in the validation set, never seen during training. The model is therefore evaluated partly in an **extrapolation regime**, which is a more demanding test — and more representative of real usage — than a random split that would have mixed the entire SOH range across both sets.

### 2.4 Normalization: removal of the target scaler (documented fix)

The notebook carries a correction note near the top: **"`scaler_y` removed"**, with the following explanation (section 8):

> A `MinMaxScaler` fit on the train set (SOH ∈ [0.78, 1.0]) maps the val set values (SOH ∈ [0.70, 0.78]) to **negative** values ([−0.36, 0]). Since the `sigmoid` output is bounded to [0, 1], the model then caps all its predictions toward ≈ 0 right from the start and receives no useful gradient.

This is a concrete, well-diagnosed example of a **classic normalization pitfall in the presence of temporal drift (distribution shift)**: normalizing a target with statistics learned only on the training set, while the test set explores a range of values outside that learning range, produces normalized targets outside the interval covered by the output activation function — a textbook case that completely blocks learning (zero gradient in `sigmoid` saturation). The fix adopted is physically justified: SOH is **already bounded to [0, 1] by physical construction** (it is a capacity fraction), so there is no statistical need to normalize it — only the input features (`X`) are normalized (`StandardScaler`, fit on train only), with the SOH target remaining in raw values throughout.

### 2.5 Sequence construction

```python
WINDOW_SIZE=30  STEP_SIZE=3  HORIZON=1
```

Logic strictly identical to `sequences_construction.py` from module 04 (section 5). The resulting `X_train`/`X_val` tensors have shape `(N, 30, 20)`.

---

## 3. v2 architecture — reference model

### 3.1 Description (`build_model_v2`)

```
Input (30, 20)
   → LSTM(128, return_sequences=True, unroll=True) + LayerNorm + Dropout(0.2)
   → LSTM(64,  return_sequences=True, unroll=True) + LayerNorm + Dropout(0.2)
   → MultiHeadAttention(4 heads, key_dim=16) [query=key=value=x] + residual connection + LayerNorm
   → GlobalAveragePooling1D
   → Dense(64, activation='elu') + Dropout(0.1)
   → Dense(1, activation='sigmoid')
```

**Total parameters: 147,073** (574.5 KB in fp32), confirmed by `model.summary()` and the saved `.keras` file (1.86 MB, including optimizer state and metadata).

### 3.2 Justification of architecture choices (notebook comments)

- **`unroll=True` on the LSTM layers**: explicitly unrolls the recurrent loop into a static graph rather than a dynamic loop — speeds up execution for short sequences (30 steps) and, above all, **facilitates later conversion to an embedded inference engine** (TensorFlow Lite / TFLite Micro), which handles the dynamic `tf.while_loop` loops underlying the non-unrolled mode poorly.
- **Multi-head self-attention block** (`MultiHeadAttention`, 4 heads × 16 dim = 64, matching exactly the output dimension of the second LSTM — no dimensional mismatch): allows the model to dynamically weight the importance of each cycle in the 30-step window — for example an isolated voltage drop, a thermal spike, or the inflection point of capacity fade — rather than relying solely on the sequential compression of the LSTM's final hidden state. The residual connection (`attn_out + x`) followed by a `LayerNormalization` reproduces the standard Transformer block pattern (pre/post-norm residual).
- **`GlobalAveragePooling1D`**: reduces the attention-weighted sequence of representations `(batch, 30, 64)` into a single vector `(batch, 64)` by averaging along the time axis — a choice explicitly justified in a comment as "cleaner than taking only the last time step", since it exploits information from **all** the steps already weighted by attention rather than relying solely on the last state.
- **ELU activation in the dense layer**: commented as preferable to ReLU since its derivative never vanishes for negative values ("no dead neurons"), which makes it easier for the network to correct SOH **overestimations** — a relevant choice for a regression task where the sign of the error matters.

### 3.3 Optimization and regularization

| Parameter | Value | Role |
|---|---|---|
| Optimizer | `AdamW` | Adam with decoupled weight decay |
| `learning_rate` | 3×10⁻⁴ | initial learning rate |
| `weight_decay` | 2×10⁻⁴ | decoupled L2 regularization |
| `clipnorm` | 1.0 | gradient norm clipping (anti-explosion protection, specifically added with the comment `# ← ADD THIS`) |
| Loss | `Huber(delta=0.01)` | robust to noisy cycles, quadratic/linear transition at 1% SOH error |
| Tracked metric | MAE | |
| `EarlyStopping` | `patience=25`, best-weights restoration | early stopping on `val_loss` |
| `ReduceLROnPlateau` | `factor=0.3`, `patience=5`, `min_lr=1e-6` | learning-rate reduction on stagnation |
| `batch_size` (training) | 128 | *(note: a `BATCH_SIZE=64` constant is defined in section 2 but is not the one actually used by `model.fit`, which receives a hard-coded `batch_size=128` — a minor inconsistency between the displayed configuration and the value actually applied, similar to the unused `SEQ_BUILD_BATCH_SIZE` documented in module 04)* |
| Max / actually run epochs | 400 max, early stop around epoch ≈ 90 (graphical reading of `loss_curves.png`) | |

### 3.4 Results obtained (validation set)

| Metric | Value |
|---|---|
| MAE | 0.112% SOH |
| RMSE | 0.130% SOH |
| MAPE | 2.56% |
| R² | 0.9936 |
| Explained variance | 0.9966 |

These results, read directly from the notebook's executed output, exceed the performance targets declared in module 04's `config.py` (`RMSE_TARGET=0.02`, `MAE_TARGET=0.015`, `R2_TARGET=0.95`) — the v2 model achieves an error roughly **15 to 20 times lower** than the acceptance thresholds set a priori. The real vs. predicted SOH curve (`soh_pred_vs_true.png`) visually confirms excellent tracking of the degradation trend, including in the extrapolation zone at the end of validation (SOH ≈ 0.70).

---

## 4. v3 architecture — model optimized for embedded use

### 4.1 Explicit hardware target

The docstring of `build_model_v3` mentions a specific target: **STM32F401RE microcontroller (94 KB of RAM, 512 KB of flash)**. This is the first mention, across all modules documented so far, of a **concrete embedded-inference hardware constraint** — consistent with the CubeSat mission mentioned since module 02, where a battery-health diagnostic model would run directly onboard (to be distinguished from training, carried out on the ground on Colab/GPU).

### 4.2 Description (`build_model_v3`)

```
Input (30, 20)
   → LSTM(64, return_sequences=True, unroll=True) + LayerNorm + Dropout(0.2)
   → LSTM(32, return_sequences=True, unroll=True) + LayerNorm + Dropout(0.2)
   → Additive attention (Bahdanau):
        Dense(16, tanh)  → score per time step
        Dense(1)         → a scalar score per step
        Softmax (time axis)  → normalized attention weights
        Multiply (x ⊙ weights) then sum over the time axis (custom `AttentionContext` layer)
     → context vector (batch, 32)
   → Dense(24, activation='elu') + Dropout(0.1)
   → Dense(1, activation='sigmoid')
```

**Total parameters: 35,730** (139.6 KB in fp32; ≈ 34.9 KB estimated after post-training int8 quantization), versus 147,073 for v2 — a **reduction by a factor of ≈ 4.1** in the number of parameters, consistent with the file-size ratio observed between the two saved `.keras` files (505 KB versus 1.86 MB).

### 4.3 Nature of the changes and technical justification (documented in the notebook)

| Change | v2 → v3 | Given justification |
|---|---|---|
| LSTM layer width | 128→64, 64→32 | 20 features over a 30-step window is a signal of modest dimensionality (voltage/temperature/capacity statistics); 128+64 units were sized for a much richer input, 64+32 retains the same two-layer temporal structure with enough margin for a monotonic degradation curve like SOH fade |
| Attention mechanism | `MultiHeadAttention` (4 heads, 64×64 Q/K/V/O projections) → additive attention (Bahdanau) | same purpose ("learn which cycles in the window matter most: a voltage drop, a thermal spike, the fade's inflection point") achieved with just two small scoring Dense layers rather than four learned projection matrices — also entirely replaces `GlobalAveragePooling1D`, since weighted attention already collapses the time axis via the weighted sum |
| Dense head | 64 → 24 | resized to stay consistent with the reduced upstream representation |

### 4.4 Details of the additive attention mechanism

A custom `AttentionContext` layer is declared and registered for Keras serialization:

```python
@keras.saving.register_keras_serializable(package="cubesat_soh")
class AttentionContext(layers.Layer):
    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=1)
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])
```

The full mechanism (Bahdanau-type attention, historically used in neural machine translation before the Transformer became widespread) works in three steps:
1. each time step of the LSTM output (32 dimensions) is projected by a small `Dense(16, tanh)` layer, then scored by a `Dense(1)` layer → a scalar score per cycle in the window;
2. the 30 scores are normalized by a `Softmax` along the time axis → attention weights summing to 1;
3. the LSTM output is element-wise multiplied by these weights and then summed over the time axis (`AttentionContext`) → a single 32-dimensional context vector, a learned weighted average of the 30 time steps.

This is functionally equivalent, in modeling power, to v2's multi-head self-attention (learned weighting of time steps), but at a much lower parameter cost: 2 scoring Dense layers (528 + 17 = 545 parameters) versus the four 64×64 projections of `MultiHeadAttention` (16,640 parameters in v2) — most of the model's size reduction comes from this substitution, combined with the reduced width of the LSTM layers.

The explicit registration via `@keras.saving.register_keras_serializable` is a **necessary and correctly applied precaution**: a custom Keras layer (a subclass of `layers.Layer`) must be registered for the model to be reloaded (`keras.models.load_model`) without having to pass the layer object back in via the `custom_objects` parameter — an important point of rigor for porting the `.keras` model to a deployment environment different from the training one.

### 4.5 Results obtained (validation set)

| Metric | v2 (reference) | v3 (optimized) | Gap |
|---|---|---|---|
| Parameters | 147,073 | 35,730 | ÷4.1 |
| fp32 size | 574.5 KB | 139.6 KB | ÷4.1 |
| MAE | 0.112% SOH | 0.128% SOH | +0.016 pt |
| RMSE | 0.130% SOH | 0.165% SOH | +0.035 pt |
| MAPE | 2.56% | 2.49% | ≈ stable |
| R² | 0.9936 | 0.9897 | −0.0039 |
| Explained variance | 0.9966 | 0.9901 | −0.0065 |

**Technical reading of the trade-off:** for a reduction of **more than 4 times** in the model's memory footprint (a decisive parameter for deployment on a 94 KB-RAM microcontroller), the performance degradation remains **marginal** — a few hundredths of a point on R² and an MAE delta on the order of 0.016 SOH percentage points, well below module 04's acceptance targets (`MAE_TARGET=1.5%`) for both versions. This is a result consistent with the nature of the signal being modeled: an overall monotonic, smooth degradation trend (see the module 03 EDA analysis) likely does not require the full representational capacity of a 64-dimension-projection multi-head self-attention block; a simpler attention mechanism is enough to capture most of the useful signal.

---

## 5. Technical issues identified in the evaluation code (common to both notebooks)

Sections 12 to 14 (prediction, evaluation, visualizations) are **identical between v2 and v3**. A careful reading of them, combined with examination of the figures actually produced during execution, reveals a reproducible and significant code defect.

### 5.1 Broadcasting bug in the error computation

```python
soh_pred  = model.predict(X_val, verbose=0).flatten().reshape(-1, 1)   # shape (N, 1)
soh_true  = y_val                                                       # shape (N,)
...
abs_err = np.abs(soh_pred - soh_true)
```

`soh_pred` is explicitly reshaped into a column `(N, 1)`, while `soh_true` remains a 1-D vector `(N,)`. When computing `soh_pred - soh_true`, **NumPy broadcasting** rules combine these two shapes not element-wise as intended, but into an **`(N, N)`** array containing, at position `(i, j)`, the difference `soh_pred[i] − soh_true[j]` for **every pair of indices `(i, j)`**, including pairs where `i ≠ j` — that is, differences between a prediction and a real value **from a completely different cycle**, with no physical meaning whatsoever.

**Direct empirical proof in the provided results:** the numerical evaluation (section 13, correctly computed via `sklearn` functions, which handle array shaping internally and are therefore unaffected by this bug) gives, for v2, a `Max Error = 0.003565` and a `Median Abs Error = 0.016351` — **the latter being nearly 5 times greater than the former**, which is **mathematically impossible** for the same set of errors (a median can never exceed a maximum). This is only possible if `medae` (computed directly in raw NumPy, outside `sklearn`) does indeed come from a contaminated `(N, N)` array containing cross-differences unrelated to the actual prediction error. The same pattern is observed for v3 (`Max Error = 0.004855` versus `Median Abs Error = 0.015909`).

The figure `abs_error.png` (section 14, "Absolute error over cycles" cell) makes this bug **directly visible**: instead of a single absolute-error curve per validation sequence (consistent with the displayed MAE of 0.0011), the figure shows a **dense band oscillating between 0.03 and 0.06**, unrelated to the MAE actually achieved by the model (the red dashed line is squashed near zero on the graph). This is technically explained by the fact that `plt.plot()`, called on a 2-D array of shape `(N, N)`, plots **one curve per column** of the array rather than a single series — hence the appearance of a thick band resulting from the superposition of hundreds of individual curves, each built from irrelevant cross-differences.

**Scope of the bug:** this defect affects three elements of the notebook, identical in both versions:
- the variable `residuals = soh_pred - soh_true` (section 14, scatter/residuals cell) — computed but never actually used for plotting (see §5.2);
- the variable `abs_err` and the resulting `abs_error.png` figure — a **misleading figure produced and saved as-is**;
- the `medae` ("Median Abs Error") statistic displayed in the evaluation summary and exported to `metrics.csv` — a **numerically incorrect value**.

**What is not affected, however**: the `MAE`, `RMSE`, `R²`, `MAPE`, `SMAPE`, and `Max Error` metrics, all computed via `sklearn.metrics` functions, which internally apply validation and consistent shaping of input arrays (`_check_reg_targets`) before any computation, and are therefore not subject to this broadcasting issue. The figures `soh_pred_vs_true.png` (plotting two simple series, one 1-D and one 2-D single-column — unaffected since `plt.plot` on an `(N,1)` array does correctly plot only a single curve) and `scatter_residuals.png` (a scatter plot, also unaffected by the shape of the arrays passed to `scatter`) therefore remain reliable.

**Recommended fix:** explicitly align the shapes before any raw NumPy operation, for example `soh_true = y_val.reshape(-1, 1)` right from section 12, or more simply `soh_pred = model.predict(X_val, verbose=0).flatten()` (without `reshape(-1,1)`) to keep both arrays 1-D throughout — the `sklearn` functions used for the official metrics would not be affected, and the raw NumPy computations (`medae`, `abs_err`, `residuals`) would become correct again.

### 5.2 Incomplete code: residuals subplot never plotted

In the cell producing `scatter_residuals.png` (section 14):

```python
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].scatter(soh_true, soh_pred, ...)
...
residuals = soh_pred - soh_true
# (no subsequent call to axes[1])
plt.savefig(...)
```

The variable `residuals` is computed but **never used** to populate the second subplot (`axes[1]`), which remains empty in the exported figure — confirmed by direct inspection of the image produced when the notebook was run. This is likely a residuals visualization (a histogram or scatter plot of the errors) left unfinished, to be completed (and simultaneously fixed for the shape bug described in 5.1, since `residuals` would directly inherit it).

### 5.3 Documentation inconsistency: BatchNorm vs. LayerNormalization

The markdown cell introducing the architecture (section 10) states: *"Dropout → BatchNorm in each block"*. The code actually implements a **`LayerNormalization`** after each LSTM block (not a `BatchNormalization`) — two normalization techniques that differ in principle (normalization over the batch, for instance, vs. per-feature normalization within a single example), with `LayerNormalization` actually being the more suitable choice for recurrent networks (behavior independent of batch size, consistent with `batch_size=1` inference typical of a real-time embedded deployment). This is therefore only a **comment inaccuracy**, with no consequence for the model actually trained, but one that should be corrected to avoid any confusion on a future review.

### 5.4 Documented but accepted inconsistency: sigmoid vs. linear output activation

A detailed comment within `build_model_v2` itself (reproduced identically in `build_model_v3`) argues at length in favor of a **`linear`** output rather than `sigmoid`:

> *The `sigmoid` gradient at SOH=1.0 is only about 0.09. The network learns very slowly in the [0.90–1.00] zone it observes most often early in training → the prediction never fully corrects itself → systematic upward bias [...] The `linear` gradient is 1.0 everywhere → identical correction signal whether SOH is at 0.70 or 1.00. The physical bound [0, 1] would then be enforced by an `np.clip()` at inference, not by the activation function — which preserves the gradient during training while still respecting the physics.*

Yet the very next line of code nevertheless builds the output layer with `Dense(1, activation='sigmoid', name='SOH_output')` — confirmed both in the source code of both notebooks and in the internal `config.json` of both saved `.keras` artifacts (`"activation": "sigmoid"`). This inconsistency, already spotted and transparently documented in module 04's `README.md` (which explains that `config.OUTPUT_ACTIVATION="sigmoid"` was chosen to stay faithful to the code actually executed), is thus indeed confirmed by a direct reading of the source code that is the origin of this observation.

**Additional technical analysis, in light of the results obtained:** despite the theoretical argument in favor of `linear`, the empirical results of the actually trained and evaluated `sigmoid` model show **no visible systematic bias** on the `soh_pred_vs_true.png` figure (the predicted curve closely tracks the real curve over the entire [0.70–0.76] range observed in validation, with no visible upward drift). This can be explained by the fact that the SOH range actually covered by the validation sequences in this dataset (0.70 to about 0.76, see §2.3) remains **relatively far from the critical saturation zone (SOH close to 1.0)** identified in the comment — the theoretical problem documented could therefore become more visible on a validation set covering a wider SOH range including early life, which remains to be tested (an A/B test of `sigmoid` vs. `linear`, already recommended in module 04's documentation).

---

## 6. Saving artifacts (section 15)

Each notebook exports, into `OUTPUT_DIR`:

| File | Content |
|---|---|
| `cubesat_soh_lstm_unrolled_test.keras` (v2) / `cubesat_soh_lstm_v3_unroll_True.keras` (v3) | full Keras model (architecture + weights + compilation configuration) |
| `scaler_X.pkl` | `StandardScaler` fit on the train set, needed to normalize any new input data identically at inference time |
| `features_used.pkl` | ordered list of feature names actually used, guaranteeing column-by-column correspondence between training and inference |
| `metrics.csv` | a summary row of all evaluation metrics (including the erroneous `medae` metric, see §5.1 — **to be corrected before using this file as a reporting reference**) |
| `loss_curves.png`, `soh_pred_vs_true.png`, `scatter_residuals.png`, `abs_error.png` | diagnostic figures (the last one being affected by the bug described in §5.1) |

A commented-out line (`#joblib.dump(bias, ...)`) suggests that a **post-hoc linear-regression calibration** step (section 11 bis, cell 29/30: a `LinearRegression` fit on the training predictions to correct a systematic scale/offset bias) was explored and then disabled in the final version of both notebooks — consistent with the fact that the bias this calibration targeted (the `sigmoid` gradient saturation, see §5.4) no longer appears to be a dominant issue once `scaler_y` was removed (see §2.4), making this after-the-fact correction step superfluous in its current state.

---

## 7. v2 / v3 comparative summary

| Criterion | v2 (reference) | v3 (embedded-optimized) |
|---|---|---|
| Recurrent layers | LSTM 128 → LSTM 64 | LSTM 64 → LSTM 32 |
| Attention mechanism | Multi-head self-attention (4×16) | Additive attention (Bahdanau), custom layer |
| Temporal aggregation | `GlobalAveragePooling1D` (separate) | integrated into the attention mechanism |
| Output head | Dense 64 → Dense 1 | Dense 24 → Dense 1 |
| Parameters | 147,073 | 35,730 (÷4.1) |
| fp32 size | 574.5 KB | 139.6 KB |
| Estimated int8 size | — (not quantized) | ≈ 34.9 KB |
| Deployment target | not specified (methodological reference) | STM32F401RE (94 KB RAM / 512 KB flash) |
| Validation MAE | 0.112% SOH | 0.128% SOH |
| Validation R² | 0.9936 | 0.9897 |
| Error-computation bug (§5.1) | present | present (inherited identically) |
| sigmoid/linear inconsistency (§5.4) | present | present (inherited identically) |

---

## 8. Limitations and recommendations

1. **NumPy broadcasting bug (§5.1)** — high priority: fixes the `abs_error.png` figure and the `medae` metric exported in `metrics.csv`, which are currently misleading in both notebooks. A simple shape normalization (`.flatten()` on both arrays before any raw operation) is sufficient.
2. **Incomplete residuals subplot (§5.2)** — to be finalized or removed to avoid a partially empty figure in the deliverables.
3. **Unresolved sigmoid/linear inconsistency (§5.4)** — a controlled A/B test (same seeds, same split, with only the output activation changing) would objectively verify whether the theoretical argument documented in the code translates into a measurable gain, particularly over an SOH range including early life (close to 1.0), where the gradient-saturation problem is expected to be most pronounced.
4. **Data pipeline duplication with module 04 (§1.1)** — risk of drift between two parallel implementations; consider having the notebook directly consume the `.npy`/`.csv` artifacts produced by module 04 rather than reimplementing `engineer_features`, the split, and normalization internally.
5. **`BATCH_SIZE` inconsistent between the displayed configuration (64) and the value actually used by `model.fit` (128)** (§3.3) — to be unified to avoid confusion during a future review or resumption of experimentation.
6. **BatchNorm/LayerNorm comment to be corrected** (§5.3) — minor documentation inaccuracy.
7. **`SOH_lag_1`**, if present among the `FEATURE_COLS` loaded from `features.pkl` (not directly verifiable in this archive, as the `.pkl` file is not provided), inherits the same caveat already documented in module 04: this autoregressive feature assumes the availability of the real SOH from the previous cycle, not guaranteed in real embedded inference — the recursive-feedback strategy (using the model's own prediction as the next step's `SOH_lag_1`) and its cumulative-drift risk remain to be specified and tested before deployment.
8. **int8 quantization not yet performed**: v3's "after quantization" size (≈ 34.9 KB) is an **estimate** (`total_params × 1 byte`, commented `est.` in the code), not a measurement on a model actually quantized and exported to TensorFlow Lite Micro format — a real post-training quantization step, followed by a new evaluation of the error induced by quantization, remains necessary before any actual deployment on the STM32F401RE.

---

## 9. Conclusion

This module documents a rigorous, well-tracked, two-stage model engineering process: a **reference architecture (v2)**, combining stacked LSTMs and multi-head self-attention, achieving high accuracy (MAE 0.11% SOH, R² 0.994) well above the acceptance targets set earlier in module 04; then an **embedded-optimized architecture (v3)**, reducing the parameter count by a factor greater than 4 by replacing multi-head attention with lighter additive attention and reducing the width of the recurrent layers, at a marginal accuracy cost — a well-justified and quantified technical trade-off, explicitly motivated by a concrete hardware target (STM32F401RE). Detailed examination of the evaluation code, cross-checked against the figures and metrics actually produced during execution, made it possible to identify a concrete, reproducible computation defect (a NumPy broadcasting bug affecting the absolute-error figure and the median metric), as well as several minor documentation inconsistencies — all concrete points to fix before using these notebooks as a final performance-reporting reference or as a starting point for the quantization and embedded-deployment stage.
