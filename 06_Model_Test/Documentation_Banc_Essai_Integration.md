# Technical Documentation — Real-Time Integration Test Bench (Module 06)

**Document scope:** `06_Model_Test`
**Subfolders analyzed:** `test_bench_essay_Reference_Model/` (model v2), `test_bench_essay_optimized_model/` (model v3)
**Files analyzed:** `cubesat_eps_main_fixed.m`, `compute_features.m`, `lstm_core_5.py` / `lstm_core_5_test.py`, `cubesat_params.m`, `cubesat_ode_ch.m`, `cubesat_ode_dch.m`, `cubesat_events.m`, `simulate_cycle.m`, `Reference_model.fig`, `the_last_optimized_model.fig`

This document complements modules 02 (physical simulation), 04 (data preparation pipeline) and 05 (LSTM v2/v3 models). It documents the **integration test bench** that makes the MATLAB physical simulation and the trained Python inference model interact, cycle by cycle — an end-to-end "Software-in-the-Loop" style test, designed to validate the SOH model's behavior under conditions close to its final embedded usage, rather than on a pre-computed static validation set.

---

## 1. Purpose and principle of the test bench

Unlike the module 05 evaluation (performed offline, on sequences already built and normalized from an existing CSV), this test bench **simulates and predicts at the same time, cycle after cycle**:

```
MATLAB loop (cubesat_eps_main_fixed.m), for each orbital cycle:
   1. simulate_cycle()      → integrates the cycle's physics (discharge + charge), see module 02
   2. compute_features()    → computes the 20 features from the accumulated history
   3. Python call (lstm_core_5.CubeSatSOHPredict.predict_soh) → real-time SOH prediction
   4. comparison SOH_true (simulation) vs SOH_lstm (model) → logging + plotting
```

This scheme faithfully reproduces the intended onboard use case for the CubeSat: at each orbit, a set of measurements is available, features are computed on the fly, and the model produces an SOH estimate without ever having access to the future — only the sliding window of the last 30 cycles feeds the prediction. This is therefore an integration test much closer to real operating conditions than the batch evaluation in module 05.

Two strictly parallel test benches are provided, one per model version (v2 "Reference_Model" and v3 "optimized_model"), allowing a direct comparison of their behavior under sequential inference conditions.

---

## 2. Reuse of the physical model (recap)

The files `cubesat_params.m`, `cubesat_ode_ch.m`, `cubesat_ode_dch.m`, `cubesat_events.m` and `simulate_cycle.m` present in both subfolders are, after direct comparison, **byte-for-byte identical** to those documented in module 02. The test bench's physical engine is therefore rigorously the same as the one used to generate the training dataset — no physical model divergence is to be reported here. It is not re-described in this document; refer to module 02 for the equation details (OCV, Arrhenius internal resistance, thermal balance, aging).

Likewise, the files `compute_features.m` and `cubesat_eps_main_fixed.m` are **identical between the two subfolders**: only the Python implementation of the inference bridge differs (`lstm_core_5.py` vs `lstm_core_5_test.py`), and of course the `.keras` file loaded.

---

## 3. Real-time feature computation in MATLAB (`compute_features.m`)

### 3.1 Principle

This function reconstructs, **in MATLAB and cycle by cycle**, the same 20 features defined in `feature_engineering.py` (module 04) and `engineer_features()` (module 05) — but in a form adapted to incremental computation rather than vectorized processing of a full DataFrame:

```matlab
idx_start = max(1, cycle - W + 1);   % W = 10, must match the training ROLLING_W
idx_range = idx_start:cycle;
feat.V_mean_rolling = mean(history.V_mean_V(idx_range));
```

Each rolling quantity (`V_mean_rolling`, `Tavg_rolling`, `QD_rolling`) is recomputed from a **history accumulated throughout the loop** (`history.*`), rather than from a full pandas `DataFrame` as in modules 04/05 — a direct and faithful translation of the `rolling(W, min_periods=1).mean()` logic into a MATLAB window bounded by `max(1, cycle-W+1)`, equivalent to pandas' `min_periods=1` behavior (window truncated at the start of the series rather than `NaN`).

All the feature formulas (coulombic efficiency `QC_Ah/QD_Ah`, `discharge_C_rate = DoD/(discharge_time_min/60)`, `cold_cycle_count` on the `Tavg_C < 5°C` threshold, `delta_T_ambient` as a difference of `T_amb_K`, etc.) are found to be **identical** to those validated in modules 04 and 05, confirming that alignment between the three implementations (Python scripts of module 04, notebook of module 05, MATLAB of module 06) has indeed been maintained across the project's three layers.

A `% ✅ FIX ADDED` comment on the `feat.cycle = cycle;` line indicates that a correction was made to an earlier version of this function, in which the `cycle` feature was probably missing — consistent with the correction-tracing style already observed in the README files of previous modules.

### 3.2 Critical point: `SOH_lag_1` computed from ground truth

The code explicitly flags this point with an unambiguous comment:

```matlab
%% =========================
%% SOH lag (GROUND TRUTH ONLY)
%% =========================
if isfield(history, 'SOH')
    if cycle > 1
        feat.SOH_lag_1 = history.SOH(cycle-1);
    else
        feat.SOH_lag_1 = 1.0;
    end
```

And, redundantly, the main loop (`cubesat_eps_main_fixed.m`) **overwrites this value again** right after the call to `compute_features()`:

```matlab
%% 🔥 TRUE SOH lag
if cycle == 1
    feat_struct.SOH_lag_1 = 1.0;
else
    feat_struct.SOH_lag_1 = SOH_true(cycle-1);
end
```

This confirms, directly and without any possible ambiguity, the concern already anticipated in modules 04 and 05: the autoregressive feature `SOH_lag_1` is fed here with the **real SOH from the physical simulation** (`SOH_true`, computed by `simulate_cycle.m`), and not with the model's own prediction from the previous cycle (`SOH_lstm`). This test bench therefore tests the model in a **semi-guided (teacher forcing)** regime with respect to this specific feature: the model receives, at every cycle, reliable aging information with no accumulated error, whereas a real embedded deployment would never have the real SOH available as input (it is precisely the quantity the system is trying to estimate). **The performance results measured in this test bench (§6) are therefore optimistic compared to a real autonomous deployment**, where `SOH_lag_1` would need to be fed by the model's own output from the previous step, with the associated risk of cumulative drift — a complementary test with recursive feedback (`SOH_lag_1 = SOH_lstm(cycle-1)`) would be needed to characterize the behavior actually expected onboard.

---

## 4. MATLAB ↔ Python inference bridge (`lstm_core_5.py`)

### 4.1 Initializing the Python environment from MATLAB

```matlab
pyenv('Version', 'C:\Users\...\tf_env\Scripts\python.exe', 'ExecutionMode','OutOfProcess');
module = py.importlib.import_module('lstm_core_5');
py.importlib.reload(module);
py_predictor = module.CubeSatSOHPredict();
```

The explicit choice of `'ExecutionMode','OutOfProcess'` is a good engineering practice: it runs the Python interpreter in a **separate process** rather than in-process (the default), which avoids the classic and well-documented conflicts between native libraries loaded by MATLAB and those loaded by TensorFlow (particularly C++/MKL runtime version conflicts), a frequent source of crashes when TensorFlow is driven from MATLAB. The path to the Python executable is, however, **hard-coded** (local Windows path `C:\Users\Amine\Desktop\...`), which makes the script non-portable as-is — it should be externalized into a configuration variable or dynamically detected for reuse on another machine.

### 4.2 `CubeSatSOHPredict` class: sliding buffer management

```python
self.seq_len = 30
self.buffer = []

def predict_soh(self, feature_dict):
    ...
    feat_scaled = self.scaler_X.transform(df)[0]
    self.buffer.append(feat_scaled)
    if len(self.buffer) > self.seq_len:
        self.buffer.pop(0)
    if len(self.buffer) < self.seq_len:
        return None
    X_seq = np.array(self.buffer, dtype=np.float32).reshape(1, self.seq_len, -1)
    pred = float(self.model.predict(X_seq, verbose=0)[0][0])
    return np.clip(pred, 0.0, 1.05)
```

This class implements a **stateful sliding window (FIFO buffer)** of 30 cycles — the equivalent, in real-time sequential inference, of the `build_sequences()` function from modules 04/05 used in batch processing. Technical points to note:

- **Consistent normalization**: each new feature vector is transformed by the `scaler_X` loaded from `scaler_X.pkl` (the same artifact saved by `normalize.py` in module 04 or by the module 05 notebook), guaranteeing that the normalization statistics applied at inference are indeed those learned during training.
- **Returns `None` during the buffer fill-up phase** (the first 29 cycles): correct and explicit behavior, corresponding to the physical impossibility of a prediction before a full 30-cycle window has been accumulated — reproduced in the MATLAB loop by a `SOH_lstm(cycle) = NaN` as long as no prediction is available.
- **Output clipping `np.clip(pred, 0.0, 1.05)`**: bounds the prediction to a physically plausible interval, with a 5-point tolerance above 1.0 — likely to absorb a slight transient overshoot from the model very early in life without treating it as an anomaly. This bound is applied **in addition to** the `sigmoid` activation already bounding the network's output (see module 05, §5.4); it therefore only has a real effect on the upper bound at 1.05 (the lower bound at 0.0 is already structurally guaranteed by the sigmoid, which cannot produce a negative value).

### 4.3 Difference between the two versions of the inference bridge

The version used in the "optimized_model" test bench (`lstm_core_5_test.py`) adds two elements absent from the "Reference_Model" version (`lstm_core_5.py`):

1. **Declaration and registration of the custom `AttentionContext` layer** before loading the model:
   ```python
   @keras.saving.register_keras_serializable(package="cubesat_soh")
   class AttentionContext(layers.Layer):
       def call(self, inputs): return tf.reduce_sum(inputs, axis=1)
       def compute_output_shape(self, input_shape): return (input_shape[0], input_shape[2])
   ```
   Necessary because the v3 model uses this layer for its additive attention block (see module 05, §4.4) — a custom Keras architecture cannot be reloaded unless its class is known to the loading environment. The code comment highlights an extra precaution ("*belt-and-suspenders*"): the layer is **explicitly passed as `custom_objects`** to `load_model()`, in addition to being registered via the decorator, because some host environments (including the embedded Python interpreter driven by MATLAB, explicitly cited in the comment) do not reliably execute the `@register_keras_serializable` decorator before the call to `load_model()`. This is a concrete and well-anticipated example of a MATLAB/Python/Keras interoperability issue, resolved robustly (double guarantee rather than simple trust in the automatic registration mechanism).
2. **A standalone `if __name__ == "__main__"` test block**, which instantiates the predictor and feeds it synthetic random feature rows (`rng.uniform(0.0, 1.0)`) to verify that the inference pipeline works end-to-end independently of MATLAB — a minimal but useful unit test for isolated debugging of the Python component.

### 4.4 Naming inconsistency identified between the "optimized_model" folder and the main script

The `cubesat_eps_main_fixed.m` script in the `optimized_model` folder (identical to the one in the `Reference_Model` folder, see §2) still contains the line:

```matlab
module = py.importlib.import_module('lstm_core_5');
```

Yet the only Python file present in this folder is named **`lstm_core_5_test.py`**, not `lstm_core_5.py`. As-is, running the "optimized_model" test bench **would fail** with a `ModuleNotFoundError` on the Python side, since the `lstm_core_5` module does not exist in this directory. For this test bench to work, either the file must be renamed to `lstm_core_5.py` before execution, or the import line in `cubesat_eps_main_fixed.m` must be changed to `'lstm_core_5_test'`. The numerical results presented in §6 (extracted from the `.fig` figures already produced) show that this run **did indeed take place successfully at some point** (probably after a local rename not reflected in the delivered archive) — but the archive as provided would not allow reproducing it without this manual fix.

### 4.5 Required artifacts missing from the archive

`CubeSatSOHPredict.__init__` requires by default two files, `scaler_X.pkl` and `features_used.pkl`, produced by module 05 (section 15 of the notebooks) — **neither file is present** in the provided `06_Model_Test` folders. The test bench therefore cannot be re-run as-is without first copying these two artifacts from module 05's outputs (one for each model version, v2 and v3, since each notebook produces its own `scaler_X.pkl`).

---

## 5. Main loop (`cubesat_eps_main_fixed.m`)

### 5.1 Reuse of module 02's orbital variability logic

```matlab
rng(42);  % match cubesat_run.m's reproducible seeding
...
eclipse_min  = max(32.0, min(38.0, 35.0 + 1.0 * randn()));
sunlight_min = max(53.0, min(57.0, 55.0 + 0.8 * randn()));
if mod(cycle, 2) == 1
    p_cycle.T_amb_eclipse = p.T_operating_min;
else
    p_cycle.T_amb_eclipse = p.T_operating_min + 8.0;
end
```

This section reproduces **identically** the cycle-to-cycle orbital variability logic from module 02 (`cubesat_run.m`), including the random seed `rng(42)` — explicitly commented "*match cubesat_run.m's reproducible seeding*".

**Important methodological point to highlight:** `cubesat_run.m` (module 02, with `n_batt=1`) used `rng(batt_idx * 42) = rng(42)` to generate the `battery_dataset_normal.csv` dataset used to train and validate the module 05 models. By reusing **exactly the same seed** `rng(42)`, this test bench regenerates a sequence of random draws (`eclipse_min`, `sunlight_min`) **rigorously identical** to the one already used to produce the training data — so, strictly speaking, this is not a test on an **independent, unseen** orbital scenario, but a **re-simulation of a trajectory already known to the model** (through its training/validation history). An integration test intended to genuinely represent generalization under new operational conditions would benefit from using a different seed (or several seeds, in a Monte Carlo approach) than the one used for training — a point to correct for any future validation campaign presented as a generalization test.

### 5.2 Building the Python dictionary and calling the prediction

```matlab
py_dict = py.dict();
for i = 1:length(ordered_features)
    key = ordered_features{i};
    val = feat_struct.(key);
    if isnan(val) || isinf(val)
        val = 0.0;
    end
    py_dict{key} = val;
end
soh_pred = py_predictor.predict_soh(py_dict);
```

The `MATLAB struct → py.dict` conversion is performed field by field, with a robustness safeguard (`NaN`/`Inf` replaced with `0.0`) before being passed to the Python model — a useful protection against a potentially undefined value very early in the simulation (for example a ratio involving division by a zero quantity at cycle 1, already partly handled by the `max(..., 1e-6)` calls in `compute_features.m`, but secured a second time here).

The `try/catch` block around the Python call is **commented out (disabled)** in the delivered code: any exception raised on the Python side would therefore directly interrupt MATLAB execution rather than being caught and logged as a warning — a choice probably made deliberately during development ("*fail fast*" to immediately detect an issue), but one that should be re-enabled for more resilient execution outside a debugging context.

### 5.3 Stopping criterion and produced outputs

```matlab
if SOH_true(cycle) < 0.70
    fprintf('\n🔴 EOL reached at cycle %d | SOH = %.4f\n', cycle, SOH_true(cycle));
    break;
end
```

The same end-of-life threshold as module 02 (`SOH_eol = 0.70`) is used to stop the loop. The script exports:

- `CubeSat_REALTIME_LSTM_FINAL.csv`: `cycle`, `SOH_true`, `SOH_lstm` — direct cycle-by-cycle comparison;
- `CubeSat_REALTIME_FEATURE_LOG.csv`: the full set of 20 features computed at each cycle, explicitly intended (per the code comment) to be compared against the statistics of the features used at training time (`raw_features.csv`, mentioned in module 05) — good practice for detecting distribution drift (*data drift*) between training and inference;
- a MATLAB figure (`Reference_model.fig` / `the_last_optimized_model.fig`) plotting `SOH_true` and `SOH_lstm` overlaid, with a horizontal reference line at the EOL threshold (70%).

None of these three CSV output files, however, are included in the provided archive — only the final `.fig` figures were delivered; the quantitative analysis in the following section was therefore reconstructed directly from the numerical data (`XData`/`YData`) embedded in these figures.

---

## 6. Test bench results (reconstructed from the `.fig` figures)

Since `.fig` files are in MATLAB's proprietary format ("Handle Graphics", based on a MAT-file container), their curves were extracted directly from their internal structure to allow a quantitative analysis.

### 6.1 Curve coverage

| Model | Cycles covered (real SOH) | Cycles covered (predicted SOH) | End of life reached |
|---|---|---|---|
| v2 (Reference) | 1 → 6,808 | 30 → 6,808 (6,779 predictions) | cycle 6,808, SOH ≈ 0.700 |
| v3 (Optimized) | 1 → 6,808 | 30 → 6,808 (6,779 predictions) | cycle 6,808, SOH ≈ 0.700 |

Both test benches reach EOL at exactly the **same cycle** (6,808) with the **same real SOH trajectory** — direct confirmation that the physical part of the simulation (identical between the two folders, see §2, and using the same `rng(42)` seed) produces a strictly reproducible trajectory, consistent with the lifetime value already documented in module 02 (≈ 6,800 cycles for the "normal" scenario).

### 6.2 Overall prediction error (cycles 30 to 6,808)

| Metric | v2 (Reference) | v3 (Optimized) |
|---|---|---|
| MAE | 0.421% SOH | 0.438% SOH |
| RMSE | 0.527% SOH | 0.628% SOH |
| Maximum error | 1.975% SOH | 3.504% SOH |

**Comparison with module 05's offline evaluation:** these MAE values (0.42–0.44%) are **about 3.5 to 4 times higher** than those measured in module 05's batch evaluation (0.112% for v2, 0.128% for v3). This gap, although the model remains overall accurate (error under 0.5 SOH points on average), is consistent with a change in evaluation regime: module 05's evaluation measured the model's ability to predict correctly over a **single contiguous time segment** (the last 20% of cycles in the dataset), whereas this test bench confronts it with the **entire life trajectory**, including the sliding-buffer startup phase and the very early life period (SOH close to 1.0) — a zone not covered at all by module 05's evaluation, since its validation set only covered the SOH range [0.70, 0.76] (see module 05, §2.3). It is precisely in this early-life zone that the test bench's error turns out to be highest (see §6.3), which accounts for most of the overall gap.

The noticeably higher maximum error for v3 (3.50% vs. 1.98% for v2) is consistent with the reduced representational capacity of the optimized model (35,730 parameters vs. 147,073, see module 05): the v3 model remains overall accurate on average, but shows more pronounced occasional deviations — an expected trade-off for a model reduced by a factor of 4 for embedded deployment.

### 6.3 Error by life phase — empirical confirmation of the bias theorized in module 05

| Life phase | v2 — MAE (mean bias) | v3 — MAE (mean bias) |
|---|---|---|
| Early life (cycles 30–500, SOH ≈ 0.95–1.00) | 0.595% (**−0.595%**) | 1.865% (**−1.865%**) |
| Mid-life (cycles 3000–3500, SOH ≈ 0.85) | 0.539% (+0.539%) | 0.446% (−0.446%) |
| End of life (cycles 6300–6808, SOH ≈ 0.70–0.71) | 0.090% (−0.061%) | 0.177% (+0.158%) |

**This breakdown by life phase provides direct empirical confirmation** of the theoretical concern documented in module 05's source code (§5.4 of the module 05 documentation): the comment in `build_model_v2` anticipated that the `sigmoid` output, whose gradient becomes very small near SOH = 1.0, would prevent the model from fully correcting its prediction errors in that zone. The results measured here confirm this unambiguously:

- the error (and the systematic bias, with a negative sign — underestimation of the real SOH) is **highest very early in life**, precisely the sigmoid's saturation zone, for both models;
- it decreases sharply in mid-life, and becomes **very small at end of life** (0.09% for v2, a remarkable accuracy), a zone where SOH moves away from saturation at 1.0 and where the sigmoid gradient again becomes favorable to learning;
- the v3 model (reduced architecture) is **nearly 3 times more affected than v2** by this early-life bias (−1.87% vs. −0.60%), suggesting that the optimized model's reduced representational capacity makes it more sensitive to this learning difficulty specific to the saturation zone.

This is a valuable and directly actionable engineering observation: it confirms, through a measurement independent of module 05's evaluations (which did not cover this SOH range), that replacing the `sigmoid` output activation with `linear` — already recommended in module 05's source code but not applied in the trained version — **would likely correct a significant portion of the error observed very early in life**, the phase where the gap is currently most pronounced for both architectures.

---

## 7. Summary of limitations and recommendations

1. **Random seed identical to training (§5.1)** — high priority for any generalization conclusion: `rng(42)` exactly reproduces the orbital trajectory already seen during module 05's training/validation. This test bench therefore mainly validates **integration consistency** (the MATLAB→Python pipeline works correctly end-to-end, feature by feature) rather than **generalization capability** to a new orbital scenario. A run with a different seed (or several, in a Monte Carlo approach) is needed to assess the model's real robustness against unseen orbital variability.
2. **`SOH_lag_1` fed by ground truth (§3.2)** — high priority: the performance measured in §6 is optimistic compared to a real autonomous embedded deployment, where this feature should be filled in by the model's own recursive prediction. A second closed-loop test bench (`SOH_lag_1 = SOH_lstm(cycle-1)`) would allow quantifying the risk of cumulative drift under genuinely autonomous conditions.
3. **Python module naming inconsistency in the `optimized_model` folder (§4.4)** — blocking for execution: rename `lstm_core_5_test.py` to `lstm_core_5.py`, or adapt the `import_module` line in the MATLAB script.
4. **Missing artifacts (`scaler_X.pkl`, `features_used.pkl`, §4.5)** — blocking for execution: to be copied from module 05's outputs (one set per model version) before any new run of the test bench.
5. **Hard-coded Python path (§4.1)** — blocking on any machine other than the original one; should be externalized into a configuration variable or script argument.
6. **Error handling disabled (§5.2)** — the `try/catch` around the Python call is commented out; should be re-enabled outside a debugging context for robust execution under automated test conditions.
7. **CSV output files not provided in the archive** — the quantitative analysis in this document had to be reconstructed from the `.fig` figures; systematically keeping the `CubeSat_REALTIME_LSTM_FINAL.csv` and `CubeSat_REALTIME_FEATURE_LOG.csv` files from each future run would greatly facilitate later analyses (and would avoid dependency on a proprietary file format that is difficult to work with outside MATLAB).
8. **Actionable confirmation of the sigmoid bias (§6.3)** — direct recommendation: test the `linear` output variant (already present as a comment in module 05's code) on this same real-time test bench, particularly over the early-life range (cycles 30–500), to check whether it actually reduces the negative bias observed in that zone for both architectures.

---

## 8. Conclusion

This test bench constitutes a valuable end-to-end integration test, going beyond module 05's static evaluation by replaying an entire simulated battery life with real-time, cycle-by-cycle SOH prediction, via a robust MATLAB↔Python inference bridge (out-of-process Python execution mode, explicit handling of custom layer serialization). The results extracted directly from the produced figures confirm that both models (v2 reference and v3 optimized/embedded) remain accurate over an entire battery life (overall MAE under 0.5 SOH points), and provide a **particularly useful, independent empirical confirmation** of the sigmoid output saturation bias in early life, already anticipated — but not corrected — in module 05's source code. Two important methodological caveats, however, limit the scope of the generalization conclusions that can be drawn from this test bench as it stands: the reuse of the training random seed (the orbital scenario is not truly unseen) and the feeding of the `SOH_lag_1` feature with ground truth rather than the model's own recursive prediction (the test does not yet cover the fully autonomous inference regime targeted for a real embedded deployment). These two points, together with the few operational obstacles identified (module naming, missing artifacts, hard-coded path), form the natural roadmap for evolving this test bench into a full generalization validation campaign before deployment.
