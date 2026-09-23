# Technical Documentation — Embedded Inference via TensorFlow Lite (Module 08)

**Document scope:** `08_inference_tflite`
**Files analyzed:** `lstm_core_6_tflite.py`, `cubesat_eps_main_fixed.m`, `compute_features.m`, `cubesat_params.m`, `cubesat_ode_ch.m`, `cubesat_ode_dch.m`, `cubesat_events.m`, `simulate_cycle.m`, `cubesat_soh_lstm.keras`, `inference.jpg`

This document complements modules 02 (physical simulation), 05 (LSTM v2/v3 models) and 06 (MATLAB↔Keras integration test bench). It documents the **switch of the inference engine from native Keras to TensorFlow Lite**, a decisive technical step toward any actual deployment of the SOH model on a real embedded target (microcontroller), since the `.keras` format is not executable on that type of platform.

---

## 1. Purpose and place in the processing chain

Module 06 validated the model's behavior under real-time simulation conditions, but relied on `tf.keras.models.load_model()` — an API that requires the full TensorFlow runtime (several hundred megabytes), entirely incompatible with the memory constraints of a microcontroller such as the STM32F401RE targeted by the v3 model (see module 05, §4.1: 94 KB of RAM, 512 KB of flash). The **TensorFlow Lite (`.tflite`)** format, together with its lightweight interpreter (`tf.lite.Interpreter`, and eventually its *TFLite Micro* variant for microcontrollers), is the standard conversion step for reducing a Keras model to a format executable on a target with very limited resources.

This module therefore reuses **exactly the same test bench as module 06** (identical physics, identical simulation loop, identical feature computation), substituting only the inference component: `lstm_core_5.py` (Keras loading) is replaced with `lstm_core_6_tflite.py` (loading via the TFLite interpreter).

```
Module 06: MATLAB ──py.dict──► lstm_core_5.py       ──► tf.keras.models.load_model(.keras)
Module 08: MATLAB ──py.dict──► lstm_core_6_tflite.py ──► tf.lite.Interpreter(.tflite)
```

---

## 2. Full reuse of the physical test bench

Direct comparison (`diff`) with the module 06 files: `cubesat_params.m`, `cubesat_ode_ch.m`, `cubesat_ode_dch.m`, `cubesat_events.m`, `simulate_cycle.m`, `compute_features.m` and **`cubesat_eps_main_fixed.m` are all strictly identical**, character for character, to their module 06 counterparts. The physical module and the feature-computation logic are therefore not re-described here; see modules 02 and 06 for their full documentation (physical equations, feature formulas, sliding buffer management, etc.).

This strict code identity means in particular that the main loop still uses **`rng(42)`**, the same seed already documented as problematic in module 06 (§5.1 of that documentation) — the methodological caveat already raised there (this is not a test on a truly unseen orbital scenario, but a re-simulation of an already-known trajectory) therefore **also applies** to this module.

---

## 3. The new TensorFlow Lite inference bridge (`lstm_core_6_tflite.py`)

### 3.1 Change of loading API

```python
self.interpreter = tf.lite.Interpreter(model_path=model_path)
self.interpreter.allocate_tensors()
self.interpreter.reset_all_variables()

self.input_details  = self.interpreter.get_input_details()
self.output_details = self.interpreter.get_output_details()
```

The code comment is explicit about the reason for this change: *"Load the TFLite model via the Interpreter (NOT `tf.keras.models.load_model`, which only understands SavedModel/H5, not the `.tflite` FlatBuffer)"*. This is a fundamental technical distinction: a `.tflite` file is a **FlatBuffer** binary format, a flat, compact serialization optimized for fast loading and a minimal memory footprint at runtime — structurally different from the `.keras` format (a ZIP archive containing a JSON configuration + HDF5 weights, see module 05, §4.4), and it necessarily requires the dedicated `tf.lite.Interpreter` API to be used.

`allocate_tensors()` reserves the memory needed to run the graph only once at initialization (rather than at every inference), a standard practice for minimizing repeated inference latency in real-time use. The call to `reset_all_variables()` guarantees a clean internal model state before the first prediction (relevant if the graph contains internal state variables, which can be the case for a converted recurrent layer).

### 3.2 Dynamic reading of the input shape — a notable robustness improvement

```python
in_shape = self.input_details[0]['shape']
self.seq_len     = int(in_shape[1])
self.n_features  = int(in_shape[2])
```

Unlike `lstm_core_5.py` (module 06), where `self.seq_len = 30` was **hard-coded**, this version **directly queries the loaded model** to find out the input shape it expects (`(1, 30, 20)` according to the code comment). This is a notable robustness improvement: the script automatically adapts to whatever `.tflite` model is provided as input (for example a future v4 version with a different window), with no need to modify the Python code — eliminating a potential point of desynchronization between the configuration expected by the model and the one assumed by the inference code.

### 3.3 Explicit features/model consistency safeguard

```python
if len(self.features) != self.n_features:
    raise ValueError(
        f"❌ features_used.pkl has {len(self.features)} features but "
        f"model expects {self.n_features}."
    )
```

This check, **absent from module 06's Keras version**, raises an explicit and immediately understandable error in the event of a mismatch between the number of features listed in `features_used.pkl` and the number actually expected by the loaded model — a simple but valuable safeguard, which turns a silent error or a cryptic low-level exception (wrong tensor shape) into an explicit diagnostic message right at initialization, before even the first prediction attempt.

### 3.4 Prediction loop: unchanged logic, adapted inference call

The rest of the class (`predict_soh`) reuses **identically** the logic already documented for `lstm_core_5.py` (module 06, §4.2): checking the expected features, ordering them into a DataFrame, normalization via `scaler_X`, managing the FIFO buffer of `seq_len` cycles, returning `None` as long as the buffer is not full, final clipping `np.clip(pred, 0.0, 1.05)`. Only the inference step itself changes API:

```python
self.interpreter.set_tensor(self.input_details[0]['index'], X_seq)
self.interpreter.invoke()
out = self.interpreter.get_tensor(self.output_details[0]['index'])
pred = float(out[0][0])
```

The TFLite interpreter calling protocol (write the input tensor at the designated index, invoke the graph, read the output tensor at the designated index) is the standard, correctly applied protocol of the `tf.lite.Interpreter` API.

---

## 4. Artifacts provided, artifacts missing

### 4.1 `.tflite` file absent from the archive

The constructor's default parameter points to a file named **`soh_model_SIMPLE.tflite`**:

```python
def __init__(self, model_path='soh_model_SIMPLE.tflite', ...):
```

**This file is not present in the provided archive.** The only model delivered is `cubesat_soh_lstm.keras` (1.8 MB), whose size matches exactly that of the **v2 reference model** documented in modules 05 and 06 (1.86 MB for `cubesat_soh_lstm_unrolled.keras`), not the optimized v3 version (505 KB). This module therefore documents the **software infrastructure for the switch to TFLite**, but the conversion itself (`.keras` → `.tflite`, possibly with post-training quantization) is neither included nor documented in the files provided — this conversion step will need to be carried out separately (typically via `tf.lite.TFLiteConverter.from_keras_model()`) before this test bench can actually be run.

The file name "SIMPLE" suggests that a deliberately simplified variant of the model (possibly quantized, or a reduced architecture close to v3) was intended for this conversion — consistent with the embedded-optimization effort already underway in module 05. Without the actual `.tflite` file or its conversion script, however, it is not possible here to confirm its exact content (fp32 floating-point precision, or int8/fp16 quantization).

### 4.2 Python module naming inconsistency — recurrence of an issue already identified in module 06

`cubesat_eps_main_fixed.m`, strictly identical to its module 06 version (see §2), still contains:

```matlab
module = py.importlib.import_module('lstm_core_5');
```

Yet the only Python file present in this folder is named **`lstm_core_6_tflite.py`**, not `lstm_core_5.py`. This run would therefore fail, as-is, with a `ModuleNotFoundError` on the Python side — **exactly the same operational defect** already noted in the module 06 documentation (§4.4, `optimized_model` folder). Its reappearance here, in a third, independent set of files, confirms that this is not an isolated incident but a **recurring friction point in the workflow**: every time a new Python inference component is introduced (a new file name reflecting new logic — `_test`, `_tflite`), the orchestrating MATLAB script is not updated accordingly. A structural fix would be to externalize the name of the Python module to be imported into a configuration variable at the top of `cubesat_eps_main_fixed.m` rather than hard-coding it, so that a change of inference component would require modifying only a single, clearly identifiable line.

### 4.3 Other artifacts required but not provided

As with module 06, `scaler_X.pkl` and `features_used.pkl` are required by `CubeSatSOHPredict.__init__` but absent from the archive — to be copied from module 05's outputs corresponding to the model actually converted to `.tflite`.

---

## 5. Analysis of the provided result (`inference.jpg`)

### 5.1 Nature of the document

Unlike module 06's `.fig` files (whose vector data could be extracted and used numerically), `inference.jpg` is a **static raster image** (a screenshot of the final MATLAB figure): its underlying data cannot be recovered, only a qualitative visual reading is possible.

### 5.2 Qualitative reading of the curve

The figure reproduces the same format as those in module 06 ("CubeSat LSTM Real-Time SOH Prediction (FINAL)", real SOH in solid blue, predicted SOH in dashed red, EOL threshold at 70% in dashed black), with two notable observations:

1. **Systematic, persistent overestimation bias**: the red curve (prediction) sits **above** the blue curve (reality) over nearly the entire visible trajectory — a gap visually on the order of 2 to 3 SOH points, which only seems to close very late, right at the end of the trajectory. This is a **different** bias profile from the one quantified in module 06 for the unconverted v2 model (where the bias was instead negative in early life, see module 06 §6.3). If the `.tflite` file used here does indeed involve **quantization** (suggested by the name `soh_model_SIMPLE.tflite`, see §4.1), this change in bias profile would be consistent with a well-known effect in embedded-model deployment engineering: post-training quantization (reducing the numerical precision of weights and/or activations, typically from fp32 to int8) **can alter the model's fine-grained behavior** relative to its original floating-point version, including introducing or shifting a systematic bias — an effect that should be precisely quantified (direct comparison of `.keras` vs. `.tflite` outputs on the same input sequences) before validating this converted model for operational use. **This explanation remains a plausible, unconfirmed hypothesis**: in the absence of the actual `.tflite` file in the archive, it is not possible to directly verify whether this is a quantized model or a simple floating-point re-export of the original Keras model.

2. **End of life reached at a noticeably different cycle**: the real SOH (blue) crosses the 70% threshold around **cycle 5,050**, versus **cycle 6,808** obtained in module 06 for a test bench nevertheless based on identical physics code and the same random seed `rng(42)` (see §2). Since the `SOH_true` trajectory is entirely determined by the MATLAB physical simulation — independent of the Python inference component used — strictly identical code and an identical seed should in principle produce a **rigorously reproducible** physical trajectory, as was verified between the two module 06 test benches (v2 and v3, both reaching EOL at the same cycle 6,808). **This gap has no obvious explanation from reading the provided code alone**; possible hypotheses include a partial or interrupted run before completion (a screenshot taken mid-run rather than at its end), a different `n_max` configuration or seed state at the time of this particular capture, or an earlier version of the physics script not found in the current archive. This point would deserve clarification before using this figure as a performance reference, and a full re-run with the results CSV preserved (`CubeSat_REALTIME_LSTM_FINAL.csv`, also not provided here) would remove the ambiguity.

### 5.3 What this figure nevertheless demonstrates

Despite the two caveats above, the figure **successfully demonstrates the end-to-end operation of the TFLite inference chain** integrated into the MATLAB simulation loop: the converted model produces predictions consistent in order of magnitude with physical reality throughout a degradation trajectory of several thousand cycles, with no divergence or visible numerical instability — a first-level functional validation, though not quantified with the same rigor as that obtained in module 06 through extraction of the vector data from the `.fig` files.

---

## 6. Limitations and recommendations

1. **`.tflite` file absent from the archive (§4.1)** — blocking for execution: the converted model must be provided (or regenerated via `TFLiteConverter`) before any new run; its conversion mode (quantized or not) should be explicitly documented, as it directly determines the interpretation of the bias observed in §5.2.
2. **Recurring Python module naming inconsistency (§4.2)** — blocking for execution, and a **symptom of a broader process issue**: recommend introducing a centralized configuration variable for the Python inference module name in `cubesat_eps_main_fixed.m`, to prevent this defect from recurring at each new iteration of the inference component (already observed three times: modules `lstm_core_5`, `lstm_core_5_test`, `lstm_core_6_tflite`).
3. **Unexplained end-of-life cycle discrepancy (§5.2, point 2)** — to be investigated as a priority: re-run this test bench to full completion, keeping the `CubeSat_REALTIME_LSTM_FINAL.csv` file, to check whether the physical trajectory does reproduce module 06's 6,808 cycles, or whether an actual configuration difference (physical parameters, seed, `n_max`) explains this gap.
4. **Prediction bias apparently different from the unconverted version (§5.2, point 1)** — to be rigorously quantified: directly compare, on the same set of input sequences, the outputs of the original `.keras` model and those of the converted `.tflite` model, in order to precisely isolate the numerical error introduced by the conversion (and any quantization) from the one already characterized for the unconverted model (module 06, §6.3).
5. **No embedded performance measurements** — this module documents the functional integration of the TFLite interpreter, but provides no measurement of **inference latency**, **final model size on disk/flash**, or **runtime memory consumption** — metrics that are nevertheless central to judging the viability of a deployment on the STM32F401RE target already mentioned in module 05. These measurements should be added in a future iteration of this test bench, ideally under real hardware conditions or via an embedded-target simulator (TFLite Micro).
6. **Missing `scaler_X.pkl` / `features_used.pkl` artifacts (inherited from module 06)** — same recommendations as previously documented.

---

## 7. Conclusion

This module documents the technical step of **switching the inference engine from native Keras to TensorFlow Lite**, a necessary evolution that has been correctly undertaken on the software side: the new inference bridge (`lstm_core_6_tflite.py`) uses the appropriate API (`tf.lite.Interpreter`), introduces a dynamic reading of the model's input shape (a robustness improvement over the previous version), and an explicit safeguard for consistency between expected features and those of the loaded model. End-to-end execution is functionally demonstrated by the provided figure, but two anomalies prevent drawing a reliable quantitative conclusion as things stand: a significant, unexplained discrepancy in the end-of-life cycle compared to module 06 (even though the physics code is strictly identical), and a visibly different prediction bias profile, plausibly linked to an undocumented quantization effect. Combined with the absence of the `.tflite` file actually used and the recurrence of a module naming defect already identified in the previous module, these observations clearly point to the issues to address before this TFLite inference chain can be considered validated for final integration: provide and document the model conversion process, fix the MATLAB-Python bridge durably, and re-run a full test campaign with systematic preservation of raw data (CSV) rather than only the final figure.
