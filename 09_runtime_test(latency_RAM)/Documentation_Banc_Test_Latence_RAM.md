# Technical Documentation — Runtime Performance Test Bench: Latency and RAM (Module 09)

**Document scope:** `09_runtime_test(latency_RAM)`
**Files analyzed:** `lstm_core_7_tflite.py`, `benchmark_inference.py`, `cubesat_main.m`, `compute_features.m`, `cubesat_params.m`, `cubesat_ode_ch.m`, `cubesat_ode_dch.m`, `cubesat_events.m`, `simulate_cycle.m`, `inference.jpg`, `inference_latency_RAM.jpg`

This document complements modules 02 (physical simulation), 05 (LSTM v2/v3 models), 06 (Keras integration test bench) and 08 (switch to TensorFlow Lite). It documents the **runtime performance instrumentation** (inference latency and memory footprint) added to the TFLite inference chain — a direct response to a limitation explicitly noted in the module 08 documentation ("absence of embedded performance measurements").

---

## 1. Purpose and archive contents

This module introduces three new elements compared to module 08:

1. **`lstm_core_7_tflite.py`**: a new version of the TFLite inference bridge, enriched with a full latency and RAM measurement and reporting system.
2. **`benchmark_inference.py`**: a standalone Python script allowing model performance to be measured **independently of MATLAB**, avoiding the overhead of the `pyenv` bridge.
3. **`cubesat_main.m`**: a new version of the main simulation loop (renamed from `cubesat_eps_main_fixed.m`), which retrieves and uses the performance statistics collected on the Python side to produce a CSV log and a dedicated figure.

The physics files (`cubesat_params.m`, `cubesat_ode_ch.m`, `cubesat_ode_dch.m`, `cubesat_events.m`, `simulate_cycle.m`, `compute_features.m`) are, after direct comparison, **identical** to their module 08 counterparts — not re-described here.

---

## 2. Simplification of the simulation scenario in `cubesat_main.m`

### 2.1 Removal of random orbital variability

Unlike `cubesat_eps_main_fixed.m` (modules 06 and 08), `cubesat_main.m` **no longer contains** the `rng(42)` call, nor the truncated Gaussian draw of eclipse/sunlight durations (`eclipse_min`, `sunlight_min`), nor the alternation of `T_amb_eclipse` based on cycle parity. The loop directly calls `simulate_cycle(x, p, cycle)` with the **nominal, fixed** parameters from `cubesat_params()` (`T_amb_eclipse = 263.15 K` constant, nominal orbital durations of 35/55 minutes at every cycle, with no random draw).

**Technical justification for this choice, in the specific context of this module:** the goal here is no longer to validate the SOH model's accuracy against realistic orbital variability (already covered by modules 06 and 08), but to **isolate and characterize the behavior of the inference engine itself** (latency, memory consumption) independently of any physical simulation noise. A deterministic physical trajectory, perfectly reproducible from one run to another, is a sensible choice for this specific use case: it removes a source of variability irrelevant to the test's goal, which makes it easier to compare one benchmark run against another. **This choice does, however, have an important consequence to keep in mind**: this module should not be used as a reference for the model's accuracy on a realistic orbital scenario — for that, refer to modules 06 and 08, which retain the full random variability.

### 2.2 Retroactive clarification of a module 08 anomaly

A direct comparison of the `inference.jpg` files from module 08 and the present module 09 reveals that they are the **same file, bit for bit** (identical MD5 checksum). This **resolves the anomaly reported in the module 08 documentation** (§5.2), where the observed end-of-life cycle (≈ 5,050) did not match that of module 06 (6,808) despite apparently identical physics code: the figure delivered in module 08 actually comes from a run using the **deterministic, noise-free** scenario described in §2.1 (probably an earlier run of `cubesat_main.m`, reused by mistake or for convenience as an illustration in the module 08 delivery), rather than from a run of `cubesat_eps_main_fixed.m` with full orbital variability as the content of that archive suggested. This clarification reinforces the recommendation already made in module 08 to systematically keep the results CSV files associated with each delivered figure, in order to unambiguously trace its exact origin (script and configuration used).

---

## 3. Performance instrumentation in `lstm_core_7_tflite.py`

### 3.1 Measured quantities

The `CubeSatSOHPredict` class is enriched with three sets of measurements, collected at **every call to `predict_soh()` that actually triggers an inference** (i.e., once the 30-cycle sliding buffer is full):

| Measurement | Temporal scope | Collection method |
|---|---|---|
| `_t_total_ms` | from entering `predict_soh()` to exiting (preprocessing + normalization + inference + clipping) | `time.perf_counter()` before/after the whole function |
| `_t_infer_ms` | only the `interpreter.invoke()` call | `time.perf_counter()` strictly wrapping the three lines `set_tensor`/`invoke`/`get_tensor` |
| `_ram_mb` | resident memory of the **entire Python process** (RSS), sampled right after each inference | `psutil.Process(os.getpid()).memory_info().rss` |

The distinction between **total** latency and **pure inference** latency is a good profiling practice: it isolates the portion of time consumed by Python preprocessing (building the `DataFrame`, `scaler_X.transform` normalization, buffer management) from what is actually attributable to the TFLite engine — a useful breakdown for targeting optimization effort should total latency ever become critical.

### 3.2 Structured performance report

`get_performance_report(exclude_warmup=1)` computes, separately for total latency and pure inference latency, a standard set of statistics (mean, standard deviation, min, max, 50th/95th/99th percentiles), both over **all calls** and over a **"steady-state"** subset excluding the first `exclude_warmup` calls. This latter distinction is relevant and correctly justified in a comment: *"the first calls after `allocate_tensors()` are typically slower due to lazy initialization or cache effects"* — a well-known phenomenon in inference runtimes (the first execution of a graph is often more expensive than subsequent ones, until memory caches and any JIT optimizations stabilize).

`print_performance_report()` renders this report as readable console text, and `get_latency_history()` exposes the raw series for external use (plotting, export) — which is exactly what `cubesat_main.m` does on the MATLAB side (see §5).

### 3.3 Model size

```python
self.model_size_mb = os.path.getsize(model_path) / (1024 ** 2)
```

The size of the `.tflite` file on disk is measured directly at load time and included in the performance report — a simple but essential metric for assessing compatibility with an embedded flash storage budget (recall from module 05: STM32F401RE target, 512 KB of flash available).

---

## 4. Standalone benchmark script (`benchmark_inference.py`)

### 4.1 Principle: avoiding the MATLAB/pyenv overhead

The script runs **entirely in Python**, without going through MATLAB or `pyenv`'s `OutOfProcess` execution mode (whose inter-process communication itself introduces non-negligible latency, independent of the model). This is an important methodological separation: it provides a latency measurement **specific to the model and its inference engine**, not polluted by the cost of MATLAB↔Python communication — a complementary and more fundamental measurement than the one obtained under full integration conditions (modules 06/08/09, MATLAB loop).

### 4.2 Generating synthetic feature vectors

```python
def make_fake_feature_dict(features, cycle):
    rng = np.random.default_rng(cycle)
    ...
```

The associated docstring is explicit and methodologically sound: *"The values don't need to be physically realistic — we are timing the scaler and the interpreter, not the model's accuracy"*. This deliberate choice to **separate accuracy validation (modules 05/06/08) from performance testing (this module)** is good engineering practice: it avoids coupling two distinct concerns (is the model correct? is the model fast and lightweight?) into a single test, which, should a failure occur, would make it easier to immediately identify the nature of the problem.

### 4.3 Warm-up and measurement protocol

```python
N_WARMUP_CYCLES = 5
N_BENCH_CYCLES  = 500
n_total_calls = N_WARMUP_CYCLES + predictor.seq_len + N_BENCH_CYCLES
```

Out of the `5 + 30 + 500 = 535` total calls to `predict_soh()`, the first 30 trigger no inference (sliding buffer fill-up, see module 08 §3.4), the next 5 serve as an explicit **warm-up**, and the last 500 constitute the actual measurement sample. The script finally calls `predictor.print_performance_report(exclude_warmup=3)` — a trim value of 3 calls, more conservative than the method's default value (`exclude_warmup=1`), consistent with the 5 warm-up cycles already planned earlier in the protocol.

The script also measures process RAM **before loading the model**, **right after loading**, and **at the end of the benchmark** — allowing the memory cost to be broken down between the interpreter's base footprint (graph loading + tensor allocation) and any memory drift over repeated calls (memory leak), a relevant robustness test before any long-duration deployment.

### 4.4 Identified defect: import of a non-existent module — recurrence of an already-observed issue

```python
from lstm_core_6_tflite import CubeSatSOHPredict
```

The only inference-bridge Python file delivered in this archive is named **`lstm_core_7_tflite.py`**, not `lstm_core_6_tflite.py`. As-is, `benchmark_inference.py` would fail at runtime with a `ModuleNotFoundError`. This is the **fourth occurrence** of this same naming-consistency defect across modules 06, 08 and 09 (previously: `lstm_core_5` imported instead of `lstm_core_5_test`, then instead of `lstm_core_6_tflite`; here: `lstm_core_6_tflite` imported instead of `lstm_core_7_tflite`). It is notable that **`cubesat_main.m` correctly imports `'lstm_core_7_tflite'`** (see §5.1) — so the fix was indeed applied to the main MATLAB script for this iteration, but not carried over to the new `benchmark_inference.py` script added alongside it. This confirms, even more clearly than in previous modules, that this is a **recurring, systemic process defect** (every new file consuming the inference bridge must be manually and independently updated each time the Python module is renamed) rather than an isolated oversight — the recommendation already made in module 08 (centralizing the Python module name in a shared configuration) remains fully valid and would benefit from being applied to this script as well.

---

## 5. Integration of performance reporting into `cubesat_main.m`

### 5.1 Correct import of the inference bridge

```matlab
module = py.importlib.import_module('lstm_core_7_tflite');
```

Unlike the finding in §4.4, this line does correctly reference the file actually present in the archive — no anomaly of this type to report here.

### 5.2 Retrieval and export of performance statistics

At the end of the simulation, the MATLAB script directly queries the persistent Python object:

```matlab
py_predictor.print_performance_report();
perf = py_predictor.get_performance_report();
perf = struct(perf);
hist = py_predictor.get_latency_history();
```

The `py.dict → MATLAB struct` conversion (`struct(perf)`) and the direct indexing of the Python tuples returned by `get_latency_history()` (`hist{1}`, `hist{2}`, `hist{3}`) demonstrate a good command of MATLAB↔Python interoperability: the statistics accumulated **on the Python side throughout the entire physical simulation** (rather than recomputed separately) are used directly to produce:

- **`CubeSat_LSTM_PERF.csv`**: a per-cycle log of the three latency/RAM series, with the cycle index correctly realigned (`infer_cycle_idx = (cycle - n_inf + 1):cycle`, accounting for the offset introduced by the buffer fill-up phase);
- **`inference_latency_RAM.jpg`**: a two-panel figure (latency per cycle, RAM per cycle), analyzed in detail in §6.

This performance CSV file is, however, **not provided** in the delivered archive — only the exported figure is; the quantitative analysis in the following section is therefore based on a graphical reading of the image, with the precision limitations this implies.

---

## 6. Technical reading of `inference_latency_RAM.jpg`

### 6.1 Inference latency (top panel)

| Quantity | Approximate graphical reading |
|---|---|
| Total latency (`total_ms`) — steady state | ≈ 4 to 6 ms, with regular cycle-to-cycle noise occasionally reaching ≈ 10 ms |
| Pure inference latency (`interpreter.invoke()`) — steady state | ≈ 1 to 3 ms, noticeably lower and less noisy than total latency |
| Average total latency (black dashed line) | ≈ 4 to 5 ms |
| One-off anomaly | isolated spike reaching ≈ 45 ms around cycle 2,950–3,000, with a second, smaller spike of ≈ 16 ms around cycle 3,200 |

**Gap between total latency and pure inference latency:** the orange (total) curve is consistently about 2 to 4 ms above the blue (pure inference) curve in steady state, this gap representing the cost of **Python preprocessing** (building the `pandas` `DataFrame`, `scaler_X.transform` call, buffer management) — directly actionable information for prioritizing future optimization effort: on this measurement, preprocessing accounts for a share **at least as large as inference itself** in the total latency budget, suggesting that reimplementing normalization (for example as direct NumPy matrix computation rather than via a `pandas`/`scikit-learn` `scaler_X` object) could bring a notable performance gain — particularly relevant with a view to a future port to an embedded environment, where `pandas` would not be available anyway.

**Latency anomaly (spike at ≈ 45 ms):** this value, occurring only once and in isolation (affecting a very limited number of cycles), is consistent with a classic measurement artifact in a non-real-time environment: an OS interruption, a Python garbage-collector pass, or contention with another process on the test machine — phenomena intrinsic to running on a general-purpose OS (Windows, based on the paths observed in previous modules) rather than a flaw in the model or the inference code itself. This kind of jitter, although statistically rare here, illustrates a fundamental limitation of this measurement protocol for estimating **guaranteed real-time** behavior (see §7): a bare-metal embedded system without a general-purpose OS would normally not show this kind of stochastic spike, but also cannot be validated by a measurement taken on a Windows/Python development machine.

### 6.2 Memory consumption (bottom panel)

| Phase | Observed RAM (RSS, entire Python process) |
|---|---|
| Initial peak (loading the model and the TensorFlow runtime, cycles ≈ 0–150) | ≈ 280 to 300 MB |
| Rapid decline (cycles ≈ 150–1,000) | drops in steps from ≈ 300 MB to ≈ 45–50 MB |
| Steady state (cycles ≈ 1,000–5,050) | ≈ 38 to 48 MB, with a slow, slight step-wise decline down to ≈ 39 MB by the end of the run |

The initial peak of ≈ 300 MB very likely corresponds to **loading the full TensorFlow runtime** (native libraries, conversion graph, internal TFLite interpreter structures) rather than the model itself (whose on-disk size, on the order of a few hundred kilobytes to ≈ 1.8 MB based on previous modules, would on its own justify only a tiny fraction of this footprint). The step-wise decline observed afterward is consistent with **Python garbage-collector** activity, progressively freeing temporary objects created during the first calls (NumPy/TensorFlow conversion structures, internal caches) once they are no longer referenced.

---

## 7. Fundamental limitation: this bench measures a desktop Python process, not the real embedded target

This is the single most important point to take away from this module, and it must be stated unambiguously: **the latency and RAM measured here characterize the execution of `tf.lite.Interpreter` within a full Python process running on the development machine** (likely a Windows PC, based on the paths observed in `cubesat_main.m`), **and not an execution on the final hardware target** (STM32F401RE, 94 KB of RAM, ARM Cortex-M4 core, see module 05 §4.1). Two major gaps limit the direct transposability of these results:

1. **RAM**: the steady-state RAM measured here (≈ 40 MB) represents roughly **425 times** the total RAM budget of the target (94 KB), and the loading peak (≈ 300 MB) represents roughly **3,200 times** it. This gap does not mean the model is too large for the target — most of this footprint comes from the full Python/TensorFlow runtime, entirely absent from a real embedded deployment, which would instead use **TensorFlow Lite for Microcontrollers (TFLite Micro)**, a static C++ library with no Python interpreter and no dynamic heap allocation, whose memory footprint is several orders of magnitude smaller. The relevant RAM metric for assessing embedded viability would be the **tensor arena RAM** allocated by TFLite Micro for this specific model, a different metric from the one produced by this test bench and not yet measured in the modules provided to date.
2. **Latency**: the latency measured here (≈ 1 to 3 ms for pure inference) is measured on a desktop x86/x64 processor, clocked in the gigahertz range, with large L1/L2/L3 caches and possibly vector instructions (AVX) leveraged by the TensorFlow runtime. An ARM Cortex-M4 core such as the one in the STM32F401RE (clocked at 84 MHz, with no level-3 cache, and a much more limited instruction set) would execute the same computation graph **significantly more slowly** — a slowdown factor that cannot be estimated without a test run directly on the target, or failing that, via a dedicated cycle-simulation tool (for example the ST X-CUBE-AI toolchain, which provides a latency and tensor-arena RAM estimate directly for the targeted STM32 chip from the same `.tflite` file).

This module therefore documents a **solid, well-designed performance measurement infrastructure** (see §3 and §5), but whose current numerical results (§6) should be interpreted as **indicators of the model's relative behavior** (useful, for example, for comparing two model versions against each other on the same machine, or for detecting a performance regression or a memory leak) rather than as **embedded viability validation**, which remains a separate step still to be carried out on the real target or via a dedicated simulation tool.

---

## 8. Summary of limitations and recommendations

1. **Measurements not transposable to the embedded target (§7)** — highest priority for the rest of the project: schedule a measurement campaign on the real target (or via X-CUBE-AI / TFLite Micro on a simulator), the only way to validate the model's viability on the STM32F401RE. The current results remain useful as a regression-tracking reference on the development machine, but must not be presented as embedded validation.
2. **Persistent recurrence of the Python module naming defect (§4.4)** — fourth occurrence identified across modules 06, 08 and 09; confirms the need for a structural fix (centralized configuration of the module name) rather than repeated one-off fixes at each new iteration.
3. **Unexplained one-off latency anomaly (§6.1)** — to monitor: document whether this ≈ 45 ms spike is a reproducible isolated event (to be investigated by repeating the test) or a one-off artifact of the measurement machine with no consequence.
4. **Significant share of preprocessing in total latency (§6.1)** — concrete optimization avenue: consider normalization via direct matrix computation rather than via the current `scaler_X` object, particularly useful for any future rewrite in C/C++ for the final embedded target, where the `pandas`/`scikit-learn` libraries will not be available anyway.
5. **`CubeSat_LSTM_PERF.csv` not provided in the archive** — as with previous modules, systematically keeping the raw results CSV files (rather than just the figures) would greatly facilitate any precise quantitative analysis after the fact — the analysis in this section had to rely on a graphical reading of the image in the absence of this file.
6. **Persistent absence of model artifacts (`.tflite`, `scaler_X.pkl`, `features_used.pkl`)** — as in modules 06 and 08, these files remain necessary for any re-run of this test bench and are not provided in any of the archives examined to date.

---

## 9. Conclusion

This module provides a direct and technically solid response to the limitation identified in module 08: a complete latency and memory-consumption measurement infrastructure, consistently integrated both on the Python side (`lstm_core_7_tflite.py`, with a clean separation between total latency and pure inference latency, and a relevant distinction between raw measurements and steady-state measurements) and on the MATLAB side (`cubesat_main.m`, which uses these statistics to produce a dedicated log and visualization), complemented by a standalone benchmark script that avoids the overhead of MATLAB integration. Analysis of the produced results reveals a latency of a few milliseconds and a memory footprint of a few tens of megabytes in steady state — figures that look reassuring at first glance, but which must be interpreted with important methodological caution: they characterize a desktop Python process running the full TensorFlow runtime, not the final microcontroller target the project is aiming for, whose real validation requires a dedicated test on hardware or via a specialized embedded simulation tool. In addition, a comparative examination of the `inference.jpg` files from modules 08 and 09 made it possible to clear up an anomaly left unresolved in the previous module's documentation, incidentally illustrating the value of rigorous traceability (systematically keeping the raw data and exact configuration associated with each figure) for the reliability of any technical documentation built across several successive iterations of the same project.
