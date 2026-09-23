# Technical Documentation — CubeSat Battery Data Visualization Chain

**Document scope:** `03_visualization`
**Scripts analyzed:** `visualization.py`, `visualization - 2.py`, `visualization - 3.py`, `eda_distribution_analysis.py`, `eda_temporal_patterns.py`
**Sets of graphical outputs analyzed:** `normal_state/`, `fault_state/`, `type_state/`, `data_analysis_normal_state/eda_distribution/`, `data_analysis_normal_state/eda_temporal/`

This document complements the documentation of the simulation model (`02_battery_model_Normal_conditions`): it describes the **post-processing and visualization chain** applied to the CSV datasets produced by the simulations, together with the technical interpretation of each generated figure.

---

## 1. Purpose and place in the processing chain

The scripts in this module consume the `battery_dataset_*.csv` files (produced by `cubesat_run.m`, one record per orbital cycle) and produce two families of analyses:

1. **Operational monitoring visualization** (`visualization*.py`): evolution of key quantities (SOH, SOC, IR, temperature, capacities, voltage) as a function of cycle number — the "mission dashboard" view.
2. **Exploratory statistical analysis (EDA)** (`eda_distribution_analysis.py`, `eda_temporal_patterns.py`): in-depth statistical and temporal characterization of the dataset — variable distributions, correlations, stationarity, seasonality — typically used upstream of predictive modeling work (e.g., machine-learning-based SOH/RUL estimation).

```
battery_dataset_*.csv
        │
        ├─► visualization.py / -2 / -3   → 6 individual figures + 1 combined dashboard
        │
        ├─► eda_distribution_analysis.py → univariate analysis, correlations, normality tests
        │
        └─► eda_temporal_patterns.py     → ACF/PACF, STL decomposition, cross-battery trends
```

Three distinct datasets are visualized, corresponding to three scenarios:

| Output folder | Source file (glob) | Corresponding script | Scenario |
|---|---|---|---|
| `normal_state/` | `battery_dataset_normal.csv` | `visualization - 2.py` | Nominal operation, generic CubeSat pack (`Q_nom=4.035 Ah`) — module 02's reference case |
| `fault_state/` | `battery_dataset_fault.csv` | `visualization - 3.py` | Scenario with an injected fault (`γ > 1`, see module 02) |
| `type_state/` | `battery_dataset_Panasonic_NCR18650B_2S1P.csv` | `visualization.py` | Pack based on a real commercial cell (Panasonic NCR18650B, 2S1P configuration), used for cell-type sensitivity comparison |

The three scripts `visualization.py`, `visualization - 2.py` and `visualization - 3.py` are **identical character for character**, except for the name of the CSV file searched for by `glob.glob(...)` (line 9). This is therefore a single tool duplicated three times to be pointed at each of the three datasets, rather than a script parameterized via a command-line argument — a maintainability improvement point noted in section 7.

---

## 2. "Dashboard" visualization script (`visualization*.py`)

### 2.1 Loading and CSV reading robustness

```python
csv_files = glob.glob('battery_dataset_normal.csv')
latest_file = max(csv_files, key=os.path.getmtime)
df = pd.read_csv(latest_file)
```

The script looks for the target CSV file in the current directory, then selects — by construction, given `glob` on a fixed pattern (with no `*` wildcard) — the single file that exactly matches the expected name (the `max(..., key=os.path.getmtime)` mechanism, inherited from an earlier usage with a timestamp in the file name, only has an effect here if several files bear exactly this name, which can only happen with copies in different subdirectories).

A **fallback for the decimal format** is provided:

```python
if df[numeric_cols].dtypes.apply(lambda x: x == 'object').any():
    df = pd.read_csv(latest_file, decimal=',')
```

This safeguard anticipates the case where the CSV was generated or re-exported (for example via Excel with French regional settings) using a **comma as the decimal separator** rather than a period — in which case `pandas` would read the numeric columns as strings (`dtype='object'`). This is a relevant data-engineering precaution for a pipeline intended to be re-run in heterogeneous environments.

### 2.2 Extracted and rescaled quantities

```python
ir_mohm = df['IR_ohm'] * 1000   # Ω → mΩ conversion
```

Only internal resistance undergoes an explicit unit conversion (Ω → mΩ) before plotting, to obtain readable y-axis values (values on the order of 15 to 400 depending on the dataset, rather than 0.015 to 0.4).

### 2.3 The six individual figures

| # | Figure | Quantity/quantities plotted | Color | Expected observation |
|---|---|---|---|---|
| 1 | `SOH_vs_cycles.png` | SOH | blue | monotonic decline, Y axis fixed to `[0.65, 1.02]` for cross-run comparability |
| 2 | `SOC_DoD_vs_cycles.png` | SOC_start, SOC_end, DoD overlaid | green / red / dashed black | stable value bands with slow drift |
| 3 | `IR_vs_cycles.png` | Internal resistance (mΩ) | magenta | near-linear growth, strong cycle-to-cycle scatter |
| 4 | `Tavg_vs_cycles.png` | Average temperature | red | stable band, bimodality linked to the structural temperature alternation |
| 5 | `Capacities_vs_cycles.png` | QD, QC overlaid | blue / red | parallel near-linear decline, QC systematically below QD (effect of efficiency η) |
| 6 | `Vmean_vs_cycles.png` | Average voltage | black | strong cycle-to-cycle variability, no clear trend |

Each individual figure is produced via the generic `plot_single()` function (except figures 2 and 5, multi-curve, plotted "hard-coded"), with consistent formatting: semi-transparent dashed grid, export resolution `dpi=200`, `bbox_inches='tight'`.

### 2.4 Combined dashboard (`dashboard_combined.png`)

A summary figure with 6 subplots (2×3 grid) reproduces the six quantities above in a compact layout, with a title taken from the source file name — useful for a quick "one-pager" review of a simulation's state, or for direct visual comparison between scenarios (normal / fault / cell type) by juxtaposing the three images.

### 2.5 Comparative reading of the three scenarios

Examination of the three produced `dashboard_combined.png` files reveals results consistent with the model physics documented in module 02:

- **`normal_state`**: the pack reaches the end-of-life threshold (SOH = 0.70) after **about 6,800 cycles**. Internal resistance grows from ≈ 40 mΩ to ≈ 110 mΩ. Average temperature oscillates within a stable band (roughly [14, 19] °C), with a clear bimodal pattern due to the programmed alternation of `T_amb_eclipse` (−10 °C / −2 °C) every two orbits.
- **`fault_state`**: the same SOH = 0.70 threshold is reached as early as **≈ 3,000 cycles**, i.e., a lifetime reduced by more than half — a direct signature of the `γ > 1` factor applied to the aging equation (`dSOH/dt`) documented in module 02. Internal resistance also shows **discontinuous jumps** (steps visible around cycles 900, 1,400 and 2,800) absent from the nominal scenario, suggesting that the fault scenario is not limited to a simple continuous acceleration of `γ`, but introduces regime changes (steps of increasing severity) over the course of the simulation.
- **`type_state`**: with a Panasonic NCR18650B cell in a 2S1P configuration (nominal capacity ≈ 3.4 Ah, noticeably lower than the generic pack's 4.035 Ah), the same SOH = 0.70 criterion is reached in only **≈ 330 cycles** — much faster than the two previous scenarios. Internal resistance is also much higher in absolute value (up to ≈ 400 mΩ versus ≈ 110 mΩ nominally), indicating series-resistance parameters (`R0`, `k_R`) recalibrated specifically to represent the real, less favorable behavior of this commercial cell compared to module 02's generic pack. This scenario therefore serves as a **model sensitivity test to the cell's physico-chemical parameter set**, rather than an operational fault test.

---

## 3. Exploratory distribution analysis (`eda_distribution_analysis.py`)

This script performs a systematic univariate and bivariate statistical characterization of the nominal dataset's **16 numerical variables**, with outputs in `outputs/eda_distribution/`.

### 3.1 Histograms + kernel density estimation (KDE)

Function `plot_histkde_grid()`: for each of the 16 variables, a normalized (density) histogram is overlaid with:
- a **Gaussian kernel density estimate** (`scipy.stats.gaussian_kde`), which smooths the histogram without assuming a parametric shape;
- a **Gaussian fit** (`scipy.stats.norm.pdf` fit on the empirical mean/standard deviation), for direct visual comparison to a normal distribution;
- two vertical markers: mean (black line) and median (dashed orange line);
- skewness and kurtosis annotated numerically in each subplot.

**Technical reading of the result (`hist_kde_grid.png`, nominal dataset):**

- `SOC_start` shows **marked negative skewness** (sk = −0.51, the only variable flagged in orange in the skewness/kurtosis summary): the distribution is bunched toward the upper bound (`SOC_max = 0.99`), consistent with the charge controller systematically targeting this ceiling at the end of charge.
- `QD_Ah`, `QC_Ah` and `SOH` show **near-uniform** distributions (strongly negative kurtosis, ≈ −1.19) over their range of variation: this is explained by the **near-linear decrease of SOH over time** (visible on `SOH_vs_cycles.png`) combined with a roughly constant number of cycles spent in each SOH bracket — uniform temporal sampling of a quantity that varies linearly in time produces a near-uniform distribution.
- `Tavg_C` and `Tmin_C` show **clear bimodality** (two well-separated peaks), a direct and expected signature of the deterministic alternation of `T_amb_eclipse` (−10 °C every other cycle, −2 °C the other) programmed in `cubesat_run.m`.
- `T_amb_K` shows a distribution with **two point masses**, consistent with the fact that this variable takes only two discrete values by construction (263.15 K and 271.15 K).
- The **Gaussian fit (green curve)** noticeably diverges from the empirical KDE (red) for most bounded or bimodal variables — expected, since none of these quantities is actually Gaussian by physical construction (strict bounding, deterministic alternation, monotonic trend).

### 3.2 Quantile-quantile plots (Q-Q plots)

Function `plot_qq_grid()`: compares each variable's empirical quantiles to the theoretical quantiles of a normal distribution (`scipy.stats.probplot`), with computation of the R² coefficient of determination of the associated regression line.

**Technical reading (`qq_plots.png`):**

- The variables with **fast, near-stationary per-cycle dynamics** (`discharge_time_min`, R²=0.9996; `chargetime_min`, R²=0.9925) are closest to normality — consistent with the fact that these durations result directly from the truncated Gaussian draw of orbital durations (`eclipse_min`, `sunlight_min`) in `cubesat_run.m`, a normal draw remaining broadly normal even after slight truncation.
- The variables with a **strong monotonic trend** over the entire run (`SOH`, `QD_Ah`, `QC_Ah`, R²≈0.95) visibly diverge from the reference line at the extremes — a classic signature of a distribution mixture (here, a non-stationary/trending process observed over its full range) rather than a stationary random draw.
- The **bimodal** variables (`Tavg_C` R²=0.80, `Tmin_C` R²=0.69, `T_amb_K` R²=0.64) show the most pronounced deviations from normality — consistent with the double mass identified in 3.1. These are the worst fits in the dataset, which is the statistically expected outcome for quantities driven by a deterministic binary process (cycle parity) rather than a continuous random phenomenon.

### 3.3 Skewness / kurtosis summary (`skew_kurt_summary.png`)

Horizontal bars of skewness and (excess) kurtosis for the 16 variables, with flagging thresholds at |skewness| > 0.5 and |excess kurtosis| > 3 (no variable exceeds this second threshold in the nominal dataset — the distributions are bounded, never heavy-tailed). This figure serves as a **quick quantitative summary** of distributional shape, as a visual complement to the histograms, and is also exported as a CSV table (`distribution_stats.csv`) for later use (for example, selecting variable transformations — log, Box-Cox — before modeling).

### 3.4 Per-battery box plots (`boxplots_per_battery.png`)

Function `plot_boxplots()`: compares, for five key variables (SOH, IR, QD, Tavg, discharge time), the full distribution **per battery identifier** (`battery_id`). With `n_batt = 1` in `cubesat_run.m`'s current configuration, only a single box plot appears per subplot — the multi-battery comparison functionality is present in the code but **not exercised** with the provided dataset, which contains only a single battery trajectory (`BATT_EO_001`). It would become relevant for a Monte Carlo-type study with `n_batt > 1`.

### 3.5 Correlation heatmap (`correlation_heatmap.png`)

Pearson correlation matrix (upper triangular mask) across the 16 variables, via `seaborn.heatmap`. The strongest correlations and their physical interpretation:

| Variable pair | Correlation | Interpretation |
|---|---|---|
| `QD_Ah` ↔ `QC_Ah` ↔ `SOH` | ≈ 1.00 | Expected by construction: `QD = SOH·Q_nom` and `QC = SOH·Q_nom·η` are **exact deterministic functions** of SOH (see `simulate_cycle.m`). |
| `QD_Ah` ↔ `IR_ohm` | −0.82 | Consistent with the impedance model coupled to aging (module 02, §4.2): resistance grows as capacity (and hence SOH) declines. |
| `SOC_start` ↔ `SOC_end` ↔ `V_mean_V`/`V_max_V` | 0.81 to 0.98 | Expected: voltage is a directly increasing function of SOC via `OCV(SOC)` (module 02, §4.1). |
| `Tavg_C` ↔ `Tmin_C` | 0.99 | Near-total redundancy: these two statistics derive from the same intra-cycle time series and are dominated by the same bimodal orbital-alternation signal. |
| `Tavg_C`/`Tmin_C` ↔ `T_amb_K` | 0.98 / 1.00 | Direct confirmation that pack temperature is very strongly driven by the imposed eclipse ambient temperature, more than by Joule self-heating (consistent with the low thermal mass and low current of the CubeSat pack). |
| `DoD` ↔ `IR_ohm` | 0.80 | Depth of discharge grows with aging (the battery needs to draw further into SOC to deliver the same energy as its capacity declines), correlated with the resistance rise. |
| `Max Temperature` ↔ `IR_ohm`/`DoD` | 0.42 / 0.50 | Moderate positive correlation, consistent with the Joule self-heating term (`R·I²`) which increases with internal resistance. |

This correlation map serves as a **cross-validation tool for the physical model**: most of the observed correlations directly confirm the analytical relationships laid out in module 02's equations (rather than revealing new empirical relationships), which is an expected and reassuring result for a **synthetic** dataset generated by a deterministic/near-deterministic model.

### 3.6 Distributions conditioned on life stage (`soh_conditioned_distributions.png`)

Function `plot_soh_conditioned()`: the dataset is segmented into three groups by SOH tertiles (`pd.cut` on the 33% and 66% quantiles) — *Early Life* (high SOH), *Mid Life*, *Late Life* (low SOH) — and the KDE distributions of five key variables are then overlaid per group.

**Technical reading:**
- `IR_ohm`: a clear separation between the three groups, with **near-zero overlap** between Early Life and Late Life — confirms that internal resistance is a **highly discriminative health indicator** of life stage, consistent with its explicit modeling as an increasing function of `(1−SOH)` in module 02.
- `QD_Ah`: also a clear, non-overlapping separation, a trivial result since `QD` is a direct affine function of SOH.
- `V_mean_V`: a larger partial overlap between groups — mean voltage is a **less discriminative health indicator on its own** than resistance or capacity, since it also strongly depends on instantaneous SOC (intra-cycle noise that "dilutes" the aging signal).
- `discharge_time_min`: near-total overlap of the three groups — confirms that this duration is essentially driven by random orbital variability (the Gaussian draw of `eclipse_min`), **independent of SOH**, consistent with the model (the discharge-stop event is on SOC, not directly related to aging).
- `Tavg_C`: the three groups nearly perfectly overlap on the same bimodal profile — confirms that pack temperature, driven mainly by the external environment (`T_amb_eclipse`/`T_amb_sun`), is **not an aging indicator** in this model (it depends on SOH only indirectly and weakly, via the Joule term).

This analysis is particularly useful upstream of a **feature-selection** task for a future battery health diagnostic or prognostic model (advanced BMS / RUL estimation): it identifies `IR_ohm` and `QD_Ah`/`QC_Ah` as signals with strong discriminative power for SOH, and `Tavg_C`/`discharge_time_min` as uninformative in this regard (in this model).

### 3.7 Normality tests

Function `run_normality_tests()`: applies, on a subsample (max 5,000 points, fixed seed `random_state=42`), two statistical normality tests:
- the **D'Agostino-Pearson K² test** (`scipy.stats.normaltest`), based on combined skewness and kurtosis;
- the **Anderson-Darling test** (`scipy.stats.anderson`), compared against the critical threshold at the 5% confidence level.

Results are exported to CSV (`normality_tests.csv`) with a boolean conclusion column for each test. Given the sample size (several thousand cycles), these tests are **almost certain to reject the normality hypothesis** for nearly all variables, even for minor deviations from the normal distribution with no practical significance (the test's statistical power grows with sample size) — a result to be interpreted with caution: the normality test's failure here has more to do with **sample size and the bounded/trending nature of the variables** than with a modeling anomaly.

---

## 4. Exploratory temporal analysis (`eda_temporal_patterns.py`)

This script studies the **temporal structure** (autocorrelation, seasonality, trend) of the 16 variables, cycle by cycle, for a subset of "representative batteries".

### 4.1 Selection of representative batteries

```python
def pick_representative(df, n=3):
    counts = df.groupby("battery_id")["cycle"].count().sort_values(ascending=False)
    return counts.index[:n].tolist()
```

Selects the `n=3` battery identifiers with the highest number of recorded cycles (i.e., those that lived the longest before EOL). With `n_batt=1` in the current configuration, only a single battery (`BATT_EO_001`) is actually processed — the function is designed for multi-battery (fleet study) usage not exercised here.

### 4.2 Autocorrelation function (ACF)

Function `plot_acf_panel()`, built on `statsmodels.graphics.tsaplots.plot_acf`, computes the autocorrelation of each variable up to a lag of 60 cycles, with a 95% confidence band (`alpha=0.05`), and detects/annotates the **first zero crossing** of the autocorrelogram.

**Technical reading (`acf_BATT_EO_001.png`):**

- **Trending quantities** (`QD_Ah`, `QC_Ah`, `SOH`, `SOC_end`, voltages): a **slow, near-linear** decline in ACF over the entire lag range tested, with no zero crossing — a typical signature of a **non-stationary, trend-dominated series** (a slow deterministic walk). From a time-series-analysis standpoint, this confirms what the physical model already imposes: aging is a cumulative process that varies slowly from one cycle to the next.
- **Structurally alternating quantities** (`Tavg_C`, `Tmin_C`, `T_amb_K`): the ACF **strictly alternates sign between even and odd lags** (strong positive correlation at even lags, negative at odd lags), the exact, expected signature of a **period-2 periodic pattern** — the direct consequence of the programmed alternation of `T_amb_eclipse` every two orbits in `cubesat_run.m`.
- **Fast, memoryless quantities** (`chargetime_min`, `discharge_time_min`, `Max Temperature`): the ACF drops almost instantly to near zero from lag 1 onward, consistent with an **independent and identically distributed (i.i.d.)** draw cycle to cycle (uncorrelated Gaussian orbital variability from one orbit to the next in the model).

### 4.3 Partial autocorrelation function (PACF)

Function `plot_pacf_panel()`, using the modified Yule-Walker method (`method="ywm"`). The PACF isolates the direct correlation at a given lag, after removing the effect of intermediate lags — a classic tool for identifying the **order of an underlying autoregressive model** (useful should the data ever need to be modeled by an AR(p) or integrated into an ARIMA-type forecasting pipeline).

### 4.4 STL seasonal decomposition (Seasonal-Trend decomposition using LOESS)

Function `plot_stl_panel()`, based on `statsmodels.tsa.seasonal.STL`, with a period fixed at `STL_PERIOD = 20` cycles (chosen as an approximate charge-discharge sub-period according to the code comment — although the dominant periodic phenomenon identified via ACF is actually of **period 2**, not 20; this parameter appears to be set more to capture a trend smoothed over 20-cycle windows than to precisely isolate the even/odd cycle pattern). Each decomposition produces four panels: observed series, trend, seasonal component, residual, with two quantitative metrics:

$$
F_t = \max\left(0,\ 1 - \frac{\mathrm{Var}(\text{residual})}{\mathrm{Var}(\text{trend}+\text{residual})}\right),\qquad
F_s = \max\left(0,\ 1 - \frac{\mathrm{Var}(\text{residual})}{\mathrm{Var}(\text{seasonal}+\text{residual})}\right)
$$

(trend strength and seasonality strength, standard measures from the Hyndman STL literature).

**Representative result observed (`stl_BATT_EO_001_Tavg_C.png`):** seasonal strength ≈ 0.97 (very close to 1 — the periodic component explains nearly all of the variance once the residual is removed), and trend strength ≈ 0.07 (weak but non-zero — a slight drift of the average temperature visible on the "Trend" panel, which can be attributed to the slow increase in internal Joule heating as resistance grows toward end of life). This result quantitatively confirms the qualitative analysis made on the ACF figures and the bimodal histograms: **the model's thermal signal is dominated by the programmed periodic environmental forcing, with a second-order aging drift.**

### 4.5 Cross-battery trends (`cross_battery_trends.png`)

For each of the 16 variables, overlays the trajectory of every battery in the fleet (one color per `battery_id`, legend shown if ≤ 12 batteries) together with the **fleet average** (dashed black curve, `groupby('cycle').mean()`). With only a single battery simulated in the provided dataset, this figure reduces to a single curve coinciding with the fleet average — a key feature for a multi-battery Monte Carlo study (`n_batt > 1` in `cubesat_run.m`), not exercised here but **ready to use** as soon as the data generator is re-run with several battery identifiers.

---

## 5. Summary of produced figures

| Directory | File | Figure type | Generating script |
|---|---|---|---|
| `normal_state/`, `fault_state/`, `type_state/` | `SOH_vs_cycles.png` | simple curve | `visualization*.py` |
| same | `SOC_DoD_vs_cycles.png` | 3 overlaid curves | same |
| same | `IR_vs_cycles.png` | simple curve | same |
| same | `Tavg_vs_cycles.png` | simple curve | same |
| same | `Capacities_vs_cycles.png` | 2 overlaid curves | same |
| same | `Vmean_vs_cycles.png` | simple curve | same |
| same | `dashboard_combined.png` | 2×3 grid | same |
| `type_state/` | `Figure_1.png` | 2×3 grid (title variant) | same |
| `eda_distribution/` | `hist_kde_grid.png` | 4×4 grid, histogram+KDE+Gaussian | `eda_distribution_analysis.py` |
| same | `qq_plots.png` | 4×4 grid, normal Q-Q | same |
| same | `skew_kurt_summary.png` | 2 horizontal bar charts | same |
| same | `boxplots_per_battery.png` | 5 stacked box plots | same |
| same | `correlation_heatmap.png` | correlation matrix | same |
| same | `soh_conditioned_distributions.png` | 5 overlaid KDEs per group | same |
| same | `distribution_stats.csv`, `normality_tests.csv` | statistical tables | same |
| `eda_temporal/` | `acf_<battery>.png` | 16 stacked autocorrelograms | `eda_temporal_patterns.py` |
| same | `pacf_<battery>.png` | 16 stacked partial autocorrelograms | same |
| same | `stl_<battery>_<variable>.png` | 4-panel STL decomposition (×16 variables) | same |
| same | `cross_battery_trends.png` | 16 subplots, trajectories + fleet average | same |

---

## 6. Identified methodological choices and good practices

- **Clear separation of visualization / statistical EDA**: the `visualization*.py` scripts address a need for **quick operational monitoring** (cycle-by-cycle reading, "telemetry" style), while the `eda_*` scripts address a need for **data analysis upstream of modeling** (distribution, correlation, temporal structure) — a relevant separation of concerns in data engineering.
- **Visual consistency**: consistent color palette and grid style across all figures within a given script, making cross-reading easier.
- **Systematic quantitative annotations**: skewness/kurtosis on histograms, R² on Q-Q plots, trend/seasonality strengths on STL — each figure carries the numerical information needed for its interpretation without requiring a separate table (although complementary CSV exports also exist).
- **Anticipation of multi-battery scaling**: the `pick_representative()` functions, the "per battery" box plots, and the cross-battery trends are already coded for `n_batt > 1`, even though the currently provided dataset contains only a single trajectory — a good design practice ahead of immediate usage.

---

## 7. Limitations and possible improvements

1. **Code triplication**: `visualization.py`, `visualization - 2.py` and `visualization - 3.py` are identical except for one line (the target CSV file name). A single script parameterized via a command-line argument (`argparse`) or a configuration variable at the top of the file would reduce duplication and the risk of divergence in future fixes.
2. **`glob` on a fixed pattern followed by `max(..., key=os.path.getmtime)`**: this mechanism, inherited from an earlier usage with a timestamp in the file name (consistent with `cubesat_run.m`, which produces names such as `battery_dataset_normal_<timestamp>.csv`), implicitly assumes the existence of a file with the exact name and no timestamp (`battery_dataset_normal.csv`) — a manual renaming or copying step between the MATLAB output and the Python input is therefore not documented in the provided scripts.
3. **Choice of the STL period (20 cycles)**: does not exactly match the actually identified dominant periodicity (period 2, related to the `T_amb_eclipse` alternation), which can dilute the real seasonal component into the trend component for bimodal variables. A period of 2 (or a reanalysis excluding the known alternation effect) would give a decomposition more faithful to the underlying physical mechanism.
4. **Normality tests not very informative at large sample sizes**: with several thousand observations, statistical normality tests almost systematically reject the null hypothesis for minor deviations with no practical significance. An effect-size metric (e.g., Kolmogorov-Smirnov distance, or a simple visual reading of the Q-Q plot and R²) is more informative here — something the scripts already provide in parallel (Q-Q plots), which partly compensates for this limitation.
5. **Multi-battery features not exercised**: per-battery analyses (box plots, cross-battery trends, representative-battery selection) cannot be fully validated with `n_batt=1`. A run with several batteries simulated in parallel (Monte Carlo over the random variability already present in `cubesat_run.m`) would allow the proper functioning of these functions to be verified and would enrich the inter-unit dispersion analysis.

---

## 8. Conclusion

This visualization chain constitutes a **complete, consistent data-analysis layer**, built on top of the datasets generated by module 02's physical model. It fulfills two complementary functions: a quick visual plausibility check and cross-scenario comparison (normal / fault / cell type) via the dashboards, and an in-depth statistical and temporal characterization (distribution, correlation, autocorrelation, seasonal decomposition) intended to prepare for downstream use — likely training a predictive battery health model (SOH/RUL). The analysis confirms, quantitatively and through an independent path (descriptive and time-series statistics), the internal consistency of the physical model documented in module 02: the observed correlations faithfully reproduce the model's analytical relationships (OCV-SOC, resistance-SOH, capacity-SOH), and the temporal structure of the thermal variables precisely reveals the orbital alternation pattern programmed into the cycle generator.
