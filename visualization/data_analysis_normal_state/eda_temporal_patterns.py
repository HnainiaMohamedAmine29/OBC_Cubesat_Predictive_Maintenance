"""
===================================================
EDA - Temporal Patterns Analysis
CubeSat EPS Battery Dataset (Normal Conditions)
===================================================
Analyses:
  1. ACF  - Autocorrelation Function
  2. PACF - Partial Autocorrelation Function
  3. STL  - Seasonal-Trend Decomposition using LOESS
===================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 0. CONFIG
# ──────────────────────────────────────────────
DATA_PATH   = "battery_dataset_normal.csv"
OUTPUT_DIR  = "outputs/eda_temporal"
LAGS        = 60          # ACF / PACF lags
STL_PERIOD  = 20          # approximate charge-discharge sub-period (cycles)
ALPHA       = 0.05        # confidence level for autocorrelation bands

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Colour palette
C_MAIN   = "#2D6A9F"
C_ACCENT = "#E07B39"
C_GRID   = "#E8EFF6"
C_NEG    = "#C0392B"
C_POS    = "#27AE60"

# ──────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────
print("=" * 60)
print("  CubeSat EPS – Temporal Pattern EDA")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

# ALL 16 numerical features — temporal analysis applied to every one
TARGETS = {
    "SOC_start"          : "SOC Start",
    "SOC_end"            : "SOC End",
    "DoD"                : "Depth of Discharge",
    "IR_ohm"             : "Internal Resistance (Ω)",
    "QD_Ah"              : "Discharge Capacity (Ah)",
    "QC_Ah"              : "Charge Capacity (Ah)",
    "V_mean_V"           : "Mean Voltage (V)",
    "V_min_V"            : "Min Voltage (V)",
    "V_max_V"            : "Max Voltage (V)",
    "Tavg_C"             : "Avg Temperature (°C)",
    "Tmin_C"             : "Min Temperature (°C)",
    "Tmax_C"             : "Max Temperature (°C)",
    "chargetime_min"     : "Charge Time (min)",
    "discharge_time_min" : "Discharge Time (min)",
    "SOH"                : "State of Health",
    "T_amb_K"            : "Ambient Temperature (K)",
}

batteries = sorted(df["battery_id"].unique())
print(f"\nDataset : {len(df)} rows | {len(batteries)} batteries")
print(f"Cycles  : {df['cycle'].min()} → {df['cycle'].max()}\n")


# ──────────────────────────────────────────────
# 2. HELPER – select representative battery
#    (longest cycle history)
# ──────────────────────────────────────────────
def pick_representative(df, n=3):
    counts = df.groupby("battery_id")["cycle"].count().sort_values(ascending=False)
    return counts.index[:n].tolist()

rep_batts = pick_representative(df)
print(f"Representative batteries chosen: {rep_batts}\n")


# ──────────────────────────────────────────────
# 3. ACF ANALYSIS
# ──────────────────────────────────────────────
def plot_acf_panel(df, battery_id, targets, lags, out_dir):
    bdf = df[df["battery_id"] == battery_id].sort_values("cycle")
    n   = len(targets)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.2 * n))
    fig.suptitle(
        f"ACF Analysis — Battery: {battery_id}\n"
        f"(lags = {lags}, α = {ALPHA})",
        fontsize=14, fontweight="bold", color=C_MAIN, y=1.01
    )
    for ax, (col, label) in zip(axes, targets.items()):
        series = bdf[col].dropna()
        plot_acf(series, lags=min(lags, len(series)//2 - 1),
                 alpha=ALPHA, ax=ax, color=C_MAIN,
                 vlines_kwargs={"colors": C_MAIN})
        ax.set_title(label, fontsize=11, color=C_MAIN, pad=4)
        ax.set_xlabel("Lag (cycles)", fontsize=9)
        ax.set_ylabel("ACF", fontsize=9)
        ax.set_facecolor(C_GRID)
        ax.grid(True, ls="--", alpha=0.4, color="white")
        # annotate first zero-crossing
        acf_vals = ax.lines[0].get_ydata()[1:]
        crossings = np.where(np.diff(np.sign(acf_vals)))[0]
        if len(crossings):
            ax.axvline(crossings[0]+1, color=C_ACCENT,
                       ls="--", lw=1.5, label=f"1st zero-cross @ lag {crossings[0]+1}")
            ax.legend(fontsize=8)
    plt.tight_layout()
    path = f"{out_dir}/acf_{battery_id}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔ ACF saved → {path}")

print("── ACF ─────────────────────────────────────")
for bid in rep_batts:
    plot_acf_panel(df, bid, TARGETS, LAGS, OUTPUT_DIR)


# ──────────────────────────────────────────────
# 4. PACF ANALYSIS
# ──────────────────────────────────────────────
def plot_pacf_panel(df, battery_id, targets, lags, out_dir):
    bdf = df[df["battery_id"] == battery_id].sort_values("cycle")
    n   = len(targets)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.2 * n))
    fig.suptitle(
        f"PACF Analysis — Battery: {battery_id}\n"
        f"(lags = {lags}, method = Yule-Walker)",
        fontsize=14, fontweight="bold", color="#1A4A72", y=1.01
    )
    for ax, (col, label) in zip(axes, targets.items()):
        series = bdf[col].dropna()
        max_lags = min(lags, len(series)//2 - 1)
        plot_pacf(series, lags=max_lags, method="ywm",
                  alpha=ALPHA, ax=ax, color=C_ACCENT,
                  vlines_kwargs={"colors": C_ACCENT})
        ax.set_title(label, fontsize=11, color="#1A4A72", pad=4)
        ax.set_xlabel("Lag (cycles)", fontsize=9)
        ax.set_ylabel("PACF", fontsize=9)
        ax.set_facecolor("#FFF3EC")
        ax.grid(True, ls="--", alpha=0.4, color="white")
        ax.axhline(0, color="black", lw=0.8)
    plt.tight_layout()
    path = f"{out_dir}/pacf_{battery_id}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔ PACF saved → {path}")

print("\n── PACF ─────────────────────────────────────")
for bid in rep_batts:
    plot_pacf_panel(df, bid, TARGETS, LAGS, OUTPUT_DIR)


# ──────────────────────────────────────────────
# 5. STL DECOMPOSITION
# ──────────────────────────────────────────────
def plot_stl_panel(df, battery_id, targets, period, out_dir):
    bdf = df[df["battery_id"] == battery_id].sort_values("cycle")
    for col, label in targets.items():
        series = bdf[col].dropna().reset_index(drop=True)
        if len(series) < 2 * period + 1:
            print(f"  ⚠ {col}: too short for STL (need >{2*period+1} pts)")
            continue

        stl = STL(series, period=period, robust=True)
        res = stl.fit()

        fig = plt.figure(figsize=(15, 9))
        gs  = gridspec.GridSpec(4, 1, hspace=0.45)

        components = [
            (series,          "Observed",  C_MAIN,   0),
            (res.trend,       "Trend",     C_ACCENT,  1),
            (res.seasonal,    "Seasonal",  C_POS,    2),
            (res.resid,       "Residual",  C_NEG,    3),
        ]

        axes = []
        for data, title, color, idx in components:
            ax = fig.add_subplot(gs[idx])
            ax.plot(bdf["cycle"].values[:len(data)], data,
                    color=color, lw=1.3 if idx < 3 else 0.9,
                    alpha=0.9)
            if idx == 3:   # residuals: fill between zero
                ax.fill_between(bdf["cycle"].values[:len(data)], data,
                                alpha=0.3, color=C_NEG)
                ax.axhline(0, color="black", lw=0.8, ls="--")
            ax.set_title(title, fontsize=11, fontweight="bold",
                         color=color, pad=3)
            ax.set_facecolor(C_GRID)
            ax.grid(True, ls="--", alpha=0.4, color="white")
            if idx == 3:
                ax.set_xlabel("Cycle Number", fontsize=9)
            axes.append(ax)

        fig.suptitle(
            f"STL Decomposition — {label}\n"
            f"Battery: {battery_id} | Period: {period} cycles",
            fontsize=13, fontweight="bold", color=C_MAIN
        )

        # strength metrics
        var_resid   = np.var(res.resid)
        Ft = max(0, 1 - var_resid / np.var(res.trend   + res.resid))
        Fs = max(0, 1 - var_resid / np.var(res.seasonal + res.resid))
        fig.text(0.98, 0.5,
                 f"Trend strength:    {Ft:.3f}\nSeasonal strength: {Fs:.3f}",
                 ha="right", va="center", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.4", fc="#EAF2FF", ec=C_MAIN))

        safe_col = col.replace("/", "_")
        path = f"{out_dir}/stl_{battery_id}_{safe_col}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✔ STL [{col}] saved → {path}")

print("\n── STL Decomposition ───────────────────────")
for bid in rep_batts:          # all 3 representative batteries, all 16 features
    print(f"\n  Battery: {bid}")
    plot_stl_panel(df, bid, TARGETS, STL_PERIOD, OUTPUT_DIR)


# ──────────────────────────────────────────────
# 6. SUMMARY DASHBOARD  (cross-battery trends)
# ──────────────────────────────────────────────
print("\n── Cross-Battery Trend Summary ─────────────")

summary_cols = list(TARGETS.keys())   # all 16 features
fig, axes = plt.subplots(len(summary_cols), 1, figsize=(15, 4 * len(summary_cols)))
fig.suptitle("Cross-Battery Temporal Trends", fontsize=14,
             fontweight="bold", color=C_MAIN)

palette = plt.cm.tab10(np.linspace(0, 0.9, len(batteries)))

for ax, col in zip(axes, summary_cols):
    for i, bid in enumerate(batteries):
        bdf = df[df["battery_id"] == bid].sort_values("cycle")
        ax.plot(bdf["cycle"], bdf[col],
                alpha=0.55, lw=1.1, color=palette[i],
                label=bid if len(batteries) <= 12 else None)
    # fleet mean
    fleet_mean = df.groupby("cycle")[col].mean()
    ax.plot(fleet_mean.index, fleet_mean.values,
            color="black", lw=2.5, ls="--", label="Fleet Mean", zorder=5)
    ax.set_title(TARGETS[col], fontsize=11, color=C_MAIN)
    ax.set_xlabel("Cycle Number", fontsize=9)
    ax.set_facecolor(C_GRID)
    ax.grid(True, ls="--", alpha=0.4, color="white")
    if len(batteries) <= 12:
        ax.legend(fontsize=7, ncol=3, loc="upper right")

plt.tight_layout()
path = f"{OUTPUT_DIR}/cross_battery_trends.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✔ Cross-battery trends saved → {path}")

print(f"\n{'='*60}")
print("  Temporal Pattern EDA complete.")
print(f"  All outputs → {OUTPUT_DIR}/")
print(f"{'='*60}\n")
