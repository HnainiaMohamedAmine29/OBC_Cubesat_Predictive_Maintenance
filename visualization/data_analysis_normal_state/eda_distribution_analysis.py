"""
===================================================
EDA - Distribution Analysis
CubeSat EPS Battery Dataset (Normal Conditions)
===================================================
Analyses:
  1. Univariate distributions (histogram + KDE)
  2. Q-Q plots (normality check)
  3. Skewness & Kurtosis summary
  4. Box plots per battery
  5. Pairplot / Correlation heatmap
  6. SOH-conditioned distributions (early/mid/late life)
===================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, normaltest, anderson
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 0. CONFIG
# ──────────────────────────────────────────────
DATA_PATH  = "battery_dataset_normal.csv"
OUTPUT_DIR = "outputs/eda_distribution"

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

C_MAIN   = "#2D6A9F"
C_ACCENT = "#E07B39"
C_GRID   = "#EEF3F9"
C_HIST   = "#5B9BD5"
C_KDE    = "#C0392B"
C_GOOD   = "#27AE60"

# Features to analyse
NUM_COLS = [
    "SOC_start", "SOC_end", "DoD", "IR_ohm",
    "QD_Ah", "QC_Ah", "V_mean_V", "V_min_V", "V_max_V",
    "Tavg_C", "Tmin_C", "Tmax_C",
    "chargetime_min", "discharge_time_min", "SOH", "T_amb_K"
]

LABELS = {
    "SOC_start"         : "SOC Start",
    "SOC_end"           : "SOC End",
    "DoD"               : "Depth of Discharge",
    "IR_ohm"            : "Internal Resistance (Ω)",
    "QD_Ah"             : "Discharge Capacity (Ah)",
    "QC_Ah"             : "Charge Capacity (Ah)",
    "V_mean_V"          : "Mean Voltage (V)",
    "V_min_V"           : "Min Voltage (V)",
    "V_max_V"           : "Max Voltage (V)",
    "Tavg_C"            : "Avg Temperature (°C)",
    "Tmin_C"            : "Min Temperature (°C)",
    "Tmax_C"            : "Max Temperature (°C)",
    "chargetime_min"    : "Charge Time (min)",
    "discharge_time_min": "Discharge Time (min)",
    "SOH"               : "State of Health",
    "T_amb_K"           : "Ambient Temperature (K)",
}

# ──────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────
print("=" * 60)
print("  CubeSat EPS – Distribution Analysis EDA")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
print(f"\nDataset: {df.shape[0]} rows × {df.shape[1]} cols\n")


# ──────────────────────────────────────────────
# 2. UNIVARIATE: Histogram + KDE grid
# ──────────────────────────────────────────────
def plot_histkde_grid(df, cols, labels, out_dir):
    ncols = 4
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.5, nrows * 3.6))
    axes = axes.flatten()
    fig.suptitle("Univariate Distributions — Histogram + KDE",
                 fontsize=14, fontweight="bold", color=C_MAIN, y=1.01)

    for i, col in enumerate(cols):
        ax = axes[i]
        data = df[col].dropna()

        # histogram
        ax.hist(data, bins=40, density=True,
                color=C_HIST, alpha=0.65, edgecolor="white", lw=0.4)

        # KDE overlay
        kde_x = np.linspace(data.min(), data.max(), 300)
        kde   = stats.gaussian_kde(data)
        ax.plot(kde_x, kde(kde_x), color=C_KDE, lw=2.0)

        # normal fit overlay
        mu, sigma = data.mean(), data.std()
        norm_y = stats.norm.pdf(kde_x, mu, sigma)
        ax.plot(kde_x, norm_y, color=C_GOOD, lw=1.5, ls="--", label="Normal fit")

        # vertical lines
        ax.axvline(mu,        color="black", lw=1.2, ls="--", alpha=0.7)
        ax.axvline(data.median(), color=C_ACCENT, lw=1.2, ls=":", alpha=0.9)

        sk = data.skew()
        ku = data.kurtosis()
        ax.set_title(f"{labels[col]}", fontsize=9.5, color=C_MAIN, pad=3)
        ax.set_xlabel("")
        ax.set_ylabel("Density", fontsize=8)
        ax.set_facecolor(C_GRID)
        ax.grid(True, ls="--", alpha=0.4, color="white")
        ax.text(0.97, 0.95,
                f"sk={sk:.2f}\nku={ku:.2f}",
                transform=ax.transAxes, fontsize=7.5,
                va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C_MAIN, alpha=0.8))

    # hide unused subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = f"{out_dir}/hist_kde_grid.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Histogram+KDE grid saved → {path}")

print("── Histogram + KDE ──────────────────────────")
plot_histkde_grid(df, NUM_COLS, LABELS, OUTPUT_DIR)


# ──────────────────────────────────────────────
# 3. Q-Q PLOTS (normality visual check)
# ──────────────────────────────────────────────
def plot_qq_grid(df, cols, labels, out_dir):
    ncols = 4
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4, nrows * 3.5))
    axes = axes.flatten()
    fig.suptitle("Q-Q Plots (Normal Quantiles)",
                 fontsize=14, fontweight="bold", color="#1A4A72", y=1.01)

    for i, col in enumerate(cols):
        ax = axes[i]
        data = df[col].dropna()
        (osm, osr), (slope, intercept, r) = stats.probplot(data, dist="norm")
        ax.scatter(osm, osr, color=C_HIST, s=4, alpha=0.5, rasterized=True)
        fit_line = np.array(osm) * slope + intercept
        ax.plot(osm, fit_line, color=C_KDE, lw=1.8)

        # R² annotation
        ax.set_title(labels[col], fontsize=9.5, color="#1A4A72", pad=3)
        ax.set_xlabel("Theoretical Quantiles", fontsize=8)
        ax.set_ylabel("Sample Quantiles", fontsize=8)
        ax.set_facecolor(C_GRID)
        ax.grid(True, ls="--", alpha=0.4, color="white")
        ax.text(0.05, 0.92, f"R²={r**2:.4f}",
                transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C_ACCENT, alpha=0.8))

    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = f"{out_dir}/qq_plots.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Q-Q plots saved → {path}")

print("\n── Q-Q Plots ────────────────────────────────")
plot_qq_grid(df, NUM_COLS, LABELS, OUTPUT_DIR)


# ──────────────────────────────────────────────
# 4. SKEWNESS & KURTOSIS SUMMARY TABLE + BAR
# ──────────────────────────────────────────────
def plot_skew_kurt_summary(df, cols, labels, out_dir):
    stats_df = pd.DataFrame({
        "Feature"  : [labels[c] for c in cols],
        "Skewness" : [df[c].skew()     for c in cols],
        "Kurtosis" : [df[c].kurtosis() for c in cols],
        "Mean"     : [df[c].mean()     for c in cols],
        "Std"      : [df[c].std()      for c in cols],
        "Median"   : [df[c].median()   for c in cols],
    })
    # print to console
    print("\n  Skewness & Kurtosis:")
    print(stats_df[["Feature", "Skewness", "Kurtosis"]].to_string(index=False))

    # bar chart
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    colors_sk = [C_ACCENT if abs(s) > 0.5 else C_GOOD for s in stats_df["Skewness"]]
    colors_ku = [C_ACCENT if abs(k) > 3   else C_GOOD for k in stats_df["Kurtosis"]]

    ax1.barh(stats_df["Feature"], stats_df["Skewness"], color=colors_sk, edgecolor="white")
    ax1.axvline(0, color="black", lw=1)
    ax1.axvline( 0.5, color="gray", lw=1, ls="--", alpha=0.6)
    ax1.axvline(-0.5, color="gray", lw=1, ls="--", alpha=0.6)
    ax1.set_title("Skewness (|>0.5| flagged in orange)", fontsize=11,
                  fontweight="bold", color=C_MAIN)
    ax1.set_facecolor(C_GRID)
    ax1.grid(True, ls="--", alpha=0.4, color="white", axis="x")

    ax2.barh(stats_df["Feature"], stats_df["Kurtosis"], color=colors_ku, edgecolor="white")
    ax2.axvline(0, color="black", lw=1)
    ax2.axvline( 3, color="gray", lw=1, ls="--", alpha=0.6, label="Excess kurtosis=3")
    ax2.set_title("Excess Kurtosis (|>3| flagged in orange)", fontsize=11,
                  fontweight="bold", color="#1A4A72")
    ax2.set_facecolor(C_GRID)
    ax2.grid(True, ls="--", alpha=0.4, color="white", axis="x")

    plt.suptitle("Distributional Shape Summary", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = f"{out_dir}/skew_kurt_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Skewness/Kurtosis plot saved → {path}")

    # save CSV
    stats_df.to_csv(f"{out_dir}/distribution_stats.csv", index=False)
    print(f"  ✔ Stats table saved → {out_dir}/distribution_stats.csv")

print("\n── Skewness & Kurtosis ──────────────────────")
plot_skew_kurt_summary(df, NUM_COLS, LABELS, OUTPUT_DIR)


# ──────────────────────────────────────────────
# 5. BOX PLOTS  per battery (key features)
# ──────────────────────────────────────────────
def plot_boxplots(df, cols, labels, out_dir):
    key = ["SOH", "IR_ohm", "QD_Ah", "Tavg_C", "discharge_time_min"]
    fig, axes = plt.subplots(len(key), 1, figsize=(16, 3.5 * len(key)))
    fig.suptitle("Per-Battery Box Plots — Key Features",
                 fontsize=14, fontweight="bold", color=C_MAIN, y=1.01)

    for ax, col in zip(axes, key):
        order = sorted(df["battery_id"].unique())
        data_per_batt = [df.loc[df["battery_id"] == b, col].dropna() for b in order]
        bp = ax.boxplot(data_per_batt, patch_artist=True, notch=False,
                        medianprops=dict(color=C_KDE, lw=2))
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(order)))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xticks(range(1, len(order)+1))
        ax.set_xticklabels(order, rotation=45, ha="right", fontsize=7)
        ax.set_title(labels[col], fontsize=11, color=C_MAIN)
        ax.set_facecolor(C_GRID)
        ax.grid(True, ls="--", alpha=0.4, color="white", axis="y")

    plt.tight_layout()
    path = f"{out_dir}/boxplots_per_battery.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Box plots saved → {path}")

print("\n── Box Plots ────────────────────────────────")
plot_boxplots(df, NUM_COLS, LABELS, OUTPUT_DIR)


# ──────────────────────────────────────────────
# 6. CORRELATION HEATMAP
# ──────────────────────────────────────────────
def plot_corr_heatmap(df, cols, labels, out_dir):
    corr = df[cols].rename(columns=labels).corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                linewidths=0.5, linecolor="white",
                annot_kws={"size": 7.5}, ax=ax)
    ax.set_title("Feature Correlation Heatmap",
                 fontsize=14, fontweight="bold", color=C_MAIN, pad=12)
    plt.tight_layout()
    path = f"{out_dir}/correlation_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Correlation heatmap saved → {path}")

print("\n── Correlation Heatmap ──────────────────────")
plot_corr_heatmap(df, NUM_COLS, LABELS, OUTPUT_DIR)


# ──────────────────────────────────────────────
# 7. SOH-CONDITIONED DISTRIBUTIONS
#    (Early / Mid / Late battery life)
# ──────────────────────────────────────────────
def plot_soh_conditioned(df, cols, labels, out_dir):
    q33 = df["SOH"].quantile(0.33)
    q66 = df["SOH"].quantile(0.66)

    df["life_stage"] = pd.cut(
        df["SOH"],
        bins=[-np.inf, q33, q66, np.inf],
        labels=["Late Life\n(SOH low)", "Mid Life", "Early Life\n(SOH high)"]
    )

    key = ["IR_ohm", "QD_Ah", "V_mean_V", "discharge_time_min", "Tavg_C"]
    palette = {"Early Life\n(SOH high)": C_GOOD,
               "Mid Life":               C_ACCENT,
               "Late Life\n(SOH low)":   C_KDE}

    fig, axes = plt.subplots(1, len(key), figsize=(5*len(key), 5))
    fig.suptitle("SOH-Conditioned Feature Distributions\n(Early / Mid / Late Battery Life)",
                 fontsize=13, fontweight="bold", color=C_MAIN)

    for ax, col in zip(axes, key):
        for stage, grp in df.groupby("life_stage", observed=True):
            data = grp[col].dropna()
            kde_x = np.linspace(data.min(), data.max(), 200)
            kde   = stats.gaussian_kde(data)
            ax.fill_between(kde_x, kde(kde_x),
                            alpha=0.35, color=palette[str(stage)])
            ax.plot(kde_x, kde(kde_x), lw=2, color=palette[str(stage)],
                    label=str(stage))
        ax.set_title(labels[col], fontsize=10, color=C_MAIN)
        ax.set_xlabel("")
        ax.set_ylabel("Density", fontsize=8)
        ax.set_facecolor(C_GRID)
        ax.grid(True, ls="--", alpha=0.4, color="white")
        ax.legend(fontsize=7)

    plt.tight_layout()
    path = f"{out_dir}/soh_conditioned_distributions.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔ SOH-conditioned distributions saved → {path}")

print("\n── SOH-Conditioned Distributions ───────────")
plot_soh_conditioned(df, NUM_COLS, LABELS, OUTPUT_DIR)


# ──────────────────────────────────────────────
# 8. NORMALITY TESTS
# ──────────────────────────────────────────────
def run_normality_tests(df, cols, labels, out_dir):
    results = []
    for col in cols:
        data = df[col].dropna()
        sample = data.sample(min(5000, len(data)), random_state=42)

        # D'Agostino K²
        _, p_dagostino = normaltest(sample)
        # Anderson-Darling
        ad = anderson(sample, dist="norm")
        ad_sig = ad.significance_level[2]   # 5% level
        ad_pass = ad.statistic < ad.critical_values[2]

        results.append({
            "Feature"        : labels[col],
            "N"              : len(data),
            "D'Agostino p"   : round(p_dagostino, 5),
            "Normal (p>0.05)": "Yes" if p_dagostino > 0.05 else "No",
            "Anderson stat"  : round(ad.statistic, 4),
            "Anderson 5%"    : ad.critical_values[2],
            "AD pass"        : "Yes" if ad_pass else "No",
        })

    rdf = pd.DataFrame(results)
    rdf.to_csv(f"{out_dir}/normality_tests.csv", index=False)
    print("\n  Normality Test Results:")
    print(rdf[["Feature", "D'Agostino p", "Normal (p>0.05)", "AD pass"]].to_string(index=False))
    print(f"\n  ✔ Normality tests saved → {out_dir}/normality_tests.csv")

print("\n── Normality Tests ──────────────────────────")
run_normality_tests(df, NUM_COLS, LABELS, OUTPUT_DIR)

print(f"\n{'='*60}")
print("  Distribution Analysis EDA complete.")
print(f"  All outputs → {OUTPUT_DIR}/")
print(f"{'='*60}\n")
