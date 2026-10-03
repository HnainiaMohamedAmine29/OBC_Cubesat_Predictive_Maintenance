// soh_preproc.c - see soh_preproc.h
#include <string.h>
#include "soh_preproc.h"
#include "soh_scaler_const.h"
#if SOH_PREFILL_MODE
#include "soh_prefill_window.h"
#endif

#if SOH_N_FEAT != SOH_N_FEATURES
#error "scaler constants and tflm_c_api.h disagree on the feature count"
#endif

// Feature indices = model order (features_used.pkl)
enum {
    F_V_MEAN_V, F_V_SPREAD, F_V_MEAN_ROLLING, F_V_MEAN_LAG_1, F_V_MIN_FADE,
    F_TAVG_C, F_THERMAL_RANGE, F_DELTA_T_AMBIENT, F_COLD_CYCLE_COUNT, F_TAVG_ROLLING,
    F_ECLIPSE_FLAG, F_QD_AH, F_COULOMBIC_EFF, F_DISCHARGE_C_RATE, F_CAPACITY_RETENTION,
    F_QD_ROLLING, F_CYCLE, F_CUMUL_AH, F_SOH_LAG_1, F_QD_DIFF
};

static double max2(double a, double b) { return a > b ? a : b; }

// mean of the last min(n, W) entries of a ring, oldest first (= MATLAB mean(history(idx_start:cycle)))
static double ring_mean(const double *ring, uint32_t n) {
    uint32_t cnt = n < SOH_ROLL_W ? n : SOH_ROLL_W;
    uint32_t start = n - cnt;
    double sum = 0.0;
    for (uint32_t k = 0; k < cnt; k++) sum += ring[(start + k) % SOH_ROLL_W];
    return sum / (double)cnt;
}

void soh_preproc_reset(soh_preproc_t *s) { memset(s, 0, sizeof(*s)); }

void soh_preproc_step(soh_preproc_t *s, uint32_t cycle, const soh_raw_row_t *r,
                      double soh_lag, double *raw_feat_out)
{
    const int first = (s->n == 0);

    // --- history update (MATLAB writes history(cycle) BEFORE compute_features) ---
    const double v_mean_prev = s->v_mean_prev, qd_prev = s->qd_prev, t_amb_prev = s->t_amb_prev;
    s->v_mean_hist[s->n % SOH_ROLL_W] = r->V_mean_V;
    s->tavg_hist  [s->n % SOH_ROLL_W] = r->Tavg_C;
    s->qd_hist    [s->n % SOH_ROLL_W] = r->QD_Ah;
    s->n++;
    if (first) { s->v_min_first = r->V_min_V; s->qd_first = r->QD_Ah; }
    if (r->Tavg_C < 5.0) s->cold_count++;
    s->cumul_ah += r->QD_Ah;

    // --- features (compute_features.m, same order as features_used.pkl) ---
    double f[SOH_N_FEATURES];
    f[F_V_MEAN_V]          = r->V_mean_V;
    f[F_V_SPREAD]          = r->V_max_V - r->V_min_V;
    f[F_V_MEAN_ROLLING]    = ring_mean(s->v_mean_hist, s->n);
    f[F_V_MEAN_LAG_1]      = first ? (SOH_CYCLE1_FIX ? r->V_mean_V : 0.0) : v_mean_prev;
    f[F_V_MIN_FADE]        = r->V_min_V - s->v_min_first;
    f[F_TAVG_C]            = r->Tavg_C;
    f[F_THERMAL_RANGE]     = r->Tmax_C - r->Tmin_C;
    f[F_DELTA_T_AMBIENT]   = first ? 0.0 : (r->T_amb_K - t_amb_prev);
    f[F_COLD_CYCLE_COUNT]  = (double)s->cold_count;
    f[F_TAVG_ROLLING]      = ring_mean(s->tavg_hist, s->n);
    f[F_ECLIPSE_FLAG]      = (r->T_amb_K <= 263.15) ? 1.0 : 0.0;
    f[F_QD_AH]             = r->QD_Ah;
    f[F_COULOMBIC_EFF]     = r->QC_Ah / max2(r->QD_Ah, 1e-6);
    f[F_DISCHARGE_C_RATE]  = r->DoD / max2(r->discharge_time_min / 60.0, 1e-6);
    f[F_CAPACITY_RETENTION]= r->QD_Ah / max2(s->qd_first, 1e-6);
    f[F_QD_ROLLING]        = ring_mean(s->qd_hist, s->n);
    f[F_CYCLE]             = (double)cycle;
    f[F_CUMUL_AH]          = s->cumul_ah;
    f[F_SOH_LAG_1]         = soh_lag;
    f[F_QD_DIFF]           = first ? 0.0 : (r->QD_Ah - qd_prev);

    s->v_mean_prev = r->V_mean_V; s->qd_prev = r->QD_Ah; s->t_amb_prev = r->T_amb_K;

    // --- NaN / Inf -> 0 (scale_features.m) and scale ---
    float z[SOH_N_FEATURES];
    for (int i = 0; i < SOH_N_FEATURES; i++) {
        double x = f[i];
        if (x != x || x > 1e300 || x < -1e300) x = 0.0;
        f[i] = x;
        z[i] = (float)((x - SOH_MEAN[i]) / SOH_SCALE[i]);
    }
    if (raw_feat_out) memcpy(raw_feat_out, f, sizeof(f));

    // --- window ---
#if SOH_PREFILL_MODE
    // first call: start from the stored 30-cycle window, then slide the new row in (same as every later call)
    if (first) memcpy(s->window, SOH_PREFILL, sizeof(s->window));
    memmove(s->window, s->window + SOH_N_FEATURES,
            (SOH_WINDOW_LEN - 1) * SOH_N_FEATURES * sizeof(float));
    memcpy(&s->window[(SOH_WINDOW_LEN - 1) * SOH_N_FEATURES], z, sizeof(z));
#else
    // build_window.m: first call seeds all 30 rows with the first row, then slide
    if (first) {
        for (int t = 0; t < SOH_WINDOW_LEN; t++)
            memcpy(&s->window[t * SOH_N_FEATURES], z, sizeof(z));
    } else {
        memmove(s->window, s->window + SOH_N_FEATURES,
                (SOH_WINDOW_LEN - 1) * SOH_N_FEATURES * sizeof(float));
        memcpy(&s->window[(SOH_WINDOW_LEN - 1) * SOH_N_FEATURES], z, sizeof(z));
    }
#endif
}
