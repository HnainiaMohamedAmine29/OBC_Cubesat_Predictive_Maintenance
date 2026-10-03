// soh_preproc.h - on-board feature engineering + scaling + window for the SOH LSTM.
//
// Pure C, no HAL, no malloc: compiles for the STM32 and on a PC (used by
// test/test_preproc_parity.py). Reproduces, in double precision, exactly what
// MATLAB did before: compute_features.m -> scale_features.m -> build_window.m.
//
// The host (MATLAB) now sends only the raw per-cycle simulation row; everything
// else happens here.
#ifndef SOH_PREPROC_H_
#define SOH_PREPROC_H_

#include <stdint.h>
#include "tflm_c_api.h"        // SOH_WINDOW_LEN (30), SOH_N_FEATURES (20), SOH_INPUT_LEN (600)

#ifdef __cplusplus
extern "C" {
#endif

#define SOH_ROLL_W 10          // must match training ROLLING_W / compute_features.m

// Cycle-1 convention for V_mean_lag_1 (no previous cycle exists):
//   0 -> compute_features.m       (z = -57, never seen in training; int8 input saturates)
//   1 -> compute_features_fixed.m (use the current V_mean_V)            <-- default
#ifndef SOH_CYCLE1_FIX
#define SOH_CYCLE1_FIX 1
#endif

// Start-up window (the "warm-up window", so the LSTM can predict from cycle 1):
//   1 -> window starts as 30 DIFFERENT realistic cycles stored in flash (soh_prefill_window.h, first 30 training rows);
//        each real cycle then slides in and the oldest stored row slides out.                     <-- default
//   0 -> window seeded with 30 copies of the first real row (build_window.m, the old behaviour)
#ifndef SOH_PREFILL_MODE
#define SOH_PREFILL_MODE 1
#endif

// The fields of simulate_cycle's `row` that the features need (same names).
typedef struct {
    double V_mean_V, V_min_V, V_max_V;
    double Tavg_C, Tmin_C, Tmax_C;
    double QD_Ah, QC_Ah;
    double DoD, discharge_time_min;
    double T_amb_K;
} soh_raw_row_t;

typedef struct {
    uint32_t n;                              // rows pushed since reset (history length)
    double   v_mean_hist[SOH_ROLL_W];        // ring buffers, last W cycles
    double   tavg_hist[SOH_ROLL_W];
    double   qd_hist[SOH_ROLL_W];
    double   v_min_first, qd_first;          // value at the very first cycle
    double   v_mean_prev, qd_prev, t_amb_prev;
    uint32_t cold_count;                     // cycles with Tavg < 5 C so far
    double   cumul_ah;                       // running sum of QD_Ah
    float    window[SOH_INPUT_LEN];          // scaled [30][20], timestep-major, ready for tflm_infer()
} soh_preproc_t;

void soh_preproc_reset(soh_preproc_t *s);

// Push one cycle. `cycle` is the 1-based cycle number (also the `cycle` feature).
// `soh_lag` is SOH_lag_1 (previous SOH). Optional `raw_feat_out[20]` receives the
// 20 UNSCALED features in model order (debug / parity tests); may be NULL.
// After the call s->window holds the model input.
void soh_preproc_step(soh_preproc_t *s, uint32_t cycle, const soh_raw_row_t *row,
                      double soh_lag, double *raw_feat_out);

#ifdef __cplusplus
}
#endif
#endif // SOH_PREPROC_H_
