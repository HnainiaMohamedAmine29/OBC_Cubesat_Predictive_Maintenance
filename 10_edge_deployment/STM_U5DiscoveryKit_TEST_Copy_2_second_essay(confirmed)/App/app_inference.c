// app_inference.c
// B-U585I-IOT02A board: uses LD1/PH7 (this board's user LED).
// NOTE: LD1 is ACTIVE-LOW on this board (LED lights when pin = 0).
//
// Two modes, selected by the packet the host sends:
//   v1 (0xAA55): host sends the scaled window            -> infer                      (unchanged)
//   v2 (0xAA56): host sends the raw simulation row only  -> features + scaling + window + infer ON THE BOARD

#include "app_inference.h"
#include "tflm_c_api.h"
#include "app_uart_protocol.h"
#include "soh_preproc.h"
#include "main.h"
#include <stdio.h>
#include <string.h>

#define SOH_WARNING_THRESHOLD 0.80f
// Warm-up window: the board predicts from cycle 1. On the first call the 30-row window is seeded with copies of the
// first scaled row (soh_preproc.c, same as build_window.m), so the LSTM can run from the very first cycle.
// OPTIONAL gate (default 0 = off, i.e. your design): set SOH_WARMUP_CYCLES to N to report the nominal SOH 1.0 with
// status RAW_STATUS_WARMUP for the first N cycles and skip the model, e.g. -DSOH_WARMUP_CYCLES=30.
#ifndef SOH_WARMUP_CYCLES
#define SOH_WARMUP_CYCLES     0
#endif
#define LOG_EVERY_N_CYCLES    100     // semihosting printf is slow (ms per call): print rarely. 1 = every cycle

// ---- DWT cycle counter (Cortex-M33) for latency measurement ----
static inline void Cyc_Init(void) {
    DCB->DEMCR |= DCB_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL  |= DWT_CTRL_CYCCNTENA_Msk;
}
static inline uint32_t Cyc_Now(void) { return DWT->CYCCNT; }
static inline uint32_t Cyc_ToUs(uint32_t cycles) {
    return (uint32_t)(((uint64_t)cycles * 1000000ull) / (uint64_t)SystemCoreClock);
}

static soh_preproc_t g_pre;          // ~2.7 KB: history rings + scaled window
static uint16_t g_last_cycle = 0;    // 0 = nothing processed yet
static float    g_last_soh   = 1.0f; // model's previous output -> next SOH_lag_1 (starts at BOL = 1.0)
static struct { uint8_t status; float soh; uint32_t t_pre, t_inf; } g_cache;

int App_Inference_Init(void) {
    Cyc_Init();
    soh_preproc_reset(&g_pre);
    tflm_status_t status = tflm_init();
    if (status != TFLM_OK) {
        printf("TFLM init failed with code %d\r\n", (int)status);
        return -1;
    }
    printf("TFLM initialized OK. Arena used: %lu bytes\r\n", (unsigned long)tflm_arena_used());
    return 0;
}

static void Update_Led(float soh) {
    HAL_GPIO_WritePin(LED_GREEN_GPIO_Port, LED_GREEN_Pin,
                      soh < SOH_WARNING_THRESHOLD ? GPIO_PIN_RESET : GPIO_PIN_SET);   // 0 = LED ON
}

static void Handle_Raw(const raw_request_t *rq) {
    const uint16_t cyc = rq->cycle;

    if (cyc == 1) {                              // new run from the host: forget everything
        soh_preproc_reset(&g_pre);
        g_last_cycle = 0;
        g_last_soh = 1.0f;
    }
    if (g_last_cycle != 0 && cyc == g_last_cycle) {      // host retried (e.g. lost response): do not advance state
        Uart_SendRawResponse(cyc, RAW_STATUS_DUPLICATE, g_cache.soh, g_cache.t_pre, g_cache.t_inf, tflm_arena_used());
        return;
    }
    uint8_t status = RAW_STATUS_OK;
    if (g_last_cycle != 0 && cyc != (uint16_t)(g_last_cycle + 1)) status = RAW_STATUS_GAP;

    const double lag = (rq->flags & RAW_FLAG_EXT_LAG) ? rq->soh_lag_ext : (double)g_last_soh;

    uint32_t t0 = Cyc_Now();
    soh_preproc_step(&g_pre, cyc, &rq->row, lag, NULL);
    uint32_t t1 = Cyc_Now();

    if (cyc <= SOH_WARMUP_CYCLES) {                 // warm-up: history/window updated above, model skipped
        g_last_cycle  = cyc;
        g_last_soh    = 1.0f;                       // lag stays at the nominal value
        g_cache.soh   = 1.0f;
        g_cache.t_pre = Cyc_ToUs(t1 - t0);
        g_cache.t_inf = 0;
        Uart_SendRawResponse(cyc, RAW_STATUS_WARMUP, 1.0f, g_cache.t_pre, 0, tflm_arena_used());
        return;
    }

    float soh = 0.0f;
    tflm_status_t st = tflm_infer(g_pre.window, &soh);
    uint32_t t2 = Cyc_Now();

    g_last_cycle = cyc;
    if (st != TFLM_OK) {
        printf("Inference failed (cycle %u) code %d\r\n", cyc, (int)st);
        g_cache.status = RAW_STATUS_INFER_ERR; g_cache.soh = 0.0f / 0.0f;
        g_cache.t_pre = Cyc_ToUs(t1 - t0); g_cache.t_inf = 0;
        Uart_SendRawResponse(cyc, RAW_STATUS_INFER_ERR, g_cache.soh, g_cache.t_pre, 0, tflm_arena_used());
        return;
    }
    g_last_soh   = soh;
    g_cache.soh  = soh;
    g_cache.t_pre = Cyc_ToUs(t1 - t0);
    g_cache.t_inf = Cyc_ToUs(t2 - t1);

    Uart_SendRawResponse(cyc, status, soh, g_cache.t_pre, g_cache.t_inf, tflm_arena_used());

    if ((cyc % LOG_EVERY_N_CYCLES) == 0 || cyc == 1)
        printf("cycle %u -> SOH %.4f | preproc %lu us | infer %lu us\r\n", cyc, soh,
               (unsigned long)g_cache.t_pre, (unsigned long)g_cache.t_inf);
    Update_Led(soh);
}

static void Handle_Window(uint16_t cycle_num, const float *window) {
    float soh_pred = 0.0f;
    tflm_status_t status = tflm_infer(window, &soh_pred);
    if (status != TFLM_OK) {
        printf("Inference failed (cycle %u) with code %d\r\n", cycle_num, (int)status);
        return;
    }
    Uart_SendResponse(cycle_num, soh_pred);
    if ((cycle_num % LOG_EVERY_N_CYCLES) == 0 || cycle_num == 1)
        printf("cycle %u -> predicted SOH = %.4f\r\n", cycle_num, soh_pred);
    Update_Led(soh_pred);
}

void App_Inference_RunOnce(void) {
    static float window[SOH_INPUT_LEN];
    static raw_request_t raw;
    uint16_t cycle_num = 0;

    switch (Uart_ReceiveAny(&cycle_num, window, &raw)) {
        case PKT_RAW:    Handle_Raw(&raw);                 break;
        case PKT_WINDOW: Handle_Window(cycle_num, window); break;
        default:         break;
    }
}
