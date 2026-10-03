// app_uart_protocol.h
#ifndef APP_UART_PROTOCOL_H_
#define APP_UART_PROTOCOL_H_

#include <stdint.h>
#include <stdbool.h>
#include "tflm_c_api.h"
#include "soh_preproc.h"

#define REQ_HEADER_0   0xAA
#define REQ_HEADER_1   0x55          // v1: host sends the scaled 30x20 window   (unchanged)
#define REQ_HEADER_1_RAW 0x56        // v2: host sends only the raw simulation row (edge preprocessing)
#define RESP_HEADER_0  0xBB
#define RESP_HEADER_1  0x66          // v1 response
#define RESP_HEADER_1_RAW 0x67       // v2 response (adds status + timing + arena)

#define REQ_PACKET_LEN   (2 + 2 + SOH_INPUT_LEN * 4 + 2)   // 2406 bytes (v1)
#define RESP_PACKET_LEN  (2 + 2 + 4 + 2)                    // 10 bytes   (v1)

// ---- v2 request, 104 bytes, little endian ------------------------------------------------
//  [0]=0xAA [1]=0x56 [2:3]=cycle u16 [4]=flags u8 [5]=0
//  [6:93]   11 x float64: V_mean_V V_min_V V_max_V Tavg_C Tmin_C Tmax_C QD_Ah QC_Ah DoD discharge_time_min T_amb_K
//  [94:101] soh_lag_ext float64 (only used when flags bit0 = 1)
//  [102:103] CRC16/CCITT-FALSE over bytes [0:101]
#define RAW_N_VALUES        11
#define RAW_REQ_PACKET_LEN  (2 + 2 + 1 + 1 + RAW_N_VALUES * 8 + 8 + 2)   // 104
#define RAW_FLAG_EXT_LAG    0x01     // use soh_lag_ext instead of the model's own previous output

// ---- v2 response, 24 bytes ----------------------------------------------------------------
//  [0]=0xBB [1]=0x67 [2:3]=cycle echo u16 [4]=status u8 [5]=0 [6:9]=soh float32
//  [10:13]=t_preproc_us u32 [14:17]=t_infer_us u32 [18:21]=arena_used_bytes u32 [22:23]=CRC16
#define RAW_RESP_PACKET_LEN (2 + 2 + 1 + 1 + 4 + 4 + 4 + 4 + 2)          // 24
#define RAW_STATUS_OK        0
#define RAW_STATUS_DUPLICATE 1       // same cycle received twice: cached answer re-sent, state not advanced
#define RAW_STATUS_GAP       2       // a cycle was skipped: history is incomplete, prediction still computed
#define RAW_STATUS_INFER_ERR 3
#define RAW_STATUS_WARMUP    4       // first SOH_WARMUP_CYCLES cycles: nominal SOH 1.0 reported, model NOT run (window not yet made of real rows)

typedef enum { PKT_NONE = 0, PKT_WINDOW, PKT_RAW } pkt_type_t;

typedef struct {
    uint16_t       cycle;
    uint8_t        flags;
    soh_raw_row_t  row;
    double         soh_lag_ext;
} raw_request_t;

// Waits for one packet of either version. Fills `window` (v1) or `raw` (v2).
pkt_type_t Uart_ReceiveAny(uint16_t *cycle_num, float *window, raw_request_t *raw);

bool Uart_ReceiveRequest(uint16_t *cycle_num, float *window);   // v1 only (kept for compatibility)
void Uart_SendResponse(uint16_t cycle_num, float soh_pred);
void Uart_SendRawResponse(uint16_t cycle_num, uint8_t status, float soh_pred,
                          uint32_t t_pre_us, uint32_t t_inf_us, uint32_t arena_used);

#endif // APP_UART_PROTOCOL_H_
