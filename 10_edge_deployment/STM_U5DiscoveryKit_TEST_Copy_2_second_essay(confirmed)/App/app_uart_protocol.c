// app_uart_protocol.c
// B-U585I-IOT02A board: uses huart1 (USART1) - the single physical UART
// wired to the ST-Link VCP / COM port on the host PC. printf debug output goes
// through semihosting (see main.c), not this wire.
//
// Two request formats share the same first header byte (0xAA):
//   0xAA 0x55  v1  scaled window from MATLAB   (original protocol, unchanged)
//   0xAA 0x56  v2  raw simulation row          (preprocessing done on the board)

#include <string.h>
#include "app_uart_protocol.h"
#include "main.h"

extern UART_HandleTypeDef huart1;

#define UART_TIMEOUT_MS 3000

static uint16_t Crc16CcittFalse(const uint8_t *data, uint32_t len) {
    uint16_t crc = 0xFFFF;
    for (uint32_t i = 0; i < len; i++) {
        crc ^= (uint16_t)(data[i] << 8);
        for (int b = 0; b < 8; b++) {
            crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021)
                                  : (uint16_t)(crc << 1);
        }
    }
    return crc;
}

pkt_type_t Uart_ReceiveAny(uint16_t *cycle_num, float *window, raw_request_t *raw) {
    static uint8_t rx_buf[REQ_PACKET_LEN];      // large enough for both formats (2406 >= 104)

    uint8_t b;
    if (HAL_UART_Receive(&huart1, &b, 1, UART_TIMEOUT_MS) != HAL_OK) return PKT_NONE;
    if (b != REQ_HEADER_0) return PKT_NONE;
    if (HAL_UART_Receive(&huart1, &b, 1, UART_TIMEOUT_MS) != HAL_OK) return PKT_NONE;

    rx_buf[0] = REQ_HEADER_0;
    rx_buf[1] = b;

    if (b == REQ_HEADER_1) {                                              // ---- v1: window ----
        if (HAL_UART_Receive(&huart1, &rx_buf[2], REQ_PACKET_LEN - 2, UART_TIMEOUT_MS) != HAL_OK) return PKT_NONE;
        uint16_t received_crc;
        memcpy(&received_crc, &rx_buf[REQ_PACKET_LEN - 2], 2);
        if (Crc16CcittFalse(rx_buf, REQ_PACKET_LEN - 2) != received_crc) return PKT_NONE;
        memcpy(cycle_num, &rx_buf[2], 2);
        memcpy(window, &rx_buf[4], SOH_INPUT_LEN * 4);
        return PKT_WINDOW;
    }

    if (b == REQ_HEADER_1_RAW) {                                          // ---- v2: raw row ----
        if (HAL_UART_Receive(&huart1, &rx_buf[2], RAW_REQ_PACKET_LEN - 2, UART_TIMEOUT_MS) != HAL_OK) return PKT_NONE;
        uint16_t received_crc;
        memcpy(&received_crc, &rx_buf[RAW_REQ_PACKET_LEN - 2], 2);
        if (Crc16CcittFalse(rx_buf, RAW_REQ_PACKET_LEN - 2) != received_crc) return PKT_NONE;

        memcpy(&raw->cycle, &rx_buf[2], 2);
        raw->flags = rx_buf[4];
        double v[RAW_N_VALUES];
        memcpy(v, &rx_buf[6], sizeof(v));                                 // memcpy: no alignment assumptions
        raw->row.V_mean_V = v[0];  raw->row.V_min_V = v[1];  raw->row.V_max_V = v[2];
        raw->row.Tavg_C   = v[3];  raw->row.Tmin_C  = v[4];  raw->row.Tmax_C  = v[5];
        raw->row.QD_Ah    = v[6];  raw->row.QC_Ah   = v[7];  raw->row.DoD     = v[8];
        raw->row.discharge_time_min = v[9];  raw->row.T_amb_K = v[10];
        memcpy(&raw->soh_lag_ext, &rx_buf[6 + RAW_N_VALUES * 8], 8);
        *cycle_num = raw->cycle;
        return PKT_RAW;
    }
    return PKT_NONE;
}

bool Uart_ReceiveRequest(uint16_t *cycle_num, float *window) {
    raw_request_t dummy;
    return Uart_ReceiveAny(cycle_num, window, &dummy) == PKT_WINDOW;
}

void Uart_SendResponse(uint16_t cycle_num, float soh_pred) {
    uint8_t tx_buf[RESP_PACKET_LEN];

    tx_buf[0] = RESP_HEADER_0;
    tx_buf[1] = RESP_HEADER_1;
    memcpy(&tx_buf[2], &cycle_num, 2);
    memcpy(&tx_buf[4], &soh_pred, 4);

    uint16_t crc = Crc16CcittFalse(tx_buf, RESP_PACKET_LEN - 2);
    memcpy(&tx_buf[8], &crc, 2);

    HAL_UART_Transmit(&huart1, tx_buf, RESP_PACKET_LEN, UART_TIMEOUT_MS);
}

void Uart_SendRawResponse(uint16_t cycle_num, uint8_t status, float soh_pred,
                          uint32_t t_pre_us, uint32_t t_inf_us, uint32_t arena_used) {
    uint8_t tx_buf[RAW_RESP_PACKET_LEN];

    tx_buf[0] = RESP_HEADER_0;
    tx_buf[1] = RESP_HEADER_1_RAW;
    memcpy(&tx_buf[2], &cycle_num, 2);
    tx_buf[4] = status;
    tx_buf[5] = 0;
    memcpy(&tx_buf[6],  &soh_pred,   4);
    memcpy(&tx_buf[10], &t_pre_us,   4);
    memcpy(&tx_buf[14], &t_inf_us,   4);
    memcpy(&tx_buf[18], &arena_used, 4);

    uint16_t crc = Crc16CcittFalse(tx_buf, RAW_RESP_PACKET_LEN - 2);
    memcpy(&tx_buf[22], &crc, 2);

    HAL_UART_Transmit(&huart1, tx_buf, RAW_RESP_PACKET_LEN, UART_TIMEOUT_MS);
}
