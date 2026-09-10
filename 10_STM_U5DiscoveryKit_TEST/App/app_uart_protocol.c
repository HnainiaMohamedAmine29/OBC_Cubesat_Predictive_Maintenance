// app_uart_protocol.c
// B-U585I-IOT02A board: uses huart2 (USART2, PA2/PA3) - the DATA link to MATLAB.
// huart1 (USART1, PA9/PA10) is this board's ST-Link VCP and is debug-printf only.

#include <string.h>
#include "app_uart_protocol.h"
#include "main.h"

extern UART_HandleTypeDef huart2;

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

bool Uart_ReceiveRequest(uint16_t *cycle_num, float *window) {
    static uint8_t rx_buf[REQ_PACKET_LEN];

    uint8_t b;
    if (HAL_UART_Receive(&huart2, &b, 1, UART_TIMEOUT_MS) != HAL_OK) return false;
    if (b != REQ_HEADER_0) return false;
    if (HAL_UART_Receive(&huart2, &b, 1, UART_TIMEOUT_MS) != HAL_OK) return false;
    if (b != REQ_HEADER_1) return false;

    rx_buf[0] = REQ_HEADER_0;
    rx_buf[1] = REQ_HEADER_1;

    if (HAL_UART_Receive(&huart2, &rx_buf[2], REQ_PACKET_LEN - 2, UART_TIMEOUT_MS) != HAL_OK) {
        return false;
    }

    uint16_t received_crc;
    memcpy(&received_crc, &rx_buf[REQ_PACKET_LEN - 2], 2);
    if (Crc16CcittFalse(rx_buf, REQ_PACKET_LEN - 2) != received_crc) {
        return false;
    }

    memcpy(cycle_num, &rx_buf[2], 2);
    memcpy(window, &rx_buf[4], SOH_INPUT_LEN * 4);
    return true;
}

void Uart_SendResponse(uint16_t cycle_num, float soh_pred) {
    uint8_t tx_buf[RESP_PACKET_LEN];

    tx_buf[0] = RESP_HEADER_0;
    tx_buf[1] = RESP_HEADER_1;
    memcpy(&tx_buf[2], &cycle_num, 2);
    memcpy(&tx_buf[4], &soh_pred, 4);

    uint16_t crc = Crc16CcittFalse(tx_buf, RESP_PACKET_LEN - 2);
    memcpy(&tx_buf[8], &crc, 2);

    HAL_UART_Transmit(&huart2, tx_buf, RESP_PACKET_LEN, UART_TIMEOUT_MS);
}
