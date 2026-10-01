// app_uart_protocol.h
#ifndef APP_UART_PROTOCOL_H_
#define APP_UART_PROTOCOL_H_

#include <stdint.h>
#include <stdbool.h>
#include "tflm_c_api.h"

#define REQ_HEADER_0   0xAA
#define REQ_HEADER_1   0x55
#define RESP_HEADER_0  0xBB
#define RESP_HEADER_1  0x66

#define REQ_PACKET_LEN   (2 + 2 + SOH_INPUT_LEN * 4 + 2)   // 2410 bytes
#define RESP_PACKET_LEN  (2 + 2 + 4 + 2)                    // 12 bytes

bool Uart_ReceiveRequest(uint16_t *cycle_num, float *window);
void Uart_SendResponse(uint16_t cycle_num, float soh_pred);

#endif // APP_UART_PROTOCOL_H_
