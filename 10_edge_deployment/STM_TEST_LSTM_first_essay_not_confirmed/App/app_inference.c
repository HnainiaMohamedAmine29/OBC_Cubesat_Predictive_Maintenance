// app_inference.c
// Uses LD2/PA5 (this board's user LED).

#include "app_inference.h"
#include "tflm_c_api.h"
#include "app_uart_protocol.h"
#include "main.h"
#include <stdio.h>

#define SOH_WARNING_THRESHOLD 0.80f

int App_Inference_Init(void) {
    tflm_status_t status = tflm_init();
    if (status != TFLM_OK) {
        printf("TFLM init failed with code %d\r\n", (int)status);
        return -1;
    }
    printf("TFLM initialized OK. Tensor arena allocated.\r\n");
    return 0;
}

void App_Inference_RunOnce(void) {
    static float window[SOH_INPUT_LEN];
    uint16_t cycle_num;

    if (!Uart_ReceiveRequest(&cycle_num, window)) {
        return;
    }

    float soh_pred = 0.0f;
    tflm_status_t status = tflm_infer(window, &soh_pred);
    if (status != TFLM_OK) {
        printf("Inference failed (cycle %u) with code %d\r\n", cycle_num, (int)status);
        return;
    }

    printf("cycle %u -> predicted SOH = %.4f\r\n", cycle_num, soh_pred);

    Uart_SendResponse(cycle_num, soh_pred);

    if (soh_pred < SOH_WARNING_THRESHOLD) {
        HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_SET);
    } else {
        HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_RESET);
    }
}
