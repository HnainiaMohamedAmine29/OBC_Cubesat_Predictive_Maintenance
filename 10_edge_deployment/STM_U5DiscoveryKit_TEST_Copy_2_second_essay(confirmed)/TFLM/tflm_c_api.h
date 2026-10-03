// tflm_c_api.h
#ifndef TFLM_C_API_H_
#define TFLM_C_API_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SOH_WINDOW_LEN  30
#define SOH_N_FEATURES  20
#define SOH_INPUT_LEN   (SOH_WINDOW_LEN * SOH_N_FEATURES)   // 600 floats

typedef enum {
    TFLM_OK = 0,
    TFLM_ERR_INIT_FAILED,
    TFLM_ERR_ALLOC_FAILED,
    TFLM_ERR_INVOKE_FAILED,
    TFLM_ERR_BAD_TENSOR
} tflm_status_t;

tflm_status_t tflm_init(void);
tflm_status_t tflm_infer(const float *window, float *soh_out);
uint32_t tflm_arena_used(void);   /* bytes of the tensor arena actually used (valid after tflm_init) */

#ifdef __cplusplus
}
#endif

#endif // TFLM_C_API_H_
