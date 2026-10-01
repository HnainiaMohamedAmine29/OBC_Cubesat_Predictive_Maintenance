// tflm_c_api.cpp
#include "tflm_c_api.h"
#include "usart.h"
#include <cstring>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"
#include "tensorflow/lite/micro/cortex_m_generic/debug_log_callback.h"

extern "C" void debug_log_printf(const char* s) {
    HAL_UART_Transmit(&huart1, (uint8_t*)s, strlen(s), HAL_MAX_DELAY);
}

namespace {

// >>> EDIT: your Section 0 measured value + 10% headroom <<<
constexpr int kTensorArenaSize = 300 * 1024;   // REPLACE before the full build
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

constexpr int N_OPS = 19;

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

alignas(alignof(tflite::MicroInterpreter)) uint8_t interpreter_buf[sizeof(tflite::MicroInterpreter)];

}  // namespace

extern "C" tflm_status_t tflm_init(void) {

    tflite::InitializeTarget();

    model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
    	printf("Schema mismatch: model=%lu expected=%lu\r\n",
    	       (unsigned long)model->version(), (unsigned long)TFLITE_SCHEMA_VERSION);
        return TFLM_ERR_INIT_FAILED;
    }

    static tflite::MicroMutableOpResolver<N_OPS> resolver;   // was N_OPS=21, now 19
    TfLiteStatus s = kTfLiteOk;

    #define TRY_ADD(call) \
    s = call; \
    if (s != kTfLiteOk) { \
        MicroPrintf("FAILED registering: " #call); \
        return TFLM_ERR_INIT_FAILED; \
      }

    TRY_ADD(resolver.AddReshape());
    TRY_ADD(resolver.AddMean());
    TRY_ADD(resolver.AddNeg());
    TRY_ADD(resolver.AddSquaredDifference());
    TRY_ADD(resolver.AddAdd());
    TRY_ADD(resolver.AddRsqrt());
    TRY_ADD(resolver.AddMul());
    TRY_ADD(resolver.AddFullyConnected());
    TRY_ADD(resolver.AddTranspose());
    TRY_ADD(resolver.AddBatchMatMul());
    TRY_ADD(resolver.AddSoftmax());
    TRY_ADD(resolver.AddElu());
    TRY_ADD(resolver.AddLogistic());
    TRY_ADD(resolver.AddSplit());
    TRY_ADD(resolver.AddTanh());
    TRY_ADD(resolver.AddQuantize());
    TRY_ADD(resolver.AddDequantize());
    TRY_ADD(resolver.AddPack());
    TRY_ADD(resolver.AddUnpack());
    interpreter = new (interpreter_buf) tflite::MicroInterpreter(
        model, resolver, tensor_arena, kTensorArenaSize);

    TfLiteStatus alloc_status = interpreter->AllocateTensors();
    if (alloc_status != kTfLiteOk) {
        MicroPrintf("AllocateTensors() failed - current arena = %d bytes",
                    kTensorArenaSize);
        return TFLM_ERR_ALLOC_FAILED;
    }

    MicroPrintf("Arena used bytes: %d / %d",
                (int)interpreter->arena_used_bytes(), kTensorArenaSize);

    input  = interpreter->input(0);
    output = interpreter->output(0);

    if (input->dims->size != 3 ||
        input->dims->data[1] != SOH_WINDOW_LEN ||
        input->dims->data[2] != SOH_N_FEATURES ||
        input->type != kTfLiteFloat32) {
        MicroPrintf("Unexpected input tensor shape/type");
        return TFLM_ERR_BAD_TENSOR;
    }

    return TFLM_OK;
}

extern "C" tflm_status_t tflm_infer(const float *window, float *soh_out) {
    if (interpreter == nullptr) {
        return TFLM_ERR_INIT_FAILED;
    }

    for (int i = 0; i < SOH_INPUT_LEN; i++) {
        input->data.f[i] = window[i];
    }

    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        MicroPrintf("Invoke() failed");
        return TFLM_ERR_INVOKE_FAILED;
    }

    *soh_out = output->data.f[0];
    return TFLM_OK;
}
