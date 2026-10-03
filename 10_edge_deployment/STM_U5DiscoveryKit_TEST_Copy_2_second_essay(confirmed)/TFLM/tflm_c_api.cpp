// tflm_c_api.cpp
#include "tflm_c_api.h"
#include "usart.h"
#include <cstring>
#include <cstdio>
#include <cmath>
#include <cstdint>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"
#include "tensorflow/lite/micro/cortex_m_generic/debug_log_callback.h"

extern "C" void debug_log_printf(const char* s) {
    printf("%s", s);   // route through the same semihosting path as app printf()
}

namespace {

// >>> EDIT: your Section 0 measured value + 10% headroom <<<
constexpr int kTensorArenaSize = 650 * 1024;   // REPLACE before the full build
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

constexpr int N_OPS = 25;   // 24 registered ops + 1 spare

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

alignas(alignof(tflite::MicroInterpreter)) uint8_t interpreter_buf[sizeof(tflite::MicroInterpreter)];

// ---- Quantization parameters of the model's input / output tensors ----
// The v3 model is fully INT8 (see optimize_v3_int8.py: inference_input_type
// and inference_output_type are tf.int8). MATLAB still sends float32; the
// conversion float <-> int8 happens here.
bool  in_is_int8  = false;
bool  out_is_int8 = false;
float in_scale_inv = 1.0f;   // 1 / input scale
int32_t in_zp      = 0;
float out_scale    = 1.0f;
int32_t out_zp     = 0;

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
    TRY_ADD(resolver.AddSoftmax());
    TRY_ADD(resolver.AddElu());
    TRY_ADD(resolver.AddLogistic());
    TRY_ADD(resolver.AddSplit());
    TRY_ADD(resolver.AddTanh());
    TRY_ADD(resolver.AddQuantize());
    TRY_ADD(resolver.AddDequantize());
    TRY_ADD(resolver.AddPack());
    TRY_ADD(resolver.AddUnpack());
    TRY_ADD(resolver.AddShape());
    TRY_ADD(resolver.AddStridedSlice());
    TRY_ADD(resolver.AddFill());
    TRY_ADD(resolver.AddSum());
    TRY_ADD(resolver.AddGather());
    TRY_ADD(resolver.AddConcatenation());

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
        (input->type != kTfLiteInt8 && input->type != kTfLiteFloat32)) {
        MicroPrintf("Unexpected input tensor shape/type");
        return TFLM_ERR_BAD_TENSOR;
    }
    if (output->type != kTfLiteInt8 && output->type != kTfLiteFloat32) {
        MicroPrintf("Unexpected output tensor type");
        return TFLM_ERR_BAD_TENSOR;
    }

    in_is_int8  = (input->type  == kTfLiteInt8);
    out_is_int8 = (output->type == kTfLiteInt8);
    if (in_is_int8) {
        if (input->params.scale <= 0.0f) { MicroPrintf("Bad input scale"); return TFLM_ERR_BAD_TENSOR; }
        in_scale_inv = 1.0f / input->params.scale;
        in_zp        = input->params.zero_point;
    }
    if (out_is_int8) {
        out_scale = output->params.scale;
        out_zp    = output->params.zero_point;
    }
    printf("Input : %s", in_is_int8 ? "INT8" : "FLOAT32");
    if (in_is_int8)  printf("  scale=%.8f zp=%d", (double)input->params.scale, (int)in_zp);
    printf("\r\nOutput: %s", out_is_int8 ? "INT8" : "FLOAT32");
    if (out_is_int8) printf("  scale=%.8f zp=%d", (double)out_scale, (int)out_zp);
    printf("\r\n");

    return TFLM_OK;
}

extern "C" tflm_status_t tflm_infer(const float *window, float *soh_out) {
    if (interpreter == nullptr) {
        return TFLM_ERR_INIT_FAILED;
    }

    if (in_is_int8) {
        // q = round(x / scale) + zero_point, saturated to int8
        for (int i = 0; i < SOH_INPUT_LEN; i++) {
            int32_t q = (int32_t)lroundf(window[i] * in_scale_inv) + in_zp;
            if (q >  127) q =  127;
            if (q < -128) q = -128;
            input->data.int8[i] = (int8_t)q;
        }
    } else {
        for (int i = 0; i < SOH_INPUT_LEN; i++) {
            input->data.f[i] = window[i];
        }
    }

    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        MicroPrintf("Invoke() failed");
        return TFLM_ERR_INVOKE_FAILED;
    }

    if (out_is_int8) {
        *soh_out = (float)((int32_t)output->data.int8[0] - out_zp) * out_scale;
    } else {
        *soh_out = output->data.f[0];
    }
    return TFLM_OK;
}

extern "C" uint32_t tflm_arena_used(void) {
    return interpreter ? (uint32_t)interpreter->arena_used_bytes() : 0u;
}
