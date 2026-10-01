// tflm_c_api.cpp
#include "tflm_c_api.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"

namespace {

// >>> EDIT: your Section 0 measured value + 10% headroom <<<
constexpr int kTensorArenaSize = 75 * 1024;   // REPLACE before the full build
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

constexpr int N_OPS = 17;

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
    	printf("Schema mismatch: model=%d expected=%d\r\n",
    	                   model->version(), TFLITE_SCHEMA_VERSION);;
        return TFLM_ERR_INIT_FAILED;
    }

    static tflite::MicroMutableOpResolver<N_OPS> resolver;
    TfLiteStatus s = kTfLiteOk;
    if ((s = resolver.AddReshape()) != kTfLiteOk) goto op_fail;
    if ((s = resolver.AddUnpack()) != kTfLiteOk) goto op_fail;
    if ((s = resolver.AddFullyConnected()) != kTfLiteOk) goto op_fail;
    if ((s = resolver.AddSplit()) != kTfLiteOk) goto op_fail;
    if ((s = resolver.AddLogistic()) != kTfLiteOk) goto op_fail;
    if ((s = resolver.AddTanh()) != kTfLiteOk) goto op_fail;
    if ((s = resolver.AddMul()) != kTfLiteOk) goto op_fail;
    if ((s = resolver.AddAdd()) != kTfLiteOk) goto op_fail;
    if ((s = resolver.AddPack()) != kTfLiteOk) goto op_fail;
    if ((s = resolver.AddMean()) != kTfLiteOk) goto op_fail;
    if ((s = resolver.AddNeg()) != kTfLiteOk) goto op_fail;
    if ((s = resolver.AddSquaredDifference()) != kTfLiteOk) goto op_fail;
    if ((s = resolver.AddRsqrt()) != kTfLiteOk) goto op_fail;
    if ((s = resolver.AddTranspose()) != kTfLiteOk) goto op_fail;
    if ((s = resolver.AddBatchMatMul()) != kTfLiteOk) goto op_fail;
    if ((s = resolver.AddSoftmax()) != kTfLiteOk) goto op_fail;
    if ((s = resolver.AddElu()) != kTfLiteOk) goto op_fail;
    goto op_ok;

    op_fail:
    MicroPrintf("Op resolver registration failed");
    return TFLM_ERR_INIT_FAILED;

    op_ok:

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
