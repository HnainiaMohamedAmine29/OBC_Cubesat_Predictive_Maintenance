#ifndef APP_INFERENCE_H_
#define APP_INFERENCE_H_

#ifdef __cplusplus
extern "C" {
#endif

void debug_log_printf(const char* s);
int  App_Inference_Init(void);
void App_Inference_RunOnce(void);

#ifdef __cplusplus
}
#endif

#endif
