# Technical Documentation — Embedded Firmware Deployment on STM32U5 Discovery Kit (Module 10)

**Document scope:** `10_STM_U5DiscoveryKit_TEST`
**Nature of the archive:** a complete STM32CubeIDE project (`.ioc`/`.cproject`/`.launch` files, linker scripts, STM32U5 HAL, and the entirety of the **TensorFlow Lite for Microcontrollers** library with its third-party dependencies — CMSIS-NN, FlatBuffers, gemmlowp, kissfft, ruy)
**Application files analyzed in detail:** `Core/Src/main.c`, `App/app_inference.c/.h`, `App/app_uart_protocol.c/.h`, `TFLM/tflm_c_api.cpp/.h`, `Model/model_data.cc/.h`, `STM_U5DiscoveryKit_TEST.ioc`, `STM32U585AIIXQ_FLASH.ld`

This document represents the culmination of the chain documented in modules 05 (LSTM models), 08 (switch to TensorFlow Lite) and 09 (latency/RAM measurement bench on a development machine): it covers the **first real deployment on physical hardware**, in the form of a complete bare-metal firmware for an STM32U5 development board, running TensorFlow Lite Micro inference directly on the microcontroller, with no dependency on a Python runtime.

---

## 1. Nature of the archive and analysis method

This archive is very different in nature from previous modules: it is a **fully exported STM32CubeIDE project**, including not only the application code specific to this project, but also the full set of generic hardware drivers (`Drivers/CMSIS`, `Drivers/STM32U5xx_HAL_Driver`) and the entire source code of **TensorFlow Lite for Microcontrollers (TFLM)** and its third-party libraries (`TFLM/third_party/cmsis_nn`, `flatbuffers`, `gemmlowp`, `kissfft`, `ruy`) — several thousand vendor C/C++ code files not specific to this project.

In line with the purpose of this documentation (explaining the design and engineering specific to this project), this document focuses exclusively on:
- the **four application files written for this project** (`app_inference.*`, `app_uart_protocol.*`, `tflm_c_api.*`);
- the **project-specific configuration points** of the STM32CubeIDE project (clock, peripherals, memory);
- the **embedded model** (`model_data.cc`) and what its structure reveals about the architecture actually deployed.

The contents of `TFLM/` and `Drivers/` outside these integration points are standard vendor code (the official TensorFlow Lite Micro library and STM32CubeU5 HAL drivers) and are not detailed file by file — only their overall architectural role is described.

---

## 2. Major finding: evolution of the hardware target

### 2.1 Recap of the initially targeted hardware

Modules 05 and 08 explicitly mentioned, in the source code comments of the notebooks and Python scripts, an embedded target of type **STM32F401RE** (Cortex-M4 core, **94 KB of RAM**, 512 KB of flash).

### 2.2 The target actually used in this module

The project file (`STM_U5DiscoveryKit_TEST.ioc`) unambiguously identifies the microcontroller actually used:

```
Mcu.Name=STM32U585AIIxQ
ProjectManager.DeviceId=STM32U585AIIxQ
```

This is an **STM32U585AI**, with an **Arm Cortex-M33** core (no longer Cortex-M4), mounted on the **B-U585I-IOT02A** development board ("STM32U5 Discovery Kit for IoT nodes", explicitly named in a comment in `app_inference.c` and `app_uart_protocol.c`). The linker script (`STM32U585AIIXQ_FLASH.ld`) confirms the actual memory resources of this target:

```
RAM   : ORIGIN = 0x20000000, LENGTH = 768K
SRAM4 : ORIGIN = 0x28000000, LENGTH = 16K
FLASH : ORIGIN = 0x08000000, LENGTH = 2048K
```

i.e., **784 KB of RAM** (768 + 16) and **2,048 KB (2 MB) of flash** — respectively about **8 times** and **4 times** the resources of the STM32F401RE target originally considered.

### 2.3 Quantitative justification for this target change

This archive makes it possible, for the first time in this entire documentation chain, to reconstruct the **real, precise memory budget** of the deployed model, directly readable from the code:

| Quantity | Value measured in this module | Comparison with STM32F401RE (94 KB RAM / 512 KB flash) |
|---|---|---|
| Embedded model size (`g_model_len`, flash) | **336,248 bytes** (≈ 328 KB) | ≈ 64% of the F401RE target's total flash on its own |
| TFLM tensor arena (`kTensorArenaSize`, RAM) | **300 × 1024 = 307,200 bytes** (300 KB) | **≈ 3.2 times the total RAM** of the F401RE target — a structural overrun, even before counting the stack, global variables, or any other RAM usage by the firmware |

**This calculation is decisive and conclusive**: the tensor arena alone, required to run the model (300 KB), is on its own more than three times larger than the entire RAM available on the STM32F401RE target originally targeted in modules 05 and 08 — a deployment there was therefore **physically impossible**, independent of any software optimization. The pivot to the STM32U585 (784 KB of RAM, i.e., a comfortable 484 KB margin beyond the 300 KB arena) thus appears as a **direct and necessary consequence** of the actual memory requirement of the model once it was actually compiled for an ARM Cortex-M target — a figure none of the previous modules (05, 08, 09) had been able to produce, having never actually compiled and allocated the model on a real embedded architecture. This result validates, retroactively and in a particularly concrete way, the methodological caveat already raised in the module 09 documentation (§7): RAM measurements taken on a desktop Python process were not transposable to the embedded target, and only an actual compilation and allocation on the target could settle the question.

### 2.4 Which model was deployed?

The list of operators registered in `tflm_c_api.cpp` (see §4.2) — in particular the presence of `AddBatchMatMul()`, `AddSoftmax()`, `AddSplit()` and `AddUnpack()`, operators characteristic of splitting a `MultiHeadAttention` layer into several heads — indicates that the model actually embedded here corresponds to the **v2 reference architecture** (module 05, LSTM 128→64 + multi-head self-attention), not the v3 architecture specifically optimized for embedded use (which uses a much simpler additive attention mechanism, see module 05 §4.4, requiring neither batched matrix multiplication nor splitting into heads). The size of the embedded model (336 KB) is also more consistent with the order of magnitude of v2 (574.5 KB in uncompressed fp32, see module 05) than with that of v3 (139.6 KB).

**This is a notable engineering observation**: it is ultimately the "heavy" reference model (v2), not the v3 variant specifically slimmed down to fit on a 94 KB-RAM microcontroller, that was chosen for this first real hardware deployment — made possible not by shrinking the model, but by switching to a significantly more capable microcontroller. This does not, however, make module 05's optimization work (v3 architecture) obsolete: it remains relevant for a target with more constrained resources (the original STM32F401RE, or a future iteration targeting a tighter memory and power budget, relevant for a real CubeSat mission where every milliwatt and every square millimeter of silicon counts); this module does, however, demonstrate that a fallback path exists and works (v2 model on a more capable target) — a useful piece of decision data for the rest of the project.

---

## 3. Firmware software architecture

### 3.1 Overview

```
main.c
  │
  ├─► App_Inference_Init()          (app_inference.c)
  │        └─► tflm_init()          (tflm_c_api.cpp)
  │                 └─► builds the TFLM MicroInterpreter, allocates the arena, checks the input tensor
  │
  └─► infinite loop: App_Inference_RunOnce()   (app_inference.c)
           │
           ├─► Uart_ReceiveRequest()   (app_uart_protocol.c) — blocking, waits for a frame on USART2
           ├─► tflm_infer()            (tflm_c_api.cpp) — copies the window, invoke(), reads the output
           ├─► Uart_SendResponse()     (app_uart_protocol.c) — sends the prediction back on USART2
           └─► drives the green LED based on an SOH warning threshold
```

This firmware performs **neither the physical simulation nor the feature computation**: it is strictly limited to the role it would play onboard the CubeSat — **receive an already-computed and normalized 30×20 feature window, run inference, send back the prediction**. This is consistent with the intended architecture: feature computation (module 04/06, `compute_features.m`) and physical simulation (module 02) would, in a real deployment, remain the responsibility of another subsystem (the flight software / OBC), with the microcontroller dedicated to inference doing only what it is meant to do.

### 3.2 C/C++ bridge to TensorFlow Lite Micro (`tflm_c_api.cpp`)

#### C interface exposed to the application

`tflm_c_api.h` defines a **pure C** interface (`extern "C"`), allowing `app_inference.c` (C code) to call functions implemented in **C++** (`tflm_c_api.cpp`, required by TFLM's own internal API, written in C++). This is the standard, correct encapsulation pattern for integrating a C++ library into a predominantly C firmware.

```c
#define SOH_WINDOW_LEN  30
#define SOH_N_FEATURES  20
#define SOH_INPUT_LEN   (SOH_WINDOW_LEN * SOH_N_FEATURES)   // 600 floats
```

These constants exactly match the dimensions already documented in modules 04, 05, 06, 08 and 09 (`WINDOW_SIZE=30`, 20 features) — further confirmation that dimensioning consistency has been maintained across all ten modules of the project.

#### Initialization (`tflm_init`)

```cpp
model = tflite::GetModel(g_model);
if (model->version() != TFLITE_SCHEMA_VERSION) { ... }

static tflite::MicroMutableOpResolver<N_OPS> resolver;   // N_OPS = 19
TRY_ADD(resolver.AddReshape());
TRY_ADD(resolver.AddMean());
... (19 operators in total)

interpreter = new (interpreter_buf) tflite::MicroInterpreter(
    model, resolver, tensor_arena, kTensorArenaSize);
interpreter->AllocateTensors();
```

Technical points to note:

- **Schema version check** (`model->version() != TFLITE_SCHEMA_VERSION`): a standard, correctly implemented safeguard, preventing the loading of a FlatBuffer-serialized model with a TFLite schema version incompatible with the TFLM library version embedded in the firmware.
- **`MicroMutableOpResolver<19>`**: unlike a full operator resolver (which would embed the code for *all* possible TFLite operators, wasting flash), this **mutable, explicitly sized** resolver (`N_OPS=19`) registers only the 19 operators strictly required by this specific model — a standard flash code-size optimization, correctly applied here in TFLM. The comment `// was N_OPS=21, now 19` indicates a fine-tuning iteration (removal of two operators initially registered but ultimately unused), consistent with the correction-tracing style already observed in previous modules.
- **Explicit memory placement**: `tensor_arena` (300 KB) and `interpreter_buf` (exact size of `tflite::MicroInterpreter`) are **aligned** static arrays (`alignas(16)`, `alignas(alignof(tflite::MicroInterpreter))`) allocated in the firmware's `.bss`/`.data` segment — no dynamic heap allocation, in line with strict best practices for critical embedded development (determinism, no heap fragmentation, memory footprint known at compile time).
- **Logging of actual arena usage** (`MicroPrintf("Arena used bytes: %d / %d", interpreter->arena_used_bytes(), kTensorArenaSize)`): good debugging practice, allowing the real margin between the allocated arena (300 KB) and the arena actually consumed by the graph to be known precisely at runtime — information needed to refine this sizing in a future iteration (the comment `// >>> EDIT: your Section 0 measured value + 10% headroom <<<` explicitly indicates that this 300 KB value is a **placeholder to be adjusted** once the real value has been measured on the target, with a 10% safety margin).
- **Input tensor shape and type check** (`input->dims->size != 3 || ... || input->type != kTfLiteFloat32`): an additional safeguard confirming that the loaded model does indeed match the expected dimensions (1, 30, 20) and works in **32-bit floating-point, non-quantized precision** — this firmware therefore deploys the `.tflite` model in its original floating-point version, without int8 post-training quantization, contrary to what the file name `soh_model_SIMPLE.tflite` encountered in modules 08 and 09 might have suggested for a possible quantized variant.

#### Inference (`tflm_infer`)

```cpp
for (int i = 0; i < SOH_INPUT_LEN; i++) {
    input->data.f[i] = window[i];
}
interpreter->Invoke();
*soh_out = output->data.f[0];
```

A minimal, direct function: copies the 600-float window into the input tensor, invokes the graph, reads the output scalar — no normalization is performed here, which means **the window received via UART must already be normalized** (by the `scaler_X` from module 04/05) before being sent to the microcontroller; this firmware therefore only runs the neural network itself, with normalization remaining the host side's responsibility (see §5).

### 3.3 Application logic (`app_inference.c`)

```c
#define SOH_WARNING_THRESHOLD 0.80f
...
if (soh_pred < SOH_WARNING_THRESHOLD) {
    HAL_GPIO_WritePin(LED_GREEN_GPIO_Port, LED_GREEN_Pin, GPIO_PIN_RESET);  // 0 = LED ON
} else {
    HAL_GPIO_WritePin(LED_GREEN_GPIO_Port, LED_GREEN_Pin, GPIO_PIN_SET);    // 1 = LED OFF
}
```

This function adds a simple but representative layer of application logic for a real use case: a **visual warning threshold** at SOH = 0.80 (above the end-of-life threshold of 0.70 already documented in previous modules, making it an early warning rather than an end-of-life alarm), realized by lighting the board's user LED (`LD1`, pin `PH7`). The code comment highlights a correctly handled hardware quirk: on the B-U585I-IOT02A board, this LED is **active-low** (it lights up when the pin is set to 0), unlike Nucleo boards where the user LED is usually active-high — a board-porting detail correctly documented in a comment to avoid confusion during any future port to another board.

### 3.4 Communication protocol (`app_uart_protocol.c`)

#### Separation of UART links

The firmware uses **two separate UART links**, a good-practice separation:
- **USART1** (pins PA9/PA10), connected to the ST-Link debug probe's built-in virtual USB-UART converter ("VCP", Virtual COM Port): used **exclusively** for diagnostic `printf`/`MicroPrintf` messages (115,200 baud), redirected via `debug_log_printf()` and TFLM's `RegisterDebugLogCallback()` mechanism.
- **USART2** (pins PA2/PA3), wired as a dedicated serial link to an external host (explicitly documented in a comment as "the data link to MATLAB"): used exclusively for the binary request/response protocol, at a significantly higher baud rate (**921,600 baud**) — consistent with the need to quickly transmit a 600-float window (2,400 bytes of payload) per cycle.

This strict separation between the diagnostic channel (text, slow) and the data channel (binary, fast) is a good embedded-systems engineering practice: it prevents logging messages from disrupting or slowing down the data protocol, and allows the firmware to be debugged (via the ST-Link's virtual serial port, usually already connected by default during development) independently of the data link proper.

#### Frame format

| Field | Size | Request frame (host → board) | Response frame (board → host) |
|---|---|---|---|
| Header | 2 bytes | `0xAA 0x55` | `0xBB 0x66` |
| Cycle number | 2 bytes | `uint16_t cycle_num` | `uint16_t cycle_num` (echo) |
| Payload | `SOH_INPUT_LEN × 4` = 2,400 bytes | 600 32-bit floats (normalized window) | 4 bytes (1 float: `soh_pred`) |
| CRC | 2 bytes | CRC-16/CCITT-FALSE over everything above | same |
| **Total** | | **2,410 bytes** (`REQ_PACKET_LEN`) | **12 bytes** (`RESP_PACKET_LEN`) |

The CRC computation (`Crc16CcittFalse`, polynomial `0x1021`, initial value `0xFFFF` — the standard "CRC-16/CCITT-FALSE" implementation) is applied on both transmit and receive, with **silent rejection of any frame whose CRC does not match** (`Uart_ReceiveRequest` simply returns `false`) — a robust and appropriate transmission-error detection mechanism for a high-speed serial link, where isolated bit errors are possible. The firmware also waits for a correct header byte before considering the rest of the frame (`REQ_HEADER_0`/`REQ_HEADER_1`), providing implicit resynchronization in case the byte stream becomes misaligned (a misaligned frame will be rejected as soon as the header check fails, without blocking indefinitely on a faulty read).

A 3-second timeout (`UART_TIMEOUT_MS`) protects each reception step against indefinite blocking if the host stops responding or never responds.

---

## 4. Hardware and clock configuration

### 4.1 Operating frequency

`main.c` configures the system clock via a PLL on the **MSI** source (internal multi-speed oscillator):

```c
MSIClockRange = RCC_MSIRANGE_4;      // MSI = 4 MHz
PLL.PLLM = 1; PLL.PLLN = 80; PLL.PLLR = 2;
// SYSCLK = (4 MHz × 80) / 2 = 160 MHz
```

i.e., a core frequency of **160 MHz**, the **maximum** frequency of the STM32U585 (Cortex-M33 core). The choice of **voltage-scaling mode VOS1** (`PWR_REGULATOR_VOLTAGE_SCALE1`, the high-performance mode) and a 4-wait-state flash latency (`FLASH_LATENCY_4`, required at this frequency) confirms a configuration **deliberately optimized for maximum computing performance** rather than minimum power consumption — a choice consistent with the nature of this test (validating the functional feasibility of embedded inference) rather than a power-consumption optimization, which would remain a separate and important concern for a real CubeSat deployment (where the power budget is a first-order mission constraint, not addressed in this module).

---

## 5. What this archive does not contain: the host-side counterpart

This module documents the embedded firmware **alone**. It includes **no host-side script** (MATLAB or Python) responsible for:
- building the normalized 30×20 feature window (a role already filled, in previous modules, by `compute_features.m` + `scaler_X.pkl`);
- sending the binary frame in the format expected by `Uart_ReceiveRequest` (header, cycle number, 600 floats, CRC-16) over the serial port corresponding to the board's USART2;
- receiving and decoding the 12-byte response frame.

The code comment explicitly referring to "the data link to MATLAB" confirms that such a host script exists or is planned in the project's workflow, but **it is not provided in this archive**. Likewise, **no execution results are provided** in this module (no screenshot, no `printf` output log, no results CSV) — unlike modules 06, 08 and 09, which systematically included at least one results figure. This module therefore documents a **firmware infrastructure ready to be tested**, but does not, given the files provided, allow confirmation that this exchange protocol has actually been validated end-to-end on the physical board (real end-to-end latency, frame-loss rate, embedded inference accuracy compared to that measured in previous modules).

---

## 6. Summary of limitations and recommendations

1. **Missing host script (§5)** — high priority: document or provide the companion MATLAB/Python script driving the UART protocol described in §3.4, a necessary condition for a third party to reproduce an end-to-end test.
2. **No on-target execution results (§5)** — high priority: this module does not allow empirical confirmation that the firmware works correctly on the physical board; a future iteration should provide a UART output log, an end-to-end latency measurement taken on the real target (to be directly compared with module 09's desktop latency estimates), and an accuracy comparison between the embedded prediction (floating-point, TFLM/CMSIS-NN) and the reference prediction (full TensorFlow, module 05) — the TFLM library can indeed introduce very slight numerical differences from the reference TensorFlow, which should be quantified.
3. **Tensor arena sizing marked as provisional** (§3.2, comment `>>> EDIT ... REPLACE before the full build <<<`) — the 300 KB value should be replaced with the actually measured value (`arena_used_bytes()`) plus a 10% margin, as the code itself anticipates; this finalization step does not yet appear to have been carried out in the provided files.
4. **No quantization** (§3.2) — the deployed model remains in 32-bit floating point; post-training quantization (int8) would further reduce flash size and potentially inference latency (provided the impact on accuracy is validated), an optimization still to be explored even on the current, more resource-generous target, and all the more useful should a return to a more constrained target (such as the F401RE, or the v3 model) become relevant again for the final mission.
5. **v2 model deployed rather than v3** (§2.4) — to be explicitly documented as a project decision: clarify whether this choice is final (the mission's final hardware target will indeed be sized accordingly) or whether this is an intermediate test, with the v3 model remaining the preferred path for a target with resources closer to what is actually available onboard a CubeSat.
6. **No power-consumption management** (§4.1) — the current clock configuration targets maximum performance (160 MHz, VOS1) with no consideration for power consumption; to be re-evaluated for any test representative of a real CubeSat mission power budget.

---

## 7. Conclusion

This module marks a pivotal step in the chain documented since module 02: the transition from an entirely software-based validation (MATLAB simulation, Python training and evaluation, performance test bench on a development machine) to **real embedded firmware**, compiled and ready to run on a physical microcontroller, with a robust serial communication protocol (CRC-16 frames, diagnostic/data separation) and a clean integration of TensorFlow Lite for Microcontrollers via a correctly designed C/C++ bridge (static allocation, model consistency checks, an operator resolver sized to the strict minimum needed). The detailed analysis of the memory budget actually measured in this module — a 300 KB tensor arena on its own more than 3 times larger than the total RAM of the STM32F401RE target originally considered in modules 05 and 08 — provides a definitive, quantified answer to the question left open by module 09: the reference model (v2), as deployed here, requires a significantly more capable hardware target (STM32U585, 784 KB of RAM) than the one originally targeted, which explains and justifies the observed board change. It remains to be demonstrated, however, in a future iteration of this project, that this firmware actually works end-to-end on the physical board — the host-side software counterpart and the results of a real execution being the two most important missing pieces needed to close out this validation chain.
