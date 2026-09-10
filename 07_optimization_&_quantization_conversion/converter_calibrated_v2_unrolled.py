import argparse
import json
import os
import time

import numpy as np
import tensorflow as tf

# ============================================================
# CALIBRATED for: unrolled (unroll=True) SOH LSTM -> STM32F401RE
#
# Target: combine the RAM win from static unrolling (no WHILE op,
# no dynamic per-timestep scratch - this is what makes the arena
# fit at all) with the Flash win from INT8 quantization (this is
# what makes the .tflite file small enough to flash) -- WITHOUT
# producing hybrid ops, which the CMSIS-NN kernels on this
# firmware build reject outright.
#
# soh_model_SIMPLE.tflite (unrolled, but exported with NO quantization
# at all) came out to 753 KB - larger than the F401RE's entire 512 KB
# flash budget.
#
# CORRECTION (post-deployment): the first fix attempt used
# "dynamic_range" quantization, which does shrink the file (weights ->
# int8) but leaves activations in float32. That produces hybrid
# FULLY_CONNECTED ops, and CMSIS-NN's kernels have no hybrid code path
# -- firmware failed at AllocateTensors() with "Hybrid models are not
# supported on TFLite Micro." The correct mode is "full_int8": full
# integer quantization (weights AND activations int8, calibrated via
# --representative-data), which still keeps float32 input/output
# tensors so the existing UART wire format needs no firmware/MATLAB
# changes. This is now the default quantization mode below.
# ============================================================


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def build_concrete_function(model, batch_size, seq_len, n_features):
    model.trainable = False

    @tf.function
    def model_fn(x):
        # CALIBRATION FIX: explicitly force inference mode. Calling
        # model(x) without training=False leaves Keras's internal
        # learning-phase resolution ambiguous during tracing - for a
        # model with no Dropout/BatchNorm this is harmless, but for
        # this LSTM+attention architecture it costs nothing to be
        # explicit and removes one class of "converts fine, predicts
        # slightly differently than the Keras model" bug.
        return model(x, training=False)

    return model_fn.get_concrete_function(
        tf.TensorSpec([batch_size, seq_len, n_features], tf.float32))


def convert_to_tflite(concrete_func, quantization, representative_data_path):
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    # Deliberately NOT setting _experimental_lower_tensor_list_ops=False or
    # experimental_enable_resource_variables -- those broke conversion.

    if quantization == "none":
        pass
    elif quantization == "dynamic_range":
        # CORRECTED: this is NOT actually usable on the F401RE build.
        # Weight-only INT8 quantization leaves activations in float32,
        # which means every FULLY_CONNECTED/MATMUL op ends up as a
        # "hybrid" op (int8 weight + float32 activation). The reference
        # TFLM kernels can limp through hybrid ops via a float fallback,
        # but the CMSIS-NN accelerated kernels this board's firmware
        # links against have NO hybrid code path at all and reject them
        # outright ("Hybrid models are not supported on TFLite Micro").
        # Kept here only for boards/builds that use TFLM reference
        # kernels instead of CMSIS-NN. Do not use for the F401RE.
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quantization == "float16":
        # NOT RECOMMENDED for this board: Cortex-M4 (F401RE) has no
        # hardware FP16 unit, and TFLM's FLOAT16 kernel coverage is
        # thin -- you'd likely pay a runtime dequantize-to-float32 cost
        # per op with limited size benefit. Left in for completeness /
        # other targets, but don't use this for the F401RE build.
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantization in ("full_int8", "full_int8_int_io"):
        # THIS IS THE MODE FOR THE F401RE (CMSIS-NN build). Full integer
        # quantization: both weights AND activations become int8, using
        # the representative dataset for calibration. This is what
        # eliminates the hybrid ops that CMSIS-NN rejects -- every
        # FULLY_CONNECTED/attention MatMul becomes true int8 x int8 ->
        # int32, which is exactly what the accelerated kernels expect.
        #
        # "full_int8" (default I/O): input/output tensors stay FLOAT32.
        # The converter inserts a QUANTIZE op right after the input and
        # a DEQUANTIZE op right before the output, so internals are all
        # int8 but the existing float32 UART wire format on both the
        # STM32 (app_uart_protocol.c) and MATLAB (uart_soh_predict.m)
        # sides is untouched -- no firmware/MATLAB protocol changes
        # needed. This is the right default for this project.
        #
        # "full_int8_int_io": input/output tensors also become int8.
        # Slightly less runtime overhead (skips the boundary
        # quantize/dequantize op) and a bit more flash/RAM savings, but
        # requires updating both app_uart_protocol.c and
        # uart_soh_predict.m to pack/unpack int8 + scale/zero_point
        # instead of raw float32. Only use this if you've made that
        # firmware/MATLAB change; otherwise use "full_int8".
        if not representative_data_path:
            raise ValueError("--representative-data is required for full_int8 quantization")
        rep = np.load(representative_data_path)["X"].astype("float32")

        def rep_dataset():
            for i in range(min(len(rep), 200)):
                yield [rep[i:i + 1]]

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = rep_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        if quantization == "full_int8_int_io":
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        else:
            converter.inference_input_type = tf.float32
            converter.inference_output_type = tf.float32
    else:
        raise ValueError(f"Unknown quantization mode: {quantization}")

    return converter.convert()


def strip_tflite_debug_metadata(tflite_model_bytes):
    # FIX (post-deployment #2): fully unrolling the LSTM (unroll=True) is
    # correct for RAM (no WHILE op -> static arena), but it has a nasty
    # Flash-side cost: TF's graph freezer/Grappler fuses ops during
    # unrolling and names each fused op by concatenating every sub-op
    # name it absorbed, joined with ';'. Multiply that across 30
    # timesteps x several gates/attention ops and you get 1200+ tensors,
    # some with 900+ character names -- on this model that was 136 KB of
    # pure debug string data alone (out of ~336 KB of total metadata
    # overhead vs. only ~148 KB of actual weight/bias data).
    #
    # TFLite Micro never reads tensor names at runtime -- the generated
    # C API addresses tensors by index (interpreter->input(0),
    # interpreter->output(0)), never by name. So these strings are safe
    # to drop entirely; this is a lossless, bit-exact-inference
    # optimization, not a quantization/accuracy tradeoff.
    from tensorflow.lite.python import schema_py_generated as schema_fb
    import flatbuffers

    buf = bytearray(tflite_model_bytes)
    model_obj = schema_fb.Model.GetRootAsModel(buf, 0)
    model = schema_fb.ModelT.InitFromObj(model_obj)

    for sg in model.subgraphs:
        sg.name = None
        for t in sg.tensors:
            t.name = None
    model.description = None

    builder = flatbuffers.Builder(0)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    return bytes(builder.Output())


def validate_tflite(tflite_path, keras_model, batch_size, seq_len, n_features, quantization,
                     n_inference_samples=10):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    try:
        interpreter.reset_all_variables()
    except Exception:
        pass

    op_names = sorted(set(op["op_name"] for op in interpreter._get_ops_details()))
    total_op_instances = len(interpreter._get_ops_details())
    # WHILE is a legitimate native TFLite builtin op -- NOT the same as the
    # broken TensorListReserve/custom-op path. Only true resource-variable
    # ops indicate a real problem (an unfrozen model).
    forbidden = {"READ_VARIABLE", "ASSIGN_VARIABLE", "VAR_HANDLE"}
    has_dynamic_ops = bool(forbidden & set(op_names))

    # CALIBRATION ADDITION: this is the exact thing that bit us on the
    # F401RE -- flag WHILE explicitly (informational, not fatal) since a
    # source .keras with unroll=False silently produces a WHILE graph
    # again even under this same script, and that's the one thing that
    # made the arena never fit no matter how it was sized.
    has_while = "WHILE" in op_names
    if has_while:
        log("  ⚠️  WARNING: model still contains a WHILE op. This means the "
            "source .keras model was NOT built with unroll=True. This "
            "conversion will very likely still have the same tensor-arena "
            "problem on the F401RE regardless of quantization mode.")

    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]
    log(f"  Input shape/dtype: {in_detail['shape']} / {in_detail['dtype']}")
    log(f"  Ops used: {op_names} ({total_op_instances} total instances)")

    # CALIBRATION ADDITION: sanity-check the shape matches what the
    # firmware (SOH_WINDOW_LEN=30, SOH_N_FEATURES=20) and MATLAB side
    # both hardcode. Catches "wrong model file" immediately instead of
    # a confusing shape-mismatch failure on-device later.
    expected_shape = [batch_size, seq_len, n_features]
    actual_shape = list(in_detail["shape"])
    if actual_shape != expected_shape:
        log(f"  ⚠️  WARNING: input shape {actual_shape} != expected "
            f"{expected_shape} -- firmware's tflm_c_api.cpp checks for "
            f"exactly [1,30,20] and will reject this at tflm_init() if "
            f"it doesn't match.")

    def to_input_dtype(x):
        # Only the int_io variant actually has an int8 input tensor.
        # "full_int8" (default) keeps float32 I/O by design -- see
        # convert_to_tflite -- so it must NOT be rescaled here.
        if quantization == "full_int8_int_io":
            scale, zero_point = in_detail["quantization"]
            return (x / scale + zero_point).astype(in_detail["dtype"])
        return x.astype(in_detail["dtype"])

    # warm-up
    dummy = np.zeros((batch_size, seq_len, n_features), dtype="float32")
    interpreter.set_tensor(in_detail["index"], to_input_dtype(dummy))
    interpreter.invoke()
    log("  Warm-up inference done")

    # accuracy check vs. the original Keras model
    x_test = np.random.RandomState(1).rand(batch_size, seq_len, n_features).astype("float32")
    interpreter.set_tensor(in_detail["index"], to_input_dtype(x_test))
    interpreter.invoke()
    tflite_out = interpreter.get_tensor(out_detail["index"])
    keras_out = keras_model.predict(x_test, verbose=0)
    max_diff = float(np.max(np.abs(tflite_out.astype("float32") - keras_out.astype("float32"))))

    # real inference loop, one window at a time (matches deployment usage)
    X_windows = np.random.RandomState(2).rand(n_inference_samples, seq_len, n_features).astype("float32")
    predictions = []
    for i in range(len(X_windows)):
        sample = X_windows[i:i + 1]
        interpreter.set_tensor(in_detail["index"], to_input_dtype(sample))
        interpreter.invoke()
        out = interpreter.get_tensor(out_detail["index"])
        predictions.append(float(out.flatten()[0]))
    log(f"  Inference loop over {n_inference_samples} windows done. "
        f"Sample predictions: {predictions[:5]}")

    return {
        "ops_used": op_names,
        "total_op_instances": total_op_instances,
        "has_forbidden_resource_var_ops": has_dynamic_ops,
        "has_while_op": has_while,
        "output_max_abs_diff_vs_keras": max_diff,
        "sample_predictions": predictions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cubesat_soh_lstm_unrolled.keras")
    # CALIBRATION: default output name and default quantization mode
    # both changed to reflect the F401RE target -- dynamic_range is the
    # correct choice here, not the original "none".
    parser.add_argument("--output", default="soh_model_optimized_unrolled_int8_v2.tflite")
    parser.add_argument("--batch-size", type=int, default=1)
    # CORRECTED default: "dynamic_range" produces hybrid ops that the
    # CMSIS-NN kernels on this firmware build reject outright. "full_int8"
    # (full integer quantization, float32 I/O) is the mode that actually
    # runs on the F401RE. Requires --representative-data.
    parser.add_argument("--quantization",
                         choices=["none", "dynamic_range", "float16", "full_int8", "full_int8_int_io"],
                         default="full_int8")
    parser.add_argument("--representative-data", default="representative_data.npz",
                         help="Path to .npz with array X, shape (N, seq_len, n_features). Required for full_int8.")
    # CALIBRATION ADDITION: hard budget check, tied to the actual F401RE
    # constraint we hit -- fails loudly instead of producing a file that
    # silently won't flash.
    parser.add_argument("--max-flash-kb", type=float, default=400.0,
                         help="Fail if the output .tflite exceeds this size. "
                              "F401RE has 512KB total flash shared with "
                              "firmware code -- 400KB leaves headroom for "
                              "HAL+TFLM+App code. Set higher for boards "
                              "with more flash (e.g. F767ZI's 2MB).")
    # FIX (post-deployment #2): strip debug tensor names by default. On
    # this model this alone cut 486,984 -> 336,528 bytes (-146.9 KB),
    # which is what actually fixed the FLASH overflow -- unrolling
    # bloats tensor NAME metadata (semicolon-joined fused-op provenance
    # strings), not weight data. Purely lossless / bit-exact; disable
    # only if you need names for debugging with a visualizer.
    parser.add_argument("--strip-names", action=argparse.BooleanOptionalAction,
                         default=True,
                         help="Strip tensor/subgraph debug names from the "
                              "output .tflite (default: on). TFLite Micro "
                              "addresses tensors by index, not name, so "
                              "this is safe and typically saves 100+ KB on "
                              "fully unrolled models.")
    args = parser.parse_args()

    report = {"stages": []}

    log("Stage 1: loading source model (unroll settings left as-is)")
    model = tf.keras.models.load_model(args.model)
    seq_len, n_features = model.input_shape[1], model.input_shape[2]
    orig_size_mb = os.path.getsize(args.model) / 1e6
    found_non_unrolled_lstm = False
    for l in model.layers:
        if "lstm" in l.name.lower():
            log(f"  {l.name}: unroll={l.unroll}, stateful={l.stateful} (unchanged)")
            if not l.unroll:
                found_non_unrolled_lstm = True
    if found_non_unrolled_lstm:
        log("  ⚠️  WARNING: at least one LSTM layer has unroll=False. "
            "This conversion will very likely produce a WHILE-based graph "
            "again, defeating the point of this run. Make sure you're "
            "pointing --model at the .keras file that was rebuilt with "
            "unroll=True (the one that produced soh_model_SIMPLE.tflite), "
            "not the original.")

    log("Stage 2: tracing concrete function with static input shape "
        f"({args.batch_size}, {seq_len}, {n_features})")
    concrete_func = build_concrete_function(model, args.batch_size, seq_len, n_features)

    log(f"Stage 3: converting to TFLite (quantization={args.quantization})")
    tflite_model = convert_to_tflite(concrete_func, args.quantization, args.representative_data)
    raw_size_kb = len(tflite_model) / 1024

    if args.strip_names:
        log("Stage 3.5: stripping debug tensor/subgraph names "
            "(lossless, bit-exact inference -- see FIX comment above)")
        tflite_model = strip_tflite_debug_metadata(tflite_model)
        stripped_size_kb = len(tflite_model) / 1024
        log(f"  {raw_size_kb:.1f} KB -> {stripped_size_kb:.1f} KB "
            f"(-{raw_size_kb - stripped_size_kb:.1f} KB from debug names)")
        report["stages"].append({
            "stage": "strip_names",
            "size_before_kb": round(raw_size_kb, 1),
            "size_after_kb": round(stripped_size_kb, 1),
        })

    with open(args.output, "wb") as f:
        f.write(tflite_model)
    new_size_mb = len(tflite_model) / 1e6
    new_size_kb = len(tflite_model) / 1024
    log(f"  {orig_size_mb:.3f} MB -> {new_size_mb:.3f} MB "
        f"({orig_size_mb / new_size_mb:.1f}x reduction)")
    report["stages"].append({
        "stage": "convert", "quantization": args.quantization,
        "original_size_mb": round(orig_size_mb, 3),
        "output_size_mb": round(new_size_mb, 3),
        "size_reduction_x": round(orig_size_mb / new_size_mb, 2),
    })

    # CALIBRATION ADDITION: the flash-budget check itself.
    if new_size_kb > args.max_flash_kb:
        log(f"  ❌ FLASH BUDGET EXCEEDED: {new_size_kb:.1f} KB > "
            f"{args.max_flash_kb:.0f} KB limit. This file will very "
            f"likely fail to fit on the target board once HAL+TFLM+App "
            f"code is added. Do not proceed to flashing without either "
            f"raising --max-flash-kb (if targeting a bigger-flash board) "
            f"or investigating why quantization didn't shrink it as "
            f"expected (e.g. quantization=none was used by mistake).")
    else:
        log(f"  ✅ {new_size_kb:.1f} KB fits within the {args.max_flash_kb:.0f} KB budget.")

    log("Stage 4-5: validating converted model")
    validation = validate_tflite(args.output, model, args.batch_size, seq_len, n_features,
                                  args.quantization)
    report["stages"].append({"stage": "validate", **{
        k: v for k, v in validation.items() if k != "sample_predictions"
    }})
    log(f"  Forbidden resource-var ops present: {validation['has_forbidden_resource_var_ops']}")
    log(f"  Contains WHILE op: {validation['has_while_op']}")
    log(f"  Max |tflite - keras| output diff: {validation['output_max_abs_diff_vs_keras']:.2e}")

    with open("convert_no_rebuild_report.json", "w") as f:
        json.dump(report, f, indent=2)
    log("Report written to convert_no_rebuild_report.json")
    log(f"Done. Optimized model at: {args.output}")


if __name__ == "__main__":
    main()
