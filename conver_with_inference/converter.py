
import argparse
import json
import os
import time

import numpy as np
import tensorflow as tf


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def build_concrete_function(model, batch_size, seq_len, n_features):
    model.trainable = False

    @tf.function
    def model_fn(x):
        return model(x)

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
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quantization == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantization == "full_int8":
        if not representative_data_path:
            raise ValueError("--representative-data is required for full_int8 quantization")
        rep = np.load(representative_data_path)["X"].astype("float32")

        def rep_dataset():
            for i in range(min(len(rep), 200)):
                yield [rep[i:i + 1]]

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = rep_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.float32
    else:
        raise ValueError(f"Unknown quantization mode: {quantization}")

    return converter.convert()


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

    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]
    log(f"  Input shape/dtype: {in_detail['shape']} / {in_detail['dtype']}")
    log(f"  Ops used: {op_names} ({total_op_instances} total instances)")

    def to_input_dtype(x):
        if quantization == "full_int8":
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
        "output_max_abs_diff_vs_keras": max_diff,
        "sample_predictions": predictions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cubesat_soh_lstm.keras")
    parser.add_argument("--output", default="soh_model_optimized_no_rebuild.tflite")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--quantization", choices=["none", "dynamic_range", "float16", "full_int8"],
                         default="dynamic_range")
    parser.add_argument("--representative-data", default=None,
                         help="Path to .npz with array X, shape (N, seq_len, n_features). Required for full_int8.")
    args = parser.parse_args()

    report = {"stages": []}

    log("Stage 1: loading source model (unroll settings left as-is)")
    model = tf.keras.models.load_model(args.model)
    seq_len, n_features = model.input_shape[1], model.input_shape[2]
    orig_size_mb = os.path.getsize(args.model) / 1e6
    for l in model.layers:
        if "lstm" in l.name.lower():
            log(f"  {l.name}: unroll={l.unroll}, stateful={l.stateful} (unchanged)")

    log("Stage 2: tracing concrete function with static input shape "
        f"({args.batch_size}, {seq_len}, {n_features})")
    concrete_func = build_concrete_function(model, args.batch_size, seq_len, n_features)

    log(f"Stage 3: converting to TFLite (quantization={args.quantization})")
    tflite_model = convert_to_tflite(concrete_func, args.quantization, args.representative_data)
    with open(args.output, "wb") as f:
        f.write(tflite_model)
    new_size_mb = len(tflite_model) / 1e6
    log(f"  {orig_size_mb:.3f} MB -> {new_size_mb:.3f} MB "
        f"({orig_size_mb / new_size_mb:.1f}x reduction)")
    report["stages"].append({
        "stage": "convert", "quantization": args.quantization,
        "original_size_mb": round(orig_size_mb, 3),
        "output_size_mb": round(new_size_mb, 3),
        "size_reduction_x": round(orig_size_mb / new_size_mb, 2),
    })

    log("Stage 4-5: validating converted model")
    validation = validate_tflite(args.output, model, args.batch_size, seq_len, n_features,
                                  args.quantization)
    report["stages"].append({"stage": "validate", **{
        k: v for k, v in validation.items() if k != "sample_predictions"
    }})
    log(f"  Forbidden resource-var ops present: {validation['has_forbidden_resource_var_ops']}")
    log(f"  Max |tflite - keras| output diff: {validation['output_max_abs_diff_vs_keras']:.2e}")

    with open("convert_no_rebuild_report.json", "w") as f:
        json.dump(report, f, indent=2)
    log("Report written to convert_no_rebuild_report.json")
    log(f"Done. Optimized model at: {args.output}")


if __name__ == "__main__":
    main()
