import tensorflow as tf

MODEL_PATH = "cubesat_soh_lstm.keras"
OUTPUT_PATH = "soh_model_SIMPLE.tflite"

# ---- Charge le modèle tel quel, SANS le modifier ------------------------
model = tf.keras.models.load_model(MODEL_PATH)
model.summary()

# ---- Conversion DIRECTE depuis le modèle Keras --------------------------

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

tflite_model = converter.convert()

with open(OUTPUT_PATH, "wb") as f:
    f.write(tflite_model)
print(f"Saved {OUTPUT_PATH} ({len(tflite_model)} bytes)")

# ---- Vérification automatique -------------------------------------------
interpreter = tf.lite.Interpreter(model_path=OUTPUT_PATH)
interpreter.allocate_tensors()

op_names = sorted(set(op["op_name"] for op in interpreter._get_ops_details()))
print("\nOperators present in the converted model:")
for name in op_names:
    print(" -", name)

FORBIDDEN_OPS = {"WHILE", "READ_VARIABLE", "ASSIGN_VARIABLE", "VAR_HANDLE"}
found = FORBIDDEN_OPS & set(op_names)
if found:
    print(f"\n❌ Toujours présents: {found}")
    print("   → Essai suivant: désactiver AutoGraph explicitement (voir option B)")
else:
    print("\n✅ Aucun op de contrôle de flux dynamique — modèle sûr pour invoke().")