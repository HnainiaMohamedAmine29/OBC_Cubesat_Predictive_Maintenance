import numpy as np
import tensorflow as tf
import keras
from keras import layers
import joblib
import pandas as pd


# ---------------------------------------------------------------------------
# Custom layer used inside the saved .keras model's attention-pooling block.
# Must be defined + registered (matching the training notebook exactly)
# BEFORE tf.keras.models.load_model() is called, or Keras can't reconstruct
# the model graph from the saved config.
# ---------------------------------------------------------------------------
@keras.saving.register_keras_serializable(package="cubesat_soh")
class AttentionContext(layers.Layer):
    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])


class CubeSatSOHPredict:
    def __init__(self,
                 model_path='cubesat_soh_lstm_v3_unroll_True.keras',
                 scaler_X_path='scaler_X.pkl',
                 features_path='features_used.pkl'):

        self.model = tf.keras.models.load_model(
            model_path,
            compile=False,
            # Belt-and-suspenders: some host environments (e.g. MATLAB's
            # embedded Python) don't reliably run the @register_keras_serializable
            # decorator before load_model() executes, so pass it explicitly too.
            custom_objects={"AttentionContext": AttentionContext},
        )
        self.scaler_X = joblib.load(scaler_X_path)
        self.features = joblib.load(features_path)

        self.seq_len = 30
        self.buffer = []

        print(f"✅ LSTM loaded | Window = {self.seq_len}")
        print(f"✅ Features count: {len(self.features)}")

        

        print(type(self.scaler_X))
        print(self.scaler_X.mean_[:5])
        print(self.scaler_X.scale_[:5])


    def predict_soh(self, feature_dict):

        # 1. Check features
        missing = [f for f in self.features if f not in feature_dict]
        if missing:
            raise ValueError(f"❌ Missing features: {missing}")

        # 2. Build dataframe in correct order
        df = pd.DataFrame([[feature_dict[f] for f in self.features]],
                          columns=self.features).astype(np.float32)

        # 🚨 DEBUG
        #print("RAW:", df.iloc[0].to_dict())

        # 3. Scale
        feat_scaled = self.scaler_X.transform(df)[0]

        #print("SCALED (first 5):", feat_scaled[:5])

        # 4. Buffer
        self.buffer.append(feat_scaled)

        if len(self.buffer) > self.seq_len:
            self.buffer.pop(0)

        if len(self.buffer) < self.seq_len:
            return None

        # 5. Sequence
        X_seq = np.array(self.buffer, dtype=np.float32).reshape(1, self.seq_len, -1)

        # 🚨 DEBUG
        #print("SEQ VAR:", np.var(X_seq))

        # 6. Predict
        pred = float(self.model.predict(X_seq, verbose=0)[0][0])

        return np.clip(pred, 0.0, 1.05)


if __name__ == "__main__":
    # ---- Config: point these at your .keras model + scaler/feature files ----
    predictor = CubeSatSOHPredict(
        model_path='cubesat_soh_lstm_v3_unroll_True.keras',  # .keras model, not .tflite
        scaler_X_path='scaler_X.pkl',
        features_path='features_used.pkl'
    )
    print("✅ Predictor READY (testing .keras model)")

    # ---- Test with synthetic feature rows ----
    # Replace this with real telemetry rows if you have them (e.g. from a CSV).
    n_test_steps = predictor.seq_len + 5  # feed a few extra steps past the window
    rng = np.random.default_rng(42)

    print(f"\nRunning {n_test_steps} synthetic prediction steps...")
    for step in range(n_test_steps):
        # Generate a random but plausible value for every required feature.
        fake_row = {feat: float(rng.uniform(0.0, 1.0)) for feat in predictor.features}

        soh = predictor.predict_soh(fake_row)

        if soh is None:
            print(f"Step {step + 1:02d}/{n_test_steps}: buffering "
                  f"({len(predictor.buffer)}/{predictor.seq_len})")
        else:
            print(f"Step {step + 1:02d}/{n_test_steps}: predicted SOH = {soh:.4f}")

    print("\n✅ Test run complete — .keras model produced predictions above.")