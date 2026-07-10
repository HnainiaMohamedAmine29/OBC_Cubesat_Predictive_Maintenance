import numpy as np
import tensorflow as tf
import joblib
import pandas as pd


class CubeSatSOHPredict:
    def __init__(self,
                 model_path='soh_model_SIMPLE.tflite',
                 scaler_X_path='scaler_X.pkl',
                 features_path='features_used.pkl'):

        # --- Load TFLite model via the Interpreter (NOT tf.keras.models.load_model,
        #     which only understands SavedModel/H5, not the .tflite FlatBuffer) ---
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.interpreter.reset_all_variables() 

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # This model's real signature is (1, 30, 20) -> (1, 1)
        in_shape = self.input_details[0]['shape']
        self.seq_len = int(in_shape[1])
        self.n_features = int(in_shape[2])

        self.scaler_X = joblib.load(scaler_X_path)
        self.features = joblib.load(features_path)

        if len(self.features) != self.n_features:
            raise ValueError(
                f"❌ features_used.pkl has {len(self.features)} features but "
                f"model expects {self.n_features}."
            )

        self.buffer = []

        print(f"✅ TFLite LSTM loaded | Window = {self.seq_len} | Features = {self.n_features}")
        print(f"   Input:  {self.input_details[0]['shape']} ({self.input_details[0]['dtype']})")
        print(f"   Output: {self.output_details[0]['shape']} ({self.output_details[0]['dtype']})")

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
        X_seq = np.array(self.buffer, dtype=np.float32).reshape(1, self.seq_len, self.n_features)

        # 🚨 DEBUG
        #print("SEQ VAR:", np.var(X_seq))

        # 6. Predict via TFLite Interpreter
        self.interpreter.set_tensor(self.input_details[0]['index'], X_seq)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_details[0]['index'])
        pred = float(out[0][0])

        return np.clip(pred, 0.0, 1.05)


if __name__ == "__main__":
    pred = CubeSatSOHPredict()
    print("✅ Predictor READY")