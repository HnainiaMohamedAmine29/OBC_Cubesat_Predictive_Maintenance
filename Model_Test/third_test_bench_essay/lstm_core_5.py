import numpy as np
import tensorflow as tf
import joblib
import pandas as pd


class CubeSatSOHPredict:
    def __init__(self,
                 model_path='cubesat_soh_lstm.keras',
                 scaler_X_path='scaler_X.pkl',
                 features_path='features_used.pkl'):

        self.model = tf.keras.models.load_model(model_path, compile=False)
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
    pred = CubeSatSOHPredict()
    print("✅ Predictor READY")