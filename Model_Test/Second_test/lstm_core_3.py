import tensorflow as tf
import numpy as np
import joblib
import pandas as pd

FEATURES_PATH = 'features.pkl'

class CubeSatSOHPredict:
    def __init__(self):
        print("🔄 Loading LSTM SOH Predictor...")
        
        self.model = tf.keras.models.load_model('cubesat_soh_lstm_model.h5', compile=False)
        self.scaler_X = joblib.load('scaler_X.pkl')
        self.scaler_y = joblib.load('scaler_y.pkl')
        self.features = joblib.load(FEATURES_PATH)

        self.sequence_length = 20
        self.history_buffer = []

        print("✅ LSTM Predictor loaded successfully!")
        #print("TRAIN FEATURES:", list(self.scaler_X.feature_names_in_))

    def predict_soh(self, feature_dict, soh_prev):
        #print("🔥 NEW VERSION RUNNING 🔥")

        # 1. Copy input
        row = dict(feature_dict)

        # 2. Add lag feature
        row["SOH_lag_1"] = soh_prev

        # 3. Get expected order from scaler
        expected_cols = list(self.scaler_X.feature_names_in_)

        # 4. Build aligned row (STRICT ORDER)
        aligned_row = {}
        for col in expected_cols:
            aligned_row[col] = float(row.get(col, 0.0))

        # 5. Update buffer (ONLY ONCE)
        self.history_buffer.append(aligned_row)
        if len(self.history_buffer) > self.sequence_length:
            self.history_buffer.pop(0)

        # 6. Pad sequence if needed
        buffer = self.history_buffer.copy()
        while len(buffer) < self.sequence_length:
            buffer.insert(0, buffer[0])

        # 7. Create DataFrame with CORRECT ORDER
        df = pd.DataFrame(buffer, columns=expected_cols)

        # 🔍 DEBUG (IMPORTANT)
        #print("INPUT FEATURES:", row)
        #print("ORDER USED:", self.features)

        # 8. Scale (KEEP AS DATAFRAME)
        x_scaled = self.scaler_X.transform(df)

        # 9. Reshape for LSTM
        x_scaled = x_scaled.reshape(1, self.sequence_length, -1)
        
        # 10. Predict
        delta_scaled = self.model.predict(x_scaled, verbose=0)[0][0]
        delta_soh = self.scaler_y.inverse_transform([[delta_scaled]])[0][0]
        print("delta_soh:", delta_soh)
    
        # 12. Update SOH
        soh_new = soh_prev + delta_soh
        soh_new = max(0.0, min(1.0, soh_new))
        
        return soh_new


if __name__ == "__main__":
    pred = CubeSatSOHPredict()
    print("✅ Predictor ready!")