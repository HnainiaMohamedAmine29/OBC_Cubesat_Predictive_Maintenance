import tensorflow as tf
import numpy as np
import joblib
import pandas as pd

class CubeSatSOHPredictor:
    def __init__(self):
        print("🔄 Loading LSTM SOH Predictor...")
        self.model = tf.keras.models.load_model('cubesat_soh_lstm_model.h5', compile=False)
        self.scaler_X = joblib.load('scaler_X.pkl')
        self.scaler_y = joblib.load('scaler_y.pkl')
        print("✅ LSTM Predictor loaded successfully!")
        
    def predict_soh(self, feature_dict, soh_prev):
        """Predict next SOH from feature dict and previous SOH"""
        try:
            row_dict = dict(feature_dict)
            row_dict["SOH_lag_1"] = float(soh_prev)
            
            x_df = pd.DataFrame([row_dict])
            x_df = x_df.reindex(columns=self.scaler_X.feature_names_in_)
            x_df = x_df.astype(float).fillna(0)
            
            x_scaled = self.scaler_X.transform(x_df)
            x_scaled = x_scaled.reshape(1, 1, x_scaled.shape[1])
            
            delta_scaled = self.model.predict(x_scaled, verbose=0)[0][0]
            delta_soh = self.scaler_y.inverse_transform([[delta_scaled]])[0][0]
            
            soh_new = soh_prev + delta_soh
            return max(0.0, min(1.05, soh_new))
        except Exception as e:
            print(f"Prediction error: {e}")
            return float(soh_prev)

# For testing
if __name__ == "__main__":
    pred = CubeSatSOHPredictor()
    print("✅ Predictor ready!")