import tensorflow as tf
import numpy as np
import joblib

# load ONCE
model = tf.keras.models.load_model("cubesat_soh_lstm_model.h5", compile=False)
scaler = joblib.load("scaler_X.pkl")


def predict(data):
    x = np.array([[data[f] for f in features]])
    x = scaler.transform(x)
    x = x.reshape(1, 1, x.shape[1])
    return float(model.predict(x, verbose=0)[0][0])