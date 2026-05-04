"""
Script simple de data cleaning & preprocessing
Dataset batteries lithium-ion
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

# =========================
#  CONFIGURATION
# =========================
INPUT_PATH = "battery_dataset_normal.csv"
OUTPUT_PATH = "data_cleaned/cleaned_dataset.csv"

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(INPUT_PATH)
print("Dataset loaded:", df.shape)

# =========================
# DROP DUPLICATES
# =========================
df = df.drop_duplicates()
print("After removing duplicates:", df.shape)

# =========================
#  HANDLE MISSING VALUES
# =========================
for col in df.columns:
    if df[col].dtype.kind in 'biufc':
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

print("Missing values handled")

# =========================
# REMOVE OUTLIERS (IQR)
# =========================
outlier_cols = ['IR_ohm', 'DoD', 'charge_time_min', 'discharge_time_min']

for col in outlier_cols:
    if col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        before = df.shape[0]
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
        after = df.shape[0]
        
        print(f"{col}: removed {before - after} outliers")

# ========================
# LOG TRANSFORMATION
# =========================
log_cols = ['IR_ohm', 'DoD', 'charge_time_min', 'discharge_time_min']

for col in log_cols:
    if col in df.columns:
        df[col] = np.log1p(df[col])  # safer than log

print("Log transformation applied")

# =========================
#  SCALING
# =========================
numeric_cols = df.select_dtypes(include=[np.number]).columns

scaler = RobustScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("Scaling done")

# =========================
#  SAVE
# =========================
df.to_csv(OUTPUT_PATH, index=False)
print("Saved to:", OUTPUT_PATH)