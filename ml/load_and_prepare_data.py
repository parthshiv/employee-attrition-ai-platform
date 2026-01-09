import pandas as pd
from features.feature_engineering import engineer_features
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = BASE_DIR / "data" / "raw" /"employees.csv"

df = pd.read_csv(CSV_PATH)

# Inspect raw data
print("Raw Data Preview:")
print(df.head())
print("\nColumns:")
print(df.columns)


# -----------------------------
# Apply feature engineering
# -----------------------------
df_features = engineer_features(df)
# print(df_features)
# -----------------------------
# Separate features and target
# -----------------------------
X = df_features.drop(columns=["left"])
y = df_features["left"]

# Inspect engineered data
print("\nEngineered Features Preview:")
print(X.head())

print("\nTarget distribution:")
print(y.value_counts())