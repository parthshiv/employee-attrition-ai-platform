import pandas as pd
from features.feature_engineering import engineer_features
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

CSV_PATH = BASE_DIR / "data" / "raw" /"employees.csv"

df = pd.read_csv(CSV_PATH)

# print(df.head())
# print(df.columns)

df_features = engineer_features(df)

print(df_features)