import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from features.feature_engineering import engineer_features
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = BASE_DIR / "data" / "raw" /"employees.csv"

df = pd.read_csv(CSV_PATH)

df = engineer_features(df)

print("Target (left) distribution:")
print(df["left"].value_counts())
print(df["left"].value_counts(normalize=True))

sns.countplot(x="left", data=df)
plt.title("Employee Attrition Distribution")
plt.show()

sns.histplot(df["salary_ratio"], kde=True)
plt.title("Salary Ratio Distribution")
plt.show()

sns.histplot(df["exp_salary_ratio"], kde=True)
plt.title("Experience vs Salary Ratio Distribution")
plt.show()

sns.countplot(x="low_satisfaction", hue="left", data=df)
plt.title("Low Satisfaction vs Attrition")
plt.show()
