
import joblib
import pandas as pd

# Load the trained model
artifact = joblib.load("models/best_model.pkl")

print("\n========== MODEL INFORMATION ==========")
print("Artifact keys:", artifact.keys())
print("Model type:", type(artifact["model"]).__name__)

columns = artifact["columns"]

print("Total model features:", len(columns))
print("\nModel feature names:")
for col in columns:
    print(col)

# Load a small sample of the original dataset
df = pd.read_csv(
    "data/in-vehicle-coupon-recommendation.csv",
    nrows=100
)

print("\n========== DATASET INFORMATION ==========")
print("Dataset columns:")

for col in df.columns:
    values = df[col].dropna().unique().tolist()
    print(f"\n{col}")
    print("Possible values:", values[:15])
