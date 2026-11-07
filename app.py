
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request


# --------------------------------------------------
# 1. APPLICATION SETTINGS
# --------------------------------------------------

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "Best_Model.pkl"
DATA_PATH = BASE_DIR / "data" / "in-vehicle-coupon-recommendation.csv"

FREQ_COLS = [
    "Bar",
    "CoffeeHouse",
    "CarryAway",
    "RestaurantLessThan20",
    "Restaurant20To50"
]


# --------------------------------------------------
# 2. LOAD THE TRAINED MODEL
# --------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
model_columns = artifact["columns"]

print("Trained model loaded successfully!")


# --------------------------------------------------
# 3. LOAD AND CLEAN THE DATASET
# --------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

# Remove the car column, as done during training
if "car" in df.columns:
    df = df.drop(columns=["car"])

# Remove duplicate records
df = df.drop_duplicates()

# Fill missing frequency values using their dataset modes
for col in FREQ_COLS:
    if col in df.columns:
        mode_value = df[col].mode(dropna=True)

        if not mode_value.empty:
            df[col] = df[col].fillna(mode_value.iloc[0])

# Remove constant columns
constant_columns = [
    col for col in df.columns
    if df[col].nunique(dropna=False) <= 1
]

df = df.drop(columns=constant_columns)

# Keep the input columns, excluding the target
input_columns = [
    col for col in df.columns if col != "Y"
]

# Identify numeric columns for correct input conversion
numeric_columns = [
    col for col in input_columns
    if pd.api.types.is_numeric_dtype(df[col])
]


# --------------------------------------------------
# 4. PREPARE WEBSITE FORM FIELDS
# --------------------------------------------------

def make_label(column):
    labels = {
        "passanger": "Passenger",
        "has_children": "Has Children",
        "toCoupon_GEQ15min": "Coupon Distance: At Least 15 Minutes",
        "toCoupon_GEQ25min": "Coupon Distance: At Least 25 Minutes",
        "direction_same": "Same Direction",
        "direction_opp": "Opposite Direction",
        "temperature": "Temperature"
    }

    if column in labels:
        return labels[column]

    return column.replace("_", " ").title()


fields = []

for col in input_columns:

    options = df[col].dropna().unique().tolist()

    # Sort options for easier selection
    options = sorted(options, key=lambda x: str(x))

    fields.append({
        "name": col,
        "label": make_label(col),
        "options": [str(value) for value in options]
    })


# --------------------------------------------------
# 5. PREPARE PREDICTION INPUT
# --------------------------------------------------

def _clean(text):
    # XGBoost removes these characters from column names during training
    return str(text).replace("<", "").replace(">", "").replace("[", "").replace("]", "").strip()


def prepare_input(form):
    row = {}
    form_keys = sorted(form.keys(), key=len, reverse=True)

    for col in model_columns:
        value = 0

        if col in form:
            # Numeric column, e.g. temperature, has_children, direction_same
            try:
                value = float(form[col])
            except ValueError:
                value = 0
        else:
            # One-hot column, e.g. coupon_Carry out & Take away
            for key in form_keys:
                if col.startswith(key + "_"):
                    category = col[len(key) + 1:]
                    if _clean(category) == _clean(form[key]):
                        value = 1
                    break

        row[col] = value

    return pd.DataFrame([row], columns=model_columns)

# --------------------------------------------------
# 6. WEBSITE HOME PAGE
# --------------------------------------------------

@app.route("/", methods=["GET"])
def home():

    return render_template(
        "index.html",
        fields=fields,
        prediction=None,
        probability=None,
        error=None
    )


# --------------------------------------------------
# 7. PREDICT COUPON ACCEPTANCE
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    print("FORM:", dict(request.form))

    try:
        features = prepare_input(request.form)
        print("SHAPE:", features.shape)
        print("NON-ZERO:", {c: features[c].iloc[0] for c in features.columns if features[c].iloc[0] != 0})
        print("COUPON COLUMNS IN MODEL:", [c for c in model_columns if "coupon" in c.lower()])

        probability = model.predict_proba(features)[0][1]
        prediction_value = model.predict(features)[0]

        if prediction_value == 1:
            prediction = "Customer is likely to ACCEPT the coupon."
        else:
            prediction = "Customer is likely to REJECT the coupon."

        probability_percentage = round(probability * 100, 2)

        return render_template(
            "index.html",
            fields=fields,
            prediction=prediction,
            probability=probability_percentage,
            error=None
        )

    except Exception as e:
        return render_template(
            "index.html",
            fields=fields,
            prediction=None,
            probability=None,
            error=f"Prediction error: {str(e)}"
        )

# --------------------------------------------------
# 8. RUN THE APPLICATION
# --------------------------------------------------

if __name__ == "__main__":
    print("\nStarting Coupon Acceptance Prediction Website...")
    app.run(host="0.0.0.0", port=5000, debug=False)
    
