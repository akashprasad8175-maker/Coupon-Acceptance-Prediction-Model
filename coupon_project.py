"""
Coupon Acceptance Prediction - full pipeline in one file.

Run from the repo folder:
    python coupon_project.py

What it does (in order):
    1. Loads and cleans the data
    2. Saves EDA charts to images/
    3. Builds features and splits train/test
    4. Trains Logistic Regression, Random Forest and XGBoost
    5. Compares them (accuracy, precision, recall, F1, ROC-AUC, 5-fold CV)
    6. Saves confusion matrix, ROC curve, feature importance and SHAP plots
    7. Saves the best model to models/ and the results table to outputs/
"""

import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # draw charts to files, no pop-up windows
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Paths and settings
# ----------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
DATA_FILE = BASE / "data" / "in-vehicle-coupon-recommendation.csv"
IMG_DIR = BASE / "images"
OUT_DIR = BASE / "outputs"
MODEL_DIR = BASE / "models"
for folder in (IMG_DIR, OUT_DIR, MODEL_DIR):
    folder.mkdir(exist_ok=True)

TARGET = "Y"
RANDOM_STATE = 42
FREQ_COLS = ["Bar", "CoffeeHouse", "CarryAway", "RestaurantLessThan20", "Restaurant20To50"]
EDA_COLS = ["coupon", "passanger", "time", "weather", "expiration", "age", "destination"]


# ----------------------------------------------------------------------------
# 1. Load and clean
# ----------------------------------------------------------------------------
def load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Acceptance rate (Y=1): {df[TARGET].mean():.1%}")

    # 'car' is almost entirely empty
    df = df.drop(columns=["car"], errors="ignore")

    before = len(df)
    df = df.drop_duplicates()
    print(f"Removed {before - len(df)} duplicate rows")

    # Fill the few missing values with the most common value
    for col in FREQ_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Drop columns that have the same value in every row (no information)
    constant_cols = [c for c in df.columns if df[c].nunique() == 1]
    if constant_cols:
        print(f"Dropping constant columns: {constant_cols}")
        df = df.drop(columns=constant_cols)

    print(f"Missing values left: {int(df.isnull().sum().sum())}")
    print(f"Clean data: {df.shape[0]} rows, {df.shape[1]} columns\n")
    return df


# ----------------------------------------------------------------------------
# 2. EDA charts
# ----------------------------------------------------------------------------
def run_eda(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    # Class balance
    plt.figure(figsize=(5, 4))
    df[TARGET].value_counts().sort_index().plot(kind="bar", color=["#d9534f", "#5cb85c"])
    plt.xticks([0, 1], ["Rejected (0)", "Accepted (1)"], rotation=0)
    plt.title("Class balance")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "eda_class_balance.png", dpi=150)
    plt.close()

    # Acceptance rate by category
    for col in EDA_COLS:
        if col not in df.columns:
            continue
        rates = df.groupby(col)[TARGET].mean().sort_values(ascending=False)
        plt.figure(figsize=(8, 4.5))
        sns.barplot(x=rates.index, y=rates.values, color="#4c72b0")
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("Acceptance rate")
        plt.ylim(0, 1)
        plt.title(f"Coupon acceptance rate by {col}")
        plt.tight_layout()
        plt.savefig(IMG_DIR / f"eda_{col}.png", dpi=150)
        plt.close()

    print(f"EDA charts saved in {IMG_DIR.name}/\n")


# ----------------------------------------------------------------------------
# 3. Features and split
# ----------------------------------------------------------------------------
def build_features(df: pd.DataFrame):
    X = pd.get_dummies(df.drop(columns=[TARGET]), drop_first=True).astype(int)
    # XGBoost does not allow these characters in column names
    X.columns = X.columns.str.replace(r"[\[\]<>]", "", regex=True)
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )


# ----------------------------------------------------------------------------
# 4-5. Train and evaluate
# ----------------------------------------------------------------------------
def train_and_evaluate(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=14,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=5,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        ),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rows, fitted = [], {}
    for name, model in models.items():
        print(f"Training {name} ...")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        cv_acc = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred),
                "Recall": recall_score(y_test, pred),
                "F1": f1_score(y_test, pred),
                "ROC-AUC": roc_auc_score(y_test, proba),
                "CV Accuracy (mean)": cv_acc.mean(),
                "CV Accuracy (std)": cv_acc.std(),
            }
        )
        fitted[name] = model

    results = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    results.round(4).to_csv(OUT_DIR / "model_comparison.csv", index=False)
    print("\nModel comparison:")
    print(results.round(4).to_string(index=False))
    print()
    return results, fitted


# ----------------------------------------------------------------------------
# 6. Plots for the best model
# ----------------------------------------------------------------------------
def plot_best_model(name, model, X_train, X_test, y_test) -> None:
    # Confusion matrix
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=["Rejected", "Accepted"], cmap="Blues"
    )
    plt.title(f"Confusion matrix - {name}")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "best_confusion_matrix.png", dpi=150)
    plt.close()

    # ROC curve
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title(f"ROC curve - {name}")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "best_roc_curve.png", dpi=150)
    plt.close()

    # Feature importance (tree models and logistic regression)
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=X_train.columns)
    elif hasattr(model, "coef_"):
        importances = pd.Series(model.coef_[0], index=X_train.columns).abs()
    if importances is not None:
        top = importances.sort_values(ascending=False).head(15)[::-1]
        plt.figure(figsize=(8, 6))
        top.plot(kind="barh", color="#4c72b0")
        plt.title(f"Top 15 features - {name}")
        plt.tight_layout()
        plt.savefig(IMG_DIR / "best_feature_importance.png", dpi=150)
        plt.close()
        top.sort_values(ascending=False).to_csv(OUT_DIR / "top_features.csv", header=["importance"])

    # SHAP summary plot (only for tree models; skipped if shap is missing)
    if name in ("Random Forest", "XGBoost"):
        try:
            import shap

            sample = X_test.sample(min(1000, len(X_test)), random_state=RANDOM_STATE)
            values = shap.TreeExplainer(model).shap_values(sample)
            if isinstance(values, list):  # older shap + random forest
                values = values[1]
            elif getattr(values, "ndim", 2) == 3:  # newer shap + random forest
                values = values[:, :, 1]
            shap.summary_plot(values, sample, show=False, max_display=15)
            plt.tight_layout()
            plt.savefig(IMG_DIR / "best_shap_summary.png", dpi=150, bbox_inches="tight")
            plt.close()
        except Exception as exc:  # SHAP is a bonus, never stop the pipeline for it
            print(f"SHAP plot skipped: {exc}")

    print(f"Best-model charts saved in {IMG_DIR.name}/\n")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_FILE}. Put the CSV in the 'data' folder next to this script."
        )

    df = load_and_clean()
    run_eda(df)

    X_train, X_test, y_train, y_test = build_features(df)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}\n")

    results, fitted = train_and_evaluate(X_train, X_test, y_train, y_test)

    best_name = results.loc[0, "Model"]
    best_model = fitted[best_name]
    print(f"Best model by ROC-AUC: {best_name}\n")

    plot_best_model(best_name, best_model, X_train, X_test, y_test)

    joblib.dump(
        {"model": best_model, "columns": list(X_train.columns)},
        MODEL_DIR / "best_model.pkl",
        compress=3,
    )
    print(f"Saved model to {MODEL_DIR.name}/best_model.pkl")
    print(f"Saved results table to {OUT_DIR.name}/model_comparison.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()
