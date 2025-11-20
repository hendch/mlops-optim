import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import json

logging.basicConfig(level=logging.INFO)
def validate_data(df) :
    #check required columns exist
    #check target column has no missing values
    required_columns = ["charges"]
    
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing from the dataset")

    if df.isnull().any().any():
        total_missing = df.isnull().sum().sum()
        logging.warning(f"Dataset contains {total_missing} missing values (but none will be removed).")
    else:
        logging.info("No missing values detected.")

    logging.info("Data validation completed (non-intrusive).")

    return df

def prepare_data(filename):
    # Load data
    df = pd.read_csv(filename)
    df = validate_data(df)

    # Encode categorical columns
    for col in ["sex", "smoker", "region"]:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Scale numeric columns
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    # Apply KNN imputation to the scaled DF
    imputer = KNNImputer(n_neighbors=5)
    df_imputed = pd.DataFrame(imputer.fit_transform(df_scaled), columns=df.columns)

    # Split features/target
    X = df_imputed.drop(columns=["charges"])
    y = df_imputed["charges"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    gbr = GradientBoostingRegressor()
    gbr.fit(X_train, y_train)
    return gbr

def save_metrics_to_json(mae, mse, r2, filename="results/metrics.json"):
    # Save evaluation metrics to a JSON file.

    # Make sure the directory exists (results/)
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    metrics = {
        "mae": float(mae),
        "mse": float(mse),
        "r2": float(r2),
    }

    with open(filename, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Metrics saved to {filename}")

def evaluate_model(model, X_test, y_test, save_to_json=True, json_path="results/metrics.json"):
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("Model Evaluation:")
    print("  MAE:", mae)
    print("  MSE:", mse)
    print("  R² :", r2)
    if save_to_json:
        save_metrics_to_json(mae, mse, r2, filename=json_path)

    return mae, mse, r2

def save_model(gbr, filename="model.joblib"):
    joblib.dump(gbr, filename)
    print(f"Model saved to {filename}")


def load_model(filename="model.joblib"):
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model
