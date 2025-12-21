"""Model pipeline: data preparation, training, evaluation and metrics saving."""
from __future__ import annotations
import json
import logging
import os
from datetime import datetime
from typing import Tuple


import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import requests
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from datetime import datetime, timezone
import uuid
import requests

logging.basicConfig(level=logging.INFO)
EXPERIMENT_NAME = "insurance_charges_gb"


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run non-intrusive validation on the raw dataframe (log-only, no modification)."""
    required_columns = ["charges"]

    for col in required_columns:
        if col not in df.columns:
            logging.warning("Required column '%s' is missing from the dataset.", col)

    total_missing = df.isnull().sum().sum()
    if total_missing > 0:
        logging.warning("Dataset contains %s missing values (none will be removed).", total_missing)
    else:
        logging.info("No missing values detected in the dataset.")

    logging.info("Data validation completed (non-intrusive).")
    return df


def prepare_data(
    filename: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load the CSV, preprocess features and return train/test splits."""
    df = pd.read_csv(filename)
    df = validate_data(df)

    # Encode categorical columns
    for col in ["sex", "smoker", "region"]:
        if col in df.columns:
            label_encoder = LabelEncoder()
            df[col] = label_encoder.fit_transform(df[col].astype(str))

    # Scale numeric columns
    scaler = StandardScaler()
    numeric_columns = df.select_dtypes(include="number").columns
    scaled = scaler.fit_transform(df[numeric_columns])
    df[numeric_columns] = scaled

    # Apply KNN imputation
    imputer = KNNImputer(n_neighbors=5)
    imputed_array = imputer.fit_transform(df[numeric_columns])
    df[numeric_columns] = imputed_array

    # Split features / target
    features = df.drop(columns=["charges"])
    target = df["charges"]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )

    return x_train, x_test, y_train, y_test


def train_model(x_train: pd.DataFrame, y_train: pd.Series) -> GradientBoostingRegressor:
    """Train a Gradient Boosting Regressor on the training data."""
    model = GradientBoostingRegressor()
    model.fit(x_train, y_train)
    return model


def save_metrics_to_json(mae: float, mse: float, r2: float, filename="results/metrics.json"):
    """Save evaluation metrics to a JSON file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    metrics = {"mae": float(mae), "mse": float(mse), "r2": float(r2)}

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)


def send_metrics_to_elasticsearch(metrics: dict) -> None:
    index_name = "mlflow-metrics"

    # POST to /<index>/_doc creates a new document with a generated _id
    # (If you prefer controlling the ID, use PUT with a UUID doc_id.)
    url = f"http://localhost:9200/{index_name}/_doc"

    payload = {
        # Use UTC with timezone for Kibana time filtering
        "@timestamp": datetime.now(timezone.utc).isoformat(),
        # Extra unique field (handy for debugging)
        "event_id": str(uuid.uuid4()),
        "model_name": "gradient_boost",
        "mae": metrics.get("mae"),
        "mse": metrics.get("mse"),
        "r2": metrics.get("r2"),
    }

    try:
        resp = requests.post(url, json=payload, timeout=5)
        resp.raise_for_status()
    except requests.RequestException as exc:
        # Non-blocking: monitoring must not break the pipeline
        print(f"⚠ Elasticsearch logging failed: {exc}")

def evaluate_model(
    model: GradientBoostingRegressor,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    save_to_json: bool = True,
    json_path: str = "results/metrics.json",
):
    """Evaluate the model on a test set and optionally save metrics."""
    predictions = model.predict(x_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    logging.info("Model evaluation - MAE: %s, MSE: %s, R2: %s", mae, mse, r2)

    if save_to_json:
        save_metrics_to_json(mae, mse, r2, filename=json_path)
    metrics = {
        "mae": mae,
        "mse": mse,
        "r2": r2,
    }
    send_metrics_to_elasticsearch(metrics)

    return mae, mse, r2


def save_model(
    model: GradientBoostingRegressor,
    filename: str = "models/gradient_boost_model.joblib",
) -> None:
    """Persist the trained model to disk using joblib."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    joblib.dump(model, filename)
    logging.info("Model saved to %s", filename)


def load_model(
    filename: str = "models/gradient_boost_model.joblib",
) -> GradientBoostingRegressor:
    """Load a trained model from disk."""
    loaded_model = joblib.load(filename)
    logging.info("Model loaded from %s", filename)
    return loaded_model


def train_with_mlflow(
    data_path: str = "data/raw/data.csv",
    model_path: str = "models/gradient_boost_model.joblib",
    metrics_path: str = "results/metrics.json",
) -> None:
    # Configure experiment (once)
    mlflow.set_experiment(EXPERIMENT_NAME)
    # Optional: if you want a SQLite backend instead of ./mlruns:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    # Prepare data outside the run (or inside; both are fine)
    x_train, x_test, y_train, y_test = prepare_data(data_path)

    # Wrap model training + evaluation in a single MLflow run
    with mlflow.start_run(run_name="gradient_boost_default"):
        # Train
        model = train_model(x_train, y_train)

        # Evaluate (and still save metrics.json for your CI / workshop)
        mae, mse, r2 = evaluate_model(
            model,
            x_test,
            y_test,
            save_to_json=True,
            json_path=metrics_path,
        )

        # Log hyperparameters (from scikit-learn model)
        mlflow.log_params(model.get_params())

        # Log metrics
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)

        # Log model artifact
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Also save the model locally as before (for FastAPI, tests, etc.)
        save_model(model, filename=model_path)
