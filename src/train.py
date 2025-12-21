"""Training entrypoint for the insurance charges model (with MLflow)."""

from src.model_pipeline import train_with_mlflow


def main() -> None:
    """Prepare data, train the model, evaluate it, and log everything with MLflow."""
    train_with_mlflow(
        data_path="data/raw/data.csv",
        model_path="models/gradient_boost_model.joblib",
        metrics_path="results/metrics.json",
    )


if __name__ == "__main__":
    main()
