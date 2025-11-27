"""Script to load a trained model and evaluate it on the test set."""

from src.model_pipeline import prepare_data, load_model, evaluate_model


def main() -> None:
    """Load the model, prepare data and evaluate performance."""
    data_path = "data/raw/data.csv"
    x_train, x_test, y_train, y_test = prepare_data(data_path)  # noqa: F841
    model = load_model("models/gradient_boost_model.joblib")
    evaluate_model(model, x_test, y_test)


if __name__ == "__main__":
    main()
