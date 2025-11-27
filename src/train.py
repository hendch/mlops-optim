"""Training entrypoint for the insurance charges model."""

from src.model_pipeline import prepare_data, train_model, save_model


def main() -> None:
    """Prepare data, train the model and save it to disk."""
    data_path = "data/raw/data.csv"
    x_train, x_test, y_train, y_test = prepare_data(data_path)  # noqa: F841  (if x_test,y_test unused)

    model = train_model(x_train, y_train)
    save_model(model, "models/gradient_boost_model.joblib")


if __name__ == "__main__":
    main()
