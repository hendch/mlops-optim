"""Standalone script to run only the data preparation step."""

from src.model_pipeline import prepare_data


def main() -> None:
    """Execute data preparation and print basic info."""
    data_path = "data/raw/data.csv"
    x_train, x_test, y_train, y_test = prepare_data(data_path)

    print("Data prepared.")
    print("x_train shape:", x_train.shape)
    print("x_test shape:", x_test.shape)


if __name__ == "__main__":
    main()
