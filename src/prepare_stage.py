"""DVC stage script to prepare data and save it."""

from pathlib import Path
import pandas as pd
from src.model_pipeline import prepare_data


def main() -> None:
    """Prepare data and save processed features + target to disk."""
    data_path = "data/raw/data.csv"
    x_train, x_test, y_train, y_test = prepare_data(data_path)

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save train and test sets as CSV (or parquet if you prefer)
    pd.concat([x_train, y_train], axis=1).to_csv(out_dir / "train.csv", index=False)
    pd.concat([x_test, y_test], axis=1).to_csv(out_dir / "test.csv", index=False)

    print("✅ Data prepared and saved to data/processed/")


if __name__ == "__main__":
    main()
