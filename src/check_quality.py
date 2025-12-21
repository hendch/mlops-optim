"""Fail if model quality (R2) is below a given threshold."""

import sys
from pathlib import Path

import pandas as pd

from src.model_pipeline import evaluate_model, load_model


def main() -> None:
    """Load processed test data, model, and enforce a minimum R2."""
    test_path = Path("data/processed/test.csv")
    if not test_path.exists():
        print("❌ data/processed/test.csv not found. Run DVC 'prepare' stage first.")
        sys.exit(1)

    df_test = pd.read_csv(test_path)
    x_test = df_test.drop(columns=["charges"])
    y_test = df_test["charges"]

    model = load_model("models/gradient_boost_model.joblib")

    _, _, r2 = evaluate_model(model, x_test, y_test, save_to_json=True)

    threshold = 0.7
    if r2 < threshold:
        print(f"❌ R2={r2:.3f} is below threshold {threshold}")
        sys.exit(1)

    print(f"✅ R2={r2:.3f} passes threshold {threshold}")


if __name__ == "__main__":
    main()
