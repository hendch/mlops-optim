import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def prepare_data(filename):
    # Load data
    df = pd.read_csv(filename)

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
def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("Model Evaluation:")
    print("  MAE:", mae)
    print("  MSE:", mse)
    print("  R² :", r2)

    return mae, mse, r2

def save_model(gbr, filename="model.pkl"):
    joblib.dump(gbr, filename)
    print(f"Model saved to {filename}")


def load_model(filename="model.pkl"):
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model
