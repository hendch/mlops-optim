# src/app.py
import os

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = "models/gradient_boost_model.joblib"


class PredictionRequest(BaseModel):
    """Input schema for prediction."""

    features: list[float]


class PredictionResponse(BaseModel):
    """Output schema for prediction."""

    prediction: float


app = FastAPI(
    title="ML Model API",
    version="0.1.0",
    description="Simple API to serve the insurance regression model.",
)


# --------- Model loading ---------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)


# --------- Endpoints ---------
@app.get("/health")
def health():
    """Health check endpoint used to verify that the API is running."""
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    """Basic information about the trained model."""
    return {
        "model_type": type(model).__name__,
        "model_path": MODEL_PATH,
        "features_order": [
            "age",
            "sex (encoded)",
            "bmi",
            "children",
            "smoker (encoded)",
            "region (encoded)",
        ],
        "target": "charges",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    """
    Predict using the trained model.

    The `features` list must follow the same order used during training:
    [age, sex_encoded, bmi, children, smoker_encoded, region_encoded]
    """
    try:
        X = np.array(payload.features).reshape(1, -1)
        y_pred = model.predict(X)
        return PredictionResponse(prediction=float(y_pred[0]))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
