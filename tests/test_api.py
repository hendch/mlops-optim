# tests/test_api.py
from fastapi.testclient import TestClient

from src.app import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["model_type"]
    assert body["model_path"]
    assert "features_order" in body


def test_predict_valid_payload():
    # Example vector: [age, sex_encoded, bmi, children, smoker_encoded, region_encoded]
    payload = {"features": [19, 0, 27.9, 0, 1, 3]}

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    body = response.json()
    assert "prediction" in body
    assert isinstance(body["prediction"], (int, float))
