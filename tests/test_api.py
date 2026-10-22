from fastapi.testclient import TestClient
from app.main import app


class ConstantModel:
    def predict(self, features):
        return [16.25] * len(features)


def test_prediction_endpoint_uses_model_bundle():
    bundle = {"model": ConstantModel(), "model_name": "test"}

    with TestClient(app) as client:
        app.state.bundle = bundle
        response = client.post(
            "/predict",
            json={
                "order_date": "2025-03-11",
                "order_hour": 18,
                "city": "Delhi",
                "dark_store_id": "DS-DEL-01",
                "category": "Snacks",
                "quantity": 2,
                "unit_price": 40,
                "order_value": 80,
                "distance_km": 2.4,
                "payment_mode": "UPI",
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "predicted_delivery_mins": 16.25,
        "model": "test",
        "unit": "minutes",
    }


def test_home_page_serves_prediction_frontend():
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "prediction-form" in response.text
    assert 'fetch("/predict"' in response.text


def test_predict_requires_training_artifact():
    with TestClient(app) as client:
        app.state.bundle = None
        response = client.post(
            "/predict",
            json={"city": "Delhi", "category": "Snacks"},
        )

    assert response.status_code == 503


def test_prediction_rejects_invalid_input():
    with TestClient(app) as client:
        app.state.bundle = {"model": object(), "model_name": "test"}
        response = client.post(
            "/predict",
            json={"city": "Delhi", "category": "Snacks", "order_hour": 24},
        )

    assert response.status_code == 422
