"""Integration tests for the FastAPI prediction endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

VALID_CUSTOMER = {
    "customer_id": "TEST-001",
    "tenure_months": 24,
    "monthly_charges": 65.50,
    "total_charges": 1572.0,
    "num_products": 3,
    "has_internet_service": True,
    "has_phone_service": True,
    "contract_type": "month-to-month",
}


def test_health_endpoint(api_client: TestClient) -> None:
    response = api_client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_single_valid(api_client: TestClient) -> None:
    response = api_client.post("/api/v1/predict", json=VALID_CUSTOMER)
    assert response.status_code == 200
    body = response.json()
    assert "churn_probability" in body
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert isinstance(body["churn_predicted"], bool)
    assert body["customer_id"] == "TEST-001"


def test_predict_single_invalid_payload(api_client: TestClient) -> None:
    bad_payload = {**VALID_CUSTOMER, "tenure_months": -5}
    response = api_client.post("/api/v1/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_batch(api_client: TestClient) -> None:
    payload = {
        "customers": [VALID_CUSTOMER, {**VALID_CUSTOMER, "customer_id": "TEST-002"}]
    }
    response = api_client.post("/api/v1/predict/batch", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2


def test_predict_batch_empty_list(api_client: TestClient) -> None:
    response = api_client.post("/api/v1/predict/batch", json={"customers": []})
    assert response.status_code == 422


def test_model_validator_rejects_low_total(api_client: TestClient) -> None:
    bad = {**VALID_CUSTOMER, "tenure_months": 12, "total_charges": 10.0}
    response = api_client.post("/api/v1/predict", json=bad)
    assert response.status_code == 422
