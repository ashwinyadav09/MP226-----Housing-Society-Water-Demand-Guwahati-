"""Tests for the FastAPI prediction service using TestClient."""

from fastapi.testclient import TestClient
import pytest
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    """Verify GET /health returns 200 OK and expected status body."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_valid_payload(sample_predict_payload):
    """Verify POST /predict returns 200 and a positive float prediction."""
    response = client.post("/predict", json=sample_predict_payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], (int, float))
    assert data["prediction"] > 0


def test_rule_1_supply_hours_above_20_rejected(sample_predict_payload):
    """Client rule 1: supply_hours > 20 must be rejected with HTTP 422."""
    bad_payload = dict(sample_predict_payload, supply_hours=24.0)
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_rule_2_negative_garden_area_rejected(sample_predict_payload):
    """Client rule 2: garden_area_m2 < 0 must be rejected with HTTP 422."""
    bad_payload = dict(sample_predict_payload, garden_area_m2=-15.0)
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_missing_field_rejected(sample_predict_payload):
    """Missing required field must be rejected with HTTP 422."""
    bad_payload = dict(sample_predict_payload)
    del bad_payload["flats_occupied"]
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
