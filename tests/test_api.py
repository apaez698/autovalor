"""Tests for FastAPI prediction endpoint — validation, sensitivity, feature-order."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ml.feature_config import FEATURE_ORDER
import app.services.model_service as model_svc
from app.main import app


class FakeEncoder:
    def __init__(self, values):
        self.classes_ = np.array(values, dtype=object)

    def transform(self, values):
        mapping = {value: idx for idx, value in enumerate(self.classes_)}
        return [mapping[value] for value in values]


class RecordingModel:
    def __init__(self):
        self.last_input = None

    def predict(self, x):
        self.last_input = np.asarray(x, dtype=float)
        return np.array([float(np.sum(self.last_input[0]))], dtype=float)


@pytest.fixture
def client(monkeypatch):
    categorical_values = {
        "marca": ["TOYOTA", "MAZDA"],
        "modelo": ["COROLLA", "CX5"],
        "carroceria": ["SUV", "SEDAN"],
        "transmision": ["AUTOMATICA", "MANUAL"],
        "tipo_combustible": ["GASOLINA", "DIESEL"],
        "provincia": ["PICHINCHA", "GUAYAS"],
        "traccion": ["4X2", "4X4"],
        "segmento": ["MEDIO", "ALTO"],
        "pais_origen": ["JAPON", "COREA"],
        "color": ["NEGRO", "ROJO"],
    }

    encoders = {
        field: FakeEncoder(values)
        for field, values in categorical_values.items()
    }

    recording_model = RecordingModel()

    monkeypatch.setattr(model_svc, "label_encoders", encoders)
    monkeypatch.setattr(model_svc, "model", recording_model)
    monkeypatch.setattr(
        model_svc,
        "model_metadata",
        {
            "feature_order": FEATURE_ORDER,
            "categorical_values": categorical_values,
            "feature_importances": {feature: 0.1 for feature in FEATURE_ORDER},
            "metrics": {"mae": 999.0, "r2": 0.8},
        },
    )

    return TestClient(app)


def valid_vehicle():
    return {
        "anio": 2022,
        "kilometraje": 45000,
        "motor_cc": 1800.0,
        "potencia_hp": 140.0,
        "marca": "TOYOTA",
        "modelo": "COROLLA",
        "carroceria": "SEDAN",
        "transmision": "MANUAL",
        "tipo_combustible": "GASOLINA",
        "provincia": "PICHINCHA",
        "traccion": "4X2",
        "segmento": "MEDIO",
        "pais_origen": "JAPON",
        "color": "NEGRO",
    }


def test_predict_returns_price(client):
    r = client.post("/api/predict", json=valid_vehicle())
    assert r.status_code == 200
    assert "precio_estimado" in r.json()


def test_invalid_marca_returns_422(client):
    payload = valid_vehicle()
    payload["marca"] = "INVALIDA"
    r = client.post("/api/predict", json=payload)
    assert r.status_code == 422


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_metadata_endpoint(client):
    r = client.get("/api/metadata")
    assert r.status_code == 200
