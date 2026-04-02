"""Tests for API validation, sensitivity, and feature-order contract."""

import json

import numpy as np
import pytest

import api.app as app_module


FEATURE_ORDER = [
    "anio",
    "antiguedad",
    "kilometraje",
    "cilindrada",
    "marca",
    "modelo",
    "tipo",
    "transmision",
    "combustible",
    "provincia",
    "color",
    "estado_motor",
    "estado_carroceria",
]


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
    app_module.app.config["TESTING"] = True

    categorical_values = {
        "marca": ["TOYOTA", "MAZDA"],
        "modelo": ["COROLLA", "CX5"],
        "tipo": ["SUV", "SEDAN"],
        "transmision": ["AUTOMATICA", "MANUAL"],
        "combustible": ["GASOLINA", "DIESEL"],
        "provincia": ["QUITO", "GUAYAQUIL"],
        "color": ["NEGRO", "ROJO"],
        "estado_motor": ["BUENO", "REGULAR"],
        "estado_carroceria": ["BUENO", "REGULAR"],
    }

    encoders = {
        field: FakeEncoder(values)
        for field, values in categorical_values.items()
    }

    recording_model = RecordingModel()

    monkeypatch.setattr(app_module, "label_encoders", encoders)
    monkeypatch.setattr(app_module, "model", recording_model)
    monkeypatch.setattr(app_module, "scaler", None)
    monkeypatch.setattr(
        app_module,
        "model_metadata",
        {
            "feature_order": FEATURE_ORDER,
            "categorical_values": categorical_values,
            "feature_importances": {feature: 0.1 for feature in FEATURE_ORDER},
            "metrics": {"mae": 999.0, "r2": 0.8},
        },
    )

    with app_module.app.test_client() as test_client:
        yield test_client


def valid_vehicle_payload():
    return {
        "vehicle": {
            "anio": 2022,
            "kilometraje": 45000,
            "cilindrada": 1.8,
            "marca": "TOYOTA",
            "modelo": "COROLLA",
            "tipo": "SEDAN",
            "transmision": "MANUAL",
            "combustible": "GASOLINA",
            "provincia": "QUITO",
            "color": "NEGRO",
            "estado_motor": "BUENO",
            "estado_carroceria": "BUENO",
        }
    }


def extract_prediction(response):
    assert response.status_code == 200, response.get_data(as_text=True)
    data = json.loads(response.data)
    return data["predicted_value"]


def test_invalid_payload_returns_422_with_field_details(client):
    payload = {
        "vehicle": {
            "anio": 2022,
            "kilometraje": -1,
            "cilindrada": 20.0,
            "marca": "INVALIDA",
            "modelo": "COROLLA",
            "tipo": "SEDAN",
            "transmision": "MANUAL",
            "combustible": "GASOLINA",
            "provincia": "QUITO",
            "color": "NEGRO",
            "estado_motor": "BUENO",
            "estado_carroceria": "BUENO",
        }
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422

    details = json.loads(response.data)["error"]["details"]
    assert "kilometraje" in details
    assert "cilindrada" in details
    assert "marca" in details


def test_sensitivity_kilometraje_changes_prediction(client):
    payload = valid_vehicle_payload()
    base = extract_prediction(client.post("/predict", json=payload))

    payload_km = valid_vehicle_payload()
    payload_km["vehicle"]["kilometraje"] = 90000
    changed = extract_prediction(client.post("/predict", json=payload_km))

    assert changed != base


def test_sensitivity_cilindrada_changes_prediction(client):
    payload = valid_vehicle_payload()
    base = extract_prediction(client.post("/predict", json=payload))

    payload_cc = valid_vehicle_payload()
    payload_cc["vehicle"]["cilindrada"] = 2.5
    changed = extract_prediction(client.post("/predict", json=payload_cc))

    assert changed != base


def test_sensitivity_modelo_changes_prediction(client):
    payload = valid_vehicle_payload()
    base = extract_prediction(client.post("/predict", json=payload))

    payload_modelo = valid_vehicle_payload()
    payload_modelo["vehicle"]["modelo"] = "CX5"
    changed = extract_prediction(client.post("/predict", json=payload_modelo))

    assert changed != base


def test_sensitivity_marca_changes_prediction(client):
    payload = valid_vehicle_payload()
    base = extract_prediction(client.post("/predict", json=payload))

    payload_marca = valid_vehicle_payload()
    payload_marca["vehicle"]["marca"] = "MAZDA"
    changed = extract_prediction(client.post("/predict", json=payload_marca))

    assert changed != base


def test_metadata_feature_order_matches_predict_vector_order(client):
    metadata_response = client.get("/metadata")
    assert metadata_response.status_code == 200
    feature_order = json.loads(metadata_response.data)["feature_order"]

    payload = valid_vehicle_payload()
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    used_vector = app_module.model.last_input[0]
    assert len(used_vector) == len(feature_order)

    index_map = {name: idx for idx, name in enumerate(feature_order)}
    assert used_vector[index_map["kilometraje"]] == pytest.approx(45000.0)
    assert used_vector[index_map["cilindrada"]] == pytest.approx(1.8)
