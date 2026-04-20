"""Model service — loads XGBoost model, label encoders, catalog, and handles predictions."""

import json
import os
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from fastapi import HTTPException

from ml.feature_config import CATEGORICAL_COLUMNS, FEATURE_ORDER, VALIDATION_LIMITS

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
CATALOG_PATH = MODELS_DIR / "vehicle_catalog.json"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
MODEL_PATH = MODELS_DIR / "vehicle_model.pkl"


model = None
label_encoders: dict = {}
model_metadata: dict = {}
vehicle_catalog: list[dict] = []


def load_resources():
    global model, label_encoders, model_metadata, vehicle_catalog

    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)

    for column in CATEGORICAL_COLUMNS:
        encoder_path = MODELS_DIR / f"{column}_label_encoder.joblib"
        if encoder_path.exists():
            label_encoders[column] = joblib.load(encoder_path)

    if METADATA_PATH.exists():
        with METADATA_PATH.open("r", encoding="utf-8") as f:
            model_metadata = json.load(f)

    if CATALOG_PATH.exists():
        with CATALOG_PATH.open("r", encoding="utf-8") as f:
            vehicle_catalog = json.load(f)


def strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def predict(vehicle) -> dict:
    """Run XGBoost prediction for a VehicleInput."""
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")

    current_year = datetime.now().year
    transformed = {
        "anio": float(vehicle.anio),
        "antiguedad": float(current_year - vehicle.anio),
        "kilometraje": float(vehicle.kilometraje),
        "motor_cc": float(vehicle.motor_cc),
        "potencia_hp": float(vehicle.potencia_hp),
    }

    invalid = {}
    for column in CATEGORICAL_COLUMNS:
        encoder = label_encoders.get(column)
        if encoder is None:
            raise HTTPException(status_code=503, detail=f"Encoder faltante: {column}")

        value = getattr(vehicle, column).strip().upper()
        classes = list(getattr(encoder, "classes_", []))

        if value in set(classes):
            transformed[column] = int(encoder.transform([value])[0])
            continue

        value_stripped = strip_accents(value)
        match = next(
            (c for c in classes if strip_accents(str(c)) == value_stripped),
            None,
        )
        if match is not None:
            transformed[column] = int(encoder.transform([match])[0])
            continue

        invalid[column] = {
            "recibido": value,
            "opciones_validas": sorted(str(c) for c in classes),
        }

    if invalid:
        raise HTTPException(status_code=422, detail={"campos_invalidos": invalid})

    feature_order = model_metadata.get("feature_order", FEATURE_ORDER)
    features = [float(transformed[f]) for f in feature_order]
    features_np = np.asarray([features], dtype=float)

    prediction = float(model.predict(features_np)[0])

    return {
        "predicted_value": round(prediction, 2),
        "precio_estimado": round(prediction, 2),
        "estimated_price": round(prediction, 2),
        "price_range": {
            "min": round(prediction * 0.85, 2),
            "max": round(prediction * 1.15, 2),
            "tolerance_percent": 15,
        },
        "rango_precio": {
            "min": round(prediction * 0.85, 2),
            "max": round(prediction * 1.15, 2),
        },
        "vehiculo": vehicle.model_dump(),
        "modelo_metricas": model_metadata.get("metrics", {}),
    }
