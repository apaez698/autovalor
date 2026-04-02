"""Flask REST API for vehicle valuation predictions."""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import joblib
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from marshmallow import ValidationError

try:
    from src.feature_config import CATEGORICAL_COLUMNS, FEATURE_ORDER
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.feature_config import CATEGORICAL_COLUMNS, FEATURE_ORDER

try:
    from api.schemas import VehicleInputSchema
except ModuleNotFoundError:
    # Support running as a script: `python api/app.py`
    from schemas import VehicleInputSchema


app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "vehicle_model.pkl"
SCALER_PATH = MODELS_DIR / "vehicle_model_scaler.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
MODEL_ARTIFACTS_BASE_URL = os.getenv("MODEL_ARTIFACTS_BASE_URL", "").strip().rstrip("/")
MODEL_ARTIFACTS_TOKEN = os.getenv("MODEL_ARTIFACTS_TOKEN", "").strip()
MODEL_ARTIFACTS_TIMEOUT_SECONDS = int(
    os.getenv("MODEL_ARTIFACTS_TIMEOUT_SECONDS", "30")
)

DEFAULT_FEATURE_ORDER = FEATURE_ORDER

model = None
scaler = None
label_encoders = {}
model_metadata = {}
artifact_sync_error = None


def error_response(message: str, status_code: int, details=None):
    """Return a consistent JSON error body."""
    payload = {"error": {"message": message}}
    if details is not None:
        payload["error"]["details"] = details
    return jsonify(payload), status_code


def load_label_encoders():
    """Load all available label encoders from disk."""
    encoders = {}
    for column in CATEGORICAL_COLUMNS:
        encoder_path = MODELS_DIR / f"{column}_label_encoder.joblib"
        if encoder_path.exists():
            encoders[column] = joblib.load(encoder_path)
    return encoders


def load_metadata():
    """Load model metadata JSON if present."""
    if not METADATA_PATH.exists():
        return {}

    with METADATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def metadata_from_encoders():
    """Build categorical valid options from encoder classes."""
    options = {}
    for column, encoder in label_encoders.items():
        classes = getattr(encoder, "classes_", [])
        options[column] = [str(value) for value in classes]
    return options


def download_artifact(artifact_name: str, destination: Path):
    """Download one artifact from the configured base URL."""
    if not MODEL_ARTIFACTS_BASE_URL:
        raise RuntimeError("MODEL_ARTIFACTS_BASE_URL is not configured")

    url = f"{MODEL_ARTIFACTS_BASE_URL}/{artifact_name}"
    headers = {}
    if MODEL_ARTIFACTS_TOKEN:
        headers["Authorization"] = f"Bearer {MODEL_ARTIFACTS_TOKEN}"

    req = Request(url, headers=headers)
    with urlopen(req, timeout=MODEL_ARTIFACTS_TIMEOUT_SECONDS) as response:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as f:
            shutil.copyfileobj(response, f)


def ensure_remote_artifacts():
    """Fetch missing model artifacts from a private remote location."""
    if not MODEL_ARTIFACTS_BASE_URL:
        return None

    required = [
        MODEL_PATH.name,
        METADATA_PATH.name,
        *[f"{column}_label_encoder.joblib" for column in CATEGORICAL_COLUMNS],
    ]
    optional = [SCALER_PATH.name]

    missing_required = []

    for artifact_name in required:
        destination = MODELS_DIR / artifact_name
        if destination.exists():
            continue
        try:
            download_artifact(artifact_name, destination)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            missing_required.append(f"{artifact_name}: {exc}")

    for artifact_name in optional:
        destination = MODELS_DIR / artifact_name
        if destination.exists():
            continue
        try:
            download_artifact(artifact_name, destination)
        except (HTTPError, URLError, TimeoutError, OSError):
            # Optional artifact, ignore failures.
            pass

    if missing_required:
        return "Unable to download required artifacts: " + "; ".join(missing_required)

    return None


def load_resources_on_startup():
    """Load model, scaler, encoders, and metadata into memory."""
    global model, scaler, label_encoders, model_metadata, artifact_sync_error

    artifact_sync_error = ensure_remote_artifacts()

    if MODEL_PATH.exists():
        with MODEL_PATH.open("rb") as f:
            model = joblib.load(f)

    if SCALER_PATH.exists():
        with SCALER_PATH.open("rb") as f:
            scaler = joblib.load(f)

    label_encoders = load_label_encoders()
    model_metadata = load_metadata()

    # Guard against stale scaler artifacts trained with a different feature count.
    if scaler is not None:
        feature_order = model_metadata.get("feature_order", DEFAULT_FEATURE_ORDER)
        expected_features = len(feature_order)
        scaler_features = getattr(scaler, "n_features_in_", expected_features)
        if scaler_features != expected_features:
            scaler = None


def build_health_payload():
    """Build model readiness and inventory details for health endpoint."""
    payload = {
        "status": "ok",
        "service": "vehicle-valuation-api",
        "model": {
            "loaded": model is not None,
            "path": str(MODEL_PATH.relative_to(BASE_DIR)),
            "exists_on_disk": MODEL_PATH.exists(),
            "scaler_loaded": scaler is not None,
            "metadata_loaded": bool(model_metadata),
            "encoders_loaded": sorted(label_encoders.keys()),
        },
    }
    if artifact_sync_error:
        payload["model"]["artifact_sync_error"] = artifact_sync_error
    return payload


def validate_predict_payload(payload):
    """Validate base predict request structure."""
    if not isinstance(payload, dict):
        return "Invalid payload: JSON object expected"

    has_features = "features" in payload
    has_vehicle = "vehicle" in payload

    if has_features == has_vehicle:
        return "Provide exactly one of 'features' or 'vehicle'"

    return None


def encode_vehicle_payload(vehicle: dict, feature_order: list[str]):
    """Validate and encode structured vehicle payload into numeric features."""
    if not isinstance(vehicle, dict):
        raise ValueError("'vehicle' must be a JSON object")

    missing_fields = ["anio", "kilometraje", "cilindrada", *CATEGORICAL_COLUMNS]
    missing = [field for field in missing_fields if field not in vehicle]
    if missing:
        raise KeyError(f"Missing required vehicle fields: {', '.join(missing)}")

    current_year = datetime.now().year
    transformed = {
        "anio": float(vehicle["anio"]),
        "antiguedad": float(current_year - float(vehicle["anio"])),
        "kilometraje": float(vehicle["kilometraje"]),
        "cilindrada": float(vehicle["cilindrada"]),
    }

    invalid_categories = {}
    valid_options = model_metadata.get("categorical_values", {})
    fallback_options = metadata_from_encoders()

    for column in CATEGORICAL_COLUMNS:
        encoder = label_encoders.get(column)
        if encoder is None:
            raise RuntimeError(f"Missing encoder for '{column}'")

        value = str(vehicle[column]).strip().upper()
        allowed = valid_options.get(column) or fallback_options.get(column, [])
        allowed_upper = {str(item).upper() for item in allowed}
        if allowed_upper and value not in allowed_upper:
            invalid_categories[column] = {
                "received": value,
                "valid_options": sorted(allowed_upper),
            }
            continue

        if value not in set(getattr(encoder, "classes_", [])):
            invalid_categories[column] = {
                "received": value,
                "valid_options": [str(v) for v in encoder.classes_],
            }
            continue

        transformed[column] = int(encoder.transform([value])[0])

    if invalid_categories:
        raise ValueError("Invalid categorical values", invalid_categories)

    missing_features = [feature for feature in feature_order if feature not in transformed]
    if missing_features:
        raise KeyError(
            "Missing computed features required by feature_order: "
            + ", ".join(missing_features)
        )

    return [float(transformed[feature]) for feature in feature_order]


def predict_price(features: list[float]):
    """Run model inference with optional scaler transform."""
    if model is None:
        raise RuntimeError("Model is not loaded")

    active_model = model
    active_scaler = scaler
    # Backward compatibility: support wrapper objects exposing .model/.scaler.
    if hasattr(model, "model") and getattr(model, "model") is not None:
        active_model = getattr(model, "model")
    if hasattr(model, "scaler") and getattr(model, "scaler") is not None:
        active_scaler = getattr(model, "scaler")

    features_np = np.asarray([features], dtype=float)
    if active_scaler is not None:
        features_np = active_scaler.transform(features_np)

    prediction = float(active_model.predict(features_np)[0])
    return prediction


@app.route("/health", methods=["GET"])
def health_check():
    """Return API and model readiness details."""
    return jsonify(build_health_payload()), 200


@app.route("/metadata", methods=["GET"])
def metadata():
    """Return valid options per categorical field."""
    categorical_values = model_metadata.get("categorical_values", {})
    if not categorical_values:
        categorical_values = metadata_from_encoders()

    if not categorical_values:
        return error_response(
            "Metadata unavailable. No model_metadata.json or encoder classes found.",
            404,
        )

    feature_order = model_metadata.get("feature_order", DEFAULT_FEATURE_ORDER)

    return jsonify({
        "feature_order": feature_order,
        "categorical_values": categorical_values,
        "feature_importances": model_metadata.get("feature_importances", {}),
        "metrics": model_metadata.get("metrics", {}),
    }), 200


@app.route("/feature-importance", methods=["GET"])
def feature_importance():
    """Return feature importance values from metadata or loaded model."""
    importance = model_metadata.get("feature_importances", {})

    if not importance and model is not None and hasattr(model, "feature_importances_"):
        feature_order = model_metadata.get("feature_order", DEFAULT_FEATURE_ORDER)
        raw = list(getattr(model, "feature_importances_"))
        importance = {
            feature: float(raw[idx])
            for idx, feature in enumerate(feature_order)
            if idx < len(raw)
        }

    if not importance:
        return error_response("Feature importance is not available", 404)

    sorted_importance = [
        {"feature": feature, "importance": value}
        for feature, value in sorted(
            importance.items(), key=lambda item: item[1], reverse=True
        )
    ]
    return jsonify({"feature_importance": sorted_importance}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """Validate request, encode categoricals, and predict vehicle value."""
    if not request.is_json:
        return error_response("Content-Type must be application/json", 415)

    payload = request.get_json(silent=True)
    if payload is None:
        return error_response("Malformed JSON body", 400)

    structure_error = validate_predict_payload(payload)
    if structure_error:
        return error_response(structure_error, 400)

    try:
        if "features" in payload:
            features = payload["features"]
            if not isinstance(features, list) or not features:
                return error_response("'features' must be a non-empty list", 400)
            numeric_features = [float(value) for value in features]
        else:
            feature_order = model_metadata.get("feature_order", DEFAULT_FEATURE_ORDER)
            requested_order = payload.get("feature_order")
            if requested_order is not None and requested_order != feature_order:
                return jsonify(
                    {
                        "error": {
                            "message": "Validation failed",
                            "details": {
                                "feature_order": [
                                    "feature_order must match the model metadata contract"
                                ]
                            },
                        }
                    }
                ), 422

            valid_options = model_metadata.get("categorical_values") or metadata_from_encoders()
            schema = VehicleInputSchema(valid_options=valid_options)
            try:
                schema.load(payload["vehicle"])
            except ValidationError as exc:
                return jsonify(
                    {"error": {"message": "Validation failed", "details": exc.messages}}
                ), 422

            numeric_features = encode_vehicle_payload(payload["vehicle"], feature_order)

        estimated_price = predict_price(numeric_features)
        min_price = estimated_price * 0.85
        max_price = estimated_price * 1.15

        return jsonify(
            {
                "predicted_value": round(estimated_price, 2),
                "estimated_price": round(estimated_price, 2),
                "price_range": {
                    "min": round(min_price, 2),
                    "max": round(max_price, 2),
                    "tolerance_percent": 15,
                },
            }
        ), 200
    except ValueError as exc:
        if len(exc.args) == 2 and isinstance(exc.args[1], dict):
            return jsonify(
                {"error": {"message": "Validation failed", "details": exc.args[1]}}
            ), 422
        return error_response(str(exc), 400)
    except KeyError as exc:
        return jsonify(
            {
                "error": {
                    "message": "Validation failed",
                    "details": {"payload": [str(exc)]},
                }
            }
        ), 422
    except RuntimeError as exc:
        return error_response(str(exc), 503)
    except Exception as exc:
        return error_response("Prediction failed", 500, str(exc))


@app.route("/info", methods=["GET"])
def info():
    """Backward-compatible API info endpoint."""
    return jsonify(
        {
            "name": "Vehicle Valuation API",
            "version": "2.0.0",
            "endpoints": [
                "/health",
                "/metadata",
                "/predict",
                "/feature-importance",
            ],
        }
    ), 200


load_resources_on_startup()


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
