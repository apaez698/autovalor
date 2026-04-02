"""
Flask API for vehicle valuation predictions.
"""
from flask import Flask, request, jsonify
import sys
import os
import joblib

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.train_model import VehicleValuationModel


app = Flask(__name__)
MODEL_MTIME = None

CATEGORICAL_COLUMNS = [
    "marca",
    "tipo",
    "transmision",
    "combustible",
    "provincia",
    "color",
    "estado_motor",
    "estado_carroceria",
]


def load_label_encoders(models_dir: str = "models"):
    """Load label encoders saved during training from the models directory."""
    encoders = {}
    for column in CATEGORICAL_COLUMNS:
        encoder_path = os.path.join(models_dir, f"{column}_label_encoder.joblib")
        if os.path.exists(encoder_path):
            encoders[column] = joblib.load(encoder_path)
    return encoders


label_encoders = load_label_encoders()


def get_label_encoder(column: str):
    """Get encoder for a column, reloading from disk if needed."""
    if column in label_encoders:
        return label_encoders[column]

    refreshed = load_label_encoders()
    label_encoders.update(refreshed)
    return label_encoders.get(column)

# Load model on startup
model = VehicleValuationModel()
try:
    model.load_model()
    if os.path.exists(model.model_path):
        MODEL_MTIME = os.path.getmtime(model.model_path)
    print("Model loaded successfully")
except FileNotFoundError:
    print("Warning: Pre-trained model not found. Please train a model first.")


def refresh_model_if_updated():
    """Reload model/scaler if the persisted model file was updated."""
    global MODEL_MTIME
    if not os.path.exists(model.model_path):
        return

    current_mtime = os.path.getmtime(model.model_path)
    if MODEL_MTIME is None or current_mtime > MODEL_MTIME or model.model is None:
        model.load_model()
        MODEL_MTIME = current_mtime


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok", "service": "vehicle-valuation-api"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict vehicle valuation.
    
    Expected JSON:
    {
        "features": [feature1, feature2, ...]
    }

    Or structured input using saved encoders:
    {
        "vehicle": {
            "anio": 2024,
            "marca": "TOYOTA",
            "tipo": "SUV",
            "transmision": "AUTOMATICA",
            "combustible": "GASOLINA",
            "provincia": "PICHINCHA",
            "color": "NEGRO",
            "estado_motor": "BUENO",
            "estado_carroceria": "BUENO"
        },
        "feature_order": ["anio", "antiguedad", "marca", ...]
    }
    """
    try:
        refresh_model_if_updated()
        data = request.get_json()

        if not data:
            return jsonify({"error": "Missing JSON body"}), 400

        if "features" in data:
            features = data["features"]
        elif "vehicle" in data:
            vehicle = data["vehicle"]
            if not isinstance(vehicle, dict):
                return jsonify({"error": "'vehicle' must be an object"}), 400

            if "anio" not in vehicle:
                return jsonify({"error": "Missing 'anio' in vehicle payload"}), 400

            feature_order = data.get("feature_order")
            if not feature_order:
                return jsonify({
                    "error": "Missing 'feature_order' for structured vehicle payload"
                }), 400

            transformed = dict(vehicle)
            transformed["antiguedad"] = 2026 - float(vehicle["anio"])

            for column in CATEGORICAL_COLUMNS:
                if column not in transformed:
                    return jsonify({"error": f"Missing '{column}' in vehicle payload"}), 400
                encoder = get_label_encoder(column)
                if not encoder:
                    return jsonify({
                        "error": f"Encoder not found for '{column}'. Train and save encoders first."
                    }), 500

                value = str(transformed[column]).strip().upper()
                if value not in encoder.classes_:
                    return jsonify({
                        "error": f"Unknown category '{value}' for '{column}'"
                    }), 400
                transformed[column] = int(encoder.transform([value])[0])

            try:
                features = [float(transformed[col]) for col in feature_order]
            except KeyError as exc:
                return jsonify({"error": f"Missing feature in payload: {exc}"}), 400
        else:
            return jsonify({
                "error": "Missing input. Provide either 'features' or 'vehicle'"
            }), 400
        
        if not model.model:
            return jsonify({"error": "Model not loaded"}), 500
        
        # Scale features and make prediction
        X_scaled = model.scaler.transform([features])
        prediction = model.model.predict(X_scaled)[0]
        
        return jsonify({"predicted_value": float(prediction)}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/info", methods=["GET"])
def info():
    """Get API information."""
    return jsonify({
        "name": "Vehicle Valuation API",
        "version": "1.0.0",
        "endpoints": [
            "/health - Health check",
            "/predict - Make valuation prediction",
            "/info - API information"
        ]
    }), 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
