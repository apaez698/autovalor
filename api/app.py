"""
Flask API for vehicle valuation predictions.
"""
from flask import Flask, request, jsonify
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.train_model import VehicleValuationModel


app = Flask(__name__)

# Load model on startup
model = VehicleValuationModel()
try:
    model.load_model()
    print("Model loaded successfully")
except FileNotFoundError:
    print("Warning: Pre-trained model not found. Please train a model first.")


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
    """
    try:
        data = request.get_json()
        
        if not data or "features" not in data:
            return jsonify({"error": "Missing 'features' in request"}), 400
        
        features = data["features"]
        
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
