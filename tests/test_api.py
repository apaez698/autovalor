"""
Tests for the Flask API.
"""
import pytest
import sys
import os
import json
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from api.app import app


class TestFlaskAPI:
    """Test cases for Flask API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create a test client."""
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "ok"
    
    def test_info_endpoint(self, client):
        """Test info endpoint."""
        response = client.get("/info")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "endpoints" in data
        assert "name" in data
    
    def test_predict_missing_features(self, client):
        """Test predict endpoint with missing features."""
        response = client.post("/predict", json={})
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data
    
    @patch('api.app.model')
    def test_predict_success(self, mock_model, client):
        """Test successful prediction."""
        mock_model.model = MagicMock()
        mock_model.model.predict = MagicMock(return_value=[50000])
        mock_model.scaler = MagicMock()
        mock_model.scaler.transform = MagicMock(return_value=[[1, 2, 3, 4, 5]])
        
        response = client.post("/predict", 
                              json={"features": [1, 2, 3, 4, 5]})
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "predicted_value" in data

    @patch('api.app.model')
    def test_predict_with_vehicle_payload(self, mock_model, client):
        """Test structured vehicle payload using saved label encoders."""
        mock_model.model = MagicMock()
        mock_model.model.predict = MagicMock(return_value=[42000])
        mock_model.scaler = MagicMock()
        mock_model.scaler.transform = MagicMock(return_value=[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])

        # Mock encoders with known classes and transform outputs
        columns = [
            "marca",
            "tipo",
            "transmision",
            "combustible",
            "provincia",
            "color",
            "estado_motor",
            "estado_carroceria",
        ]
        encoders = {}
        for idx, col in enumerate(columns):
            encoder = MagicMock()
            encoder.classes_ = ["VALOR"]
            encoder.transform = MagicMock(return_value=[idx + 1])
            encoders[col] = encoder

        payload = {
            "vehicle": {
                "anio": 2024,
                "marca": "VALOR",
                "tipo": "VALOR",
                "transmision": "VALOR",
                "combustible": "VALOR",
                "provincia": "VALOR",
                "color": "VALOR",
                "estado_motor": "VALOR",
                "estado_carroceria": "VALOR"
            },
            "feature_order": [
                "anio",
                "antiguedad",
                "marca",
                "tipo",
                "transmision",
                "combustible",
                "provincia",
                "color",
                "estado_motor",
                "estado_carroceria",
            ]
        }

        with patch('api.app.label_encoders', encoders):
            response = client.post("/predict", json=payload)
            assert response.status_code == 200
            data = json.loads(response.data)
            assert "predicted_value" in data
