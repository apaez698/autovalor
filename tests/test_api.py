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
