"""
Tests for the vehicle valuation model.
"""
import pytest
import os
import sys
import numpy as np
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.train_model import VehicleValuationModel


class TestVehicleValuationModel:
    """Test cases for VehicleValuationModel."""
    
    @pytest.fixture
    def model(self):
        """Create a fresh model instance for each test."""
        return VehicleValuationModel(model_path="models/test_model.pkl")
    
    def test_model_initialization(self, model):
        """Test that model initializes correctly."""
        assert model.model is None
        assert model.scaler is not None
    
    def test_load_nonexistent_data(self, model):
        """Test loading non-existent data raises error."""
        with pytest.raises(FileNotFoundError):
            model.load_data("nonexistent_file.csv")
    
    def test_train_model(self, model):
        """Test model training."""
        X_train = np.random.rand(100, 5)
        y_train = np.random.rand(100)
        
        model.train(X_train, y_train)
        assert model.model is not None
        
        # Test prediction
        X_test = np.random.rand(10, 5)
        X_scaled = model.scaler.transform(X_test)
        predictions = model.model.predict(X_scaled)
        assert len(predictions) == 10


class TestModelPersistence:
    """Test model saving and loading."""
    
    @pytest.fixture
    def model(self):
        """Create a model with sample data."""
        m = VehicleValuationModel(model_path="models/test_persistence.pkl")
        X_train = np.random.rand(50, 5)
        y_train = np.random.rand(50)
        m.train(X_train, y_train)
        return m
    
    def test_save_and_load(self, model):
        """Test saving and loading model."""
        model.save_model()
        
        # Create new instance and load
        new_model = VehicleValuationModel(model_path="models/test_persistence.pkl")
        new_model.load_model()
        
        assert new_model.model is not None
        assert new_model.scaler is not None
        
        # Cleanup
        if os.path.exists(model.model_path):
            os.remove(model.model_path)
        scaler_path = model.model_path.replace(".pkl", "_scaler.pkl")
        if os.path.exists(scaler_path):
            os.remove(scaler_path)
