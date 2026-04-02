"""
Model training module for vehicle valuation ML.
"""
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


class VehicleValuationModel:
    """Train and save vehicle valuation model."""
    
    def __init__(self, model_path: str = "models/vehicle_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.scaler = StandardScaler()
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load training data from CSV."""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        return pd.read_csv(data_path)
    
    def train(self, X_train, y_train):
        """Train the model."""
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        print("Model trained successfully")
    
    def save_model(self):
        """Save model and scaler to disk."""
        os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)
        
        scaler_path = self.model_path.replace(".pkl", "_scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
        print(f"Model saved to {self.model_path}")
    
    def load_model(self):
        """Load model and scaler from disk."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
        
        scaler_path = self.model_path.replace(".pkl", "_scaler.pkl")
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
        print(f"Model loaded from {self.model_path}")


if __name__ == "__main__":
    print("Vehicle Valuation Model Training Module")
