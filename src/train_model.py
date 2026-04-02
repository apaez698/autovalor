"""
Model training module for vehicle valuation ML.
"""
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """Load and clean a vehicle dataset from CSV or Excel.

    This function is adapted to the project's schema and keeps all existing
    columns while cleaning data needed for pricing analysis.

    Steps performed:
    1. Load from CSV or Excel (`.xls` / `.xlsx`).
    2. Print DataFrame shape and dtypes.
    3. Print null count per column.
    4. Remove duplicate rows.
    5. Convert pricing columns to numeric.
    6. Cap outliers in `Precio Final Editado` using the IQR method.

    Args:
        filepath: Path to the input dataset file.

    Returns:
        A cleaned pandas DataFrame with the same columns as input.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If the file type is unsupported or required columns are
            missing.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    _, ext = os.path.splitext(filepath.lower())
    if ext == ".csv":
        df = pd.read_csv(filepath)
    elif ext in {".xls", ".xlsx"}:
        df = pd.read_excel(filepath)
    else:
        raise ValueError("Unsupported file format. Use CSV or Excel (.xls/.xlsx).")

    # Normalize column names for matching while preserving original headers.
    df.columns = df.columns.str.strip()

    print(f"Dataset shape: {df.shape}")
    print("\nData types:")
    print(df.dtypes)
    print("\nNull values per column:")
    print(df.isnull().sum())

    before = len(df)
    df = df.drop_duplicates().copy()
    print(f"\nRemoved duplicates: {before - len(df)}")

    required_columns = {
        "Fecha",
        "Instancia",
        "Empresa",
        "Agencia",
        "Línea Negocio",
        "Solicitante",
        "Coordinador",
        "Número Avalúo Sugar",
        "Número Avalúo Balcón",
        "Identificación",
        "Apellidos Cliente",
        "Nombres Cliente",
        "Teléfono Cliente",
        "Estado Seguimiento",
        "Placa",
        "Marca",
        "Modelo",
        "Año",
        "Color",
        "Recorrido",
        "Tipo Recorrido",
        "Descripción",
        "Comentario",
        "Precio Nuevo Pricing",
        "Precio Nuevo Editado",
        "Precio Final Pricing",
        "Precio Final Editado",
        "Observación Precio",
    }

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        missing_sorted = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_sorted}")

    pricing_columns = [
        "Precio Nuevo Pricing",
        "Precio Nuevo Editado",
        "Precio Final Pricing",
        "Precio Final Editado",
    ]

    # Convert price-like strings to numbers (handles commas, currency signs).
    for col in pricing_columns:
        cleaned = (
            df[col]
            .astype(str)
            .str.replace(r"[^0-9.\-]", "", regex=True)
            .replace("", pd.NA)
        )
        df[col] = pd.to_numeric(cleaned, errors="coerce")

    # Also convert key numeric columns commonly used for valuation models.
    for col in ["Año", "Recorrido", "Identificación", "Teléfono Cliente"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    target_col = "Precio Final Editado"
    q1 = df[target_col].quantile(0.25)
    q3 = df[target_col].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df[target_col] = df[target_col].clip(lower=lower_bound, upper=upper_bound)

    print(
        f"Capped '{target_col}' outliers using IQR bounds: "
        f"[{lower_bound:.2f}, {upper_bound:.2f}]"
    )

    return df


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
