"""
Model training module for vehicle valuation ML.
"""
import os
import pickle
import json
import pandas as pd
import joblib
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer

try:
    from src.feature_config import (
        CATEGORICAL_COLUMNS,
        FEATURE_ORDER,
        VALIDATION_LIMITS,
    )
except ModuleNotFoundError:
    from feature_config import (  # type: ignore
        CATEGORICAL_COLUMNS,
        FEATURE_ORDER,
        VALIDATION_LIMITS,
    )


def robust_mape(y_true, y_pred, min_target: float = 1000.0) -> float:
    """Compute MAPE (%) excluding very small target values.

    Targets with absolute value below ``min_target`` are excluded to avoid
    unstable percentage errors.
    """
    y_true_np = np.asarray(y_true, dtype=float)
    y_pred_np = np.asarray(y_pred, dtype=float)
    valid_mask = np.abs(y_true_np) >= min_target

    if not valid_mask.any():
        return float("nan")

    return float(
        np.mean(
            np.abs(
                (y_true_np[valid_mask] - y_pred_np[valid_mask])
                / y_true_np[valid_mask]
            )
        )
        * 100
    )


def _extract_cilindrada_from_description(description_series: pd.Series) -> pd.Series:
    """Extract engine displacement in liters from free-text descriptions."""
    return pd.to_numeric(
        description_series.fillna("").astype(str).str.extract(r"(\d+(?:\.\d+)?)\s*L\b", expand=False),
        errors="coerce",
    )


def build_normalized_dataframe(df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
    """Build normalized dataframe used by both training and metadata export."""
    data = df.copy()

    source_map = {
        "anio": ["anio", "Año"],
        "marca": ["marca", "Marca"],
        "modelo": ["modelo", "Modelo"],
        "color": ["color", "Color"],
        "provincia": ["provincia", "Agencia", "Provincia"],
        "tipo": ["tipo", "Línea Negocio", "Tipo Vehiculo"],
        "kilometraje": ["kilometraje", "Recorrido"],
        "cilindrada": ["cilindrada", "Cilindrada"],
        "transmision": ["transmision", "Transmision"],
    }
    if include_target:
        source_map["target"] = ["precio en el que se compro", "Precio Final Editado"]

    normalized = pd.DataFrame(index=data.index)
    for target_col, candidates in source_map.items():
        for candidate in candidates:
            if candidate in data.columns:
                normalized[target_col] = data[candidate]
                break

    if "modelo" not in normalized.columns:
        normalized["modelo"] = "DESCONOCIDO"

    desc = data.get("Descripción", pd.Series("", index=data.index)).fillna("").astype(str)
    comments = data.get("Comentario", pd.Series("", index=data.index)).fillna("").astype(str)
    obs = data.get("Observación Precio", pd.Series("", index=data.index)).fillna("").astype(str)
    combined_notes = (comments + " " + obs).str.upper()

    if "transmision" not in normalized.columns:
        normalized["transmision"] = "DESCONOCIDO"
        normalized.loc[desc.str.contains(r"\bT/A\b|\bCVT\b", case=False, regex=True), "transmision"] = "AUTOMATICA"
        normalized.loc[desc.str.contains(r"\bT/M\b", case=False, regex=True), "transmision"] = "MANUAL"

    if "combustible" not in normalized.columns:
        normalized["combustible"] = "DESCONOCIDO"
        normalized.loc[desc.str.contains(r"HYBRID|HSD|HIBRID", case=False, regex=True), "combustible"] = "HIBRIDO"
        normalized.loc[desc.str.contains(r"CRDI|DIESEL", case=False, regex=True), "combustible"] = "DIESEL"
        normalized.loc[desc.str.contains(r"TURBO|VVTI|DUALJET|BOOSTERJET|EFI", case=False, regex=True), "combustible"] = "GASOLINA"

    if "estado_motor" not in normalized.columns:
        normalized["estado_motor"] = "DESCONOCIDO"
        normalized.loc[combined_notes.str.contains(r"FALLA|MECAN", regex=True), "estado_motor"] = "REGULAR"
        normalized.loc[combined_notes.str.contains(r"MANTENIMIENTO|MANTENIMIENTOS EN CASA", regex=True), "estado_motor"] = "BUENO"

    if "estado_carroceria" not in normalized.columns:
        normalized["estado_carroceria"] = "DESCONOCIDO"
        normalized.loc[combined_notes.str.contains(r"PINTAR|DETALLES", regex=True), "estado_carroceria"] = "REGULAR"

    normalized["anio"] = pd.to_numeric(normalized.get("anio"), errors="coerce")
    normalized["kilometraje"] = pd.to_numeric(normalized.get("kilometraje"), errors="coerce")

    parsed_cilindrada = _extract_cilindrada_from_description(desc)
    normalized["cilindrada"] = pd.to_numeric(normalized.get("cilindrada"), errors="coerce")
    normalized["cilindrada"] = normalized["cilindrada"].fillna(parsed_cilindrada)

    if include_target:
        normalized["target"] = pd.to_numeric(normalized.get("target"), errors="coerce")

    return normalized


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


def prepare_features(df: pd.DataFrame):
    """Prepare model features and target from a vehicle DataFrame.

    This function:
    1. Calculates ``antiguedad = 2026 - anio``.
    2. Encodes categorical columns using ``LabelEncoder``.
    3. Saves each fitted encoder to ``models/`` with ``joblib``.
    4. Returns feature matrix ``X`` and target vector ``y``.

    Expected target column:
    - ``precio en el que se compro``
    """
    normalized = build_normalized_dataframe(df, include_target=True)

    if "target" not in normalized.columns:
        raise ValueError("Missing target column. Expected one of: precio en el que se compro, Precio Final Editado")

    required_columns = ["anio", "kilometraje", "cilindrada", *CATEGORICAL_COLUMNS]
    missing_columns = [col for col in required_columns if col not in normalized.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns for feature preparation: {missing}")

    normalized = normalized.dropna(subset=["anio", "kilometraje", "target"]).copy()

    cilindrada_min = VALIDATION_LIMITS["cilindrada"]["min"]
    cilindrada_max = VALIDATION_LIMITS["cilindrada"]["max"]
    normalized.loc[
        ~normalized["cilindrada"].between(cilindrada_min, cilindrada_max, inclusive="both"),
        "cilindrada",
    ] = np.nan
    if normalized["cilindrada"].isna().all():
        normalized["cilindrada"] = 1.6
    else:
        normalized["cilindrada"] = normalized["cilindrada"].fillna(normalized["cilindrada"].median())

    normalized["anio"] = normalized["anio"].astype(int)
    normalized["kilometraje"] = normalized["kilometraje"].astype(float).clip(lower=0)
    normalized["target"] = normalized["target"].astype(float)
    normalized["antiguedad"] = 2026 - normalized["anio"]

    data = normalized

    os.makedirs("models", exist_ok=True)
    for column in CATEGORICAL_COLUMNS:
        encoder = LabelEncoder()
        data[column] = (
            data[column]
            .fillna("desconocido")
            .astype(str)
            .str.strip()
            .str.upper()
        )
        data[column] = encoder.fit_transform(data[column])
        encoder_path = os.path.join("models", f"{column}_label_encoder.joblib")
        joblib.dump(encoder, encoder_path)

    y = data["target"]
    X = data[FEATURE_ORDER].copy()
    return X, y


def train_model(X, y):
    """Train an XGBoost regressor and return model plus metrics.

    Steps:
    1. Split data into train/test sets using 80/20.
    2. Run 5-fold cross-validation on the training split.
    3. Train a final model on the training split.
    4. Evaluate holdout MAE, R2, and MAPE on test split.
    5. Persist trained model to models/vehicle_model.pkl with joblib.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror",
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    robust_mape_scorer = make_scorer(robust_mape, greater_is_better=False)
    cv_scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
            "mape": robust_mape_scorer,
        },
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, predictions))
    r2 = float(r2_score(y_test, predictions))
    mape = robust_mape(y_test, predictions)

    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "vehicle_model.pkl")
    scaler_path = os.path.join("models", "vehicle_model_scaler.pkl")
    joblib.dump(model, model_path)
    if os.path.exists(scaler_path):
        os.remove(scaler_path)

    metrics = {
        "mae": mae,
        "r2": r2,
        "mape": mape,
        "cv_mae_mean": float(-cv_scores["test_mae"].mean()),
        "cv_mae_std": float(cv_scores["test_mae"].std()),
        "cv_r2_mean": float(cv_scores["test_r2"].mean()),
        "cv_r2_std": float(cv_scores["test_r2"].std()),
        "cv_mape_mean": float(-np.nanmean(cv_scores["test_mape"])),
        "cv_mape_std": float(np.nanstd(cv_scores["test_mape"])),
        "mape_min_target": 1000.0,
        "model_path": model_path,
        "scaler_path": None,
    }

    print(f"MAE: {mae:.4f}")
    print(f"R2: {r2:.4f}")
    print(f"MAPE: {mape:.4f}%")
    print(f"Model saved to {model_path}")

    # Keep metrics attached to the model so metadata export can include them.
    model.training_metrics = metrics

    return model, metrics


def export_metadata(df: pd.DataFrame, model, feature_cols):
    """Export safe model metadata for API/runtime validation.

    Writes ``models/model_metadata.json`` with:
    - Unique category values per categorical feature (from real training data)
    - Feature importances aligned with ``feature_cols``
    - Model metrics, when available on ``model.training_metrics``

    The file intentionally excludes raw records.
    """
    normalized = build_normalized_dataframe(df, include_target=False)

    unique_categories = {}
    for column in CATEGORICAL_COLUMNS:
        if column not in feature_cols or column not in normalized.columns:
            continue

        values = (
            normalized[column]
            .fillna("desconocido")
            .astype(str)
            .str.strip()
            .str.upper()
        )
        unique_categories[column] = sorted(values[values != ""].unique().tolist())

    feature_importances = {}
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
        for idx, feature in enumerate(feature_cols):
            feature_importances[feature] = float(importances[idx])

    metadata = {
        "feature_order": feature_cols,
        "categorical_values": unique_categories,
        "feature_importances": feature_importances,
        "metrics": getattr(model, "training_metrics", {}),
        "validation_limits": VALIDATION_LIMITS,
    }

    os.makedirs("models", exist_ok=True)
    metadata_path = os.path.join("models", "model_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Metadata exported to {metadata_path}")
    return metadata_path


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
    dataset_path = os.path.join("data", "data.csv")
    cleaned_df = load_and_clean_data(dataset_path)
    X_data, y_data = prepare_features(cleaned_df)
    trained_model, training_metrics = train_model(X_data, y_data)
    export_metadata(cleaned_df, trained_model, X_data.columns.tolist())
    print("Training metrics:")
    print(training_metrics)
