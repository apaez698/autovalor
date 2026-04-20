"""
Model training module for vehicle valuation ML.
"""
import os
import json
import pandas as pd
import joblib
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer

try:
    from ml.feature_config import (
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
        "provincia": ["provincia", "Provincia"],
        "kilometraje": ["kilometraje", "Recorrido"],
        "motor_cc": ["motor_cc"],
        "potencia_hp": ["potencia_hp"],
        "carroceria": ["carroceria"],
        "transmision": ["transmision", "Transmision"],
        "tipo_combustible": ["tipo_combustible"],
        "traccion": ["traccion"],
        "segmento": ["segmento"],
        "pais_origen": ["pais_origen"],
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

    # Fill missing categoricals with DESCONOCIDO
    for col in ["carroceria", "transmision", "tipo_combustible", "traccion",
                "segmento", "pais_origen"]:
        if col not in normalized.columns:
            normalized[col] = "DESCONOCIDO"
        else:
            normalized[col] = normalized[col].fillna("DESCONOCIDO")

    normalized["anio"] = pd.to_numeric(normalized.get("anio"), errors="coerce")
    normalized["kilometraje"] = pd.to_numeric(normalized.get("kilometraje"), errors="coerce")
    normalized["motor_cc"] = pd.to_numeric(normalized.get("motor_cc"), errors="coerce")
    normalized["potencia_hp"] = pd.to_numeric(normalized.get("potencia_hp"), errors="coerce")

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
        "Marca",
        "Modelo",
        "Año",
        "Recorrido",
        "Precio Final Editado",
    }

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        missing_sorted = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_sorted}")

    pricing_columns = [
        col for col in [
            "Precio Nuevo Pricing",
            "Precio Nuevo Editado",
            "Precio Final Pricing",
            "Precio Final Editado",
        ]
        if col in df.columns
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
    for col in ["Año", "Recorrido"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    target_col = "Precio Final Editado"
    # Use percentile-based capping (1st-99th) instead of IQR.
    # Vehicle prices are legitimately right-skewed ($3k econobox to $130k
    # Land Cruiser), so IQR×1.5 clips too aggressively and prevents the
    # model from learning high-value predictions.
    lower_bound = df[target_col].quantile(0.01)
    upper_bound = df[target_col].quantile(0.99)
    before_clip = len(df)
    df[target_col] = df[target_col].clip(lower=lower_bound, upper=upper_bound)

    print(
        f"Capped '{target_col}' outliers using 1st-99th percentile: "
        f"[{lower_bound:.2f}, {upper_bound:.2f}]"
    )

    return df


def prepare_features(df: pd.DataFrame, output_dir: str = "models"):
    """Prepare model features and target from a vehicle DataFrame.

    This function:
    1. Calculates ``antiguedad = 2026 - anio``.
    2. Encodes categorical columns using ``LabelEncoder``.
    3. Saves each fitted encoder to *output_dir* with ``joblib``.
    4. Returns feature matrix ``X`` and target vector ``y``.

    Args:
        output_dir: Directory to save encoder files. Defaults to ``models/``.
    """
    normalized = build_normalized_dataframe(df, include_target=True)

    if "target" not in normalized.columns:
        raise ValueError("Missing target column. Expected one of: precio en el que se compro, Precio Final Editado")

    required_columns = ["anio", "kilometraje", "motor_cc", "potencia_hp", *CATEGORICAL_COLUMNS]
    missing_columns = [col for col in required_columns if col not in normalized.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns for feature preparation: {missing}")

    normalized = normalized.dropna(subset=["anio", "kilometraje", "target"]).copy()

    # Fill motor_cc — validate range (600-8000 cc) and fill missing with median
    motor_cc_min = VALIDATION_LIMITS["motor_cc"]["min"]
    motor_cc_max = VALIDATION_LIMITS["motor_cc"]["max"]
    normalized.loc[
        ~normalized["motor_cc"].between(motor_cc_min, motor_cc_max, inclusive="both"),
        "motor_cc",
    ] = np.nan
    if normalized["motor_cc"].isna().all():
        normalized["motor_cc"] = 1600.0
    else:
        normalized["motor_cc"] = normalized["motor_cc"].fillna(normalized["motor_cc"].median())

    # Fill potencia_hp — validate range and fill missing with median
    hp_min = VALIDATION_LIMITS["potencia_hp"]["min"]
    hp_max = VALIDATION_LIMITS["potencia_hp"]["max"]
    normalized.loc[
        ~normalized["potencia_hp"].between(hp_min, hp_max, inclusive="both"),
        "potencia_hp",
    ] = np.nan
    if normalized["potencia_hp"].isna().all():
        normalized["potencia_hp"] = 120.0
    else:
        normalized["potencia_hp"] = normalized["potencia_hp"].fillna(normalized["potencia_hp"].median())

    normalized["anio"] = normalized["anio"].astype(int)
    normalized["kilometraje"] = normalized["kilometraje"].astype(float).clip(lower=0)
    normalized["target"] = normalized["target"].astype(float)
    normalized["antiguedad"] = 2026 - normalized["anio"]

    data = normalized

    os.makedirs(output_dir, exist_ok=True)
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
        encoder_path = os.path.join(output_dir, f"{column}_label_encoder.joblib")
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


if __name__ == "__main__":
    print("Vehicle Valuation Model Training Module")
    dataset_path = os.path.join("data", "data_limpia_entrenamiento.csv")
    cleaned_df = load_and_clean_data(dataset_path)
    X_data, y_data = prepare_features(cleaned_df)
    trained_model, training_metrics = train_model(X_data, y_data)
    export_metadata(cleaned_df, trained_model, X_data.columns.tolist())
    print("Training metrics:")
    print(training_metrics)
