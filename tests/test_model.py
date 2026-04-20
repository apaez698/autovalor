"""Tests for training pipeline feature contract and metadata."""

import json
import os

import numpy as np
import pandas as pd

from ml.feature_config import FEATURE_ORDER
from ml.train import export_metadata, prepare_features


REQUIRED_FEATURES = {"kilometraje", "motor_cc", "potencia_hp", "modelo", "marca"}


def build_training_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Año": [2020, 2021, 2019, 2022, 2018],
            "Marca": ["TOYOTA", "MAZDA", "TOYOTA", "KIA", "NISSAN"],
            "Modelo": ["COROLLA", "CX5", "YARIS", "RIO", "SENTRA"],
            "Color": ["NEGRO", "ROJO", "AZUL", "BLANCO", "PLATA"],
            "Recorrido": [50000, 30000, 70000, 20000, 90000],
            "carroceria": ["sedán", "SUV", "hatchback", "sedán", "sedán"],
            "motor_cc": [1800.0, 2500.0, 1500.0, 1400.0, 2000.0],
            "potencia_hp": [140.0, 187.0, 107.0, 100.0, 149.0],
            "transmision": ["manual", "automática", "manual", "manual", "automática"],
            "traccion": ["4x2", "4x4", "4x2", "4x2", "4x2"],
            "tipo_combustible": ["gasolina", "gasolina", "gasolina", "gasolina", "gasolina"],
            "provincia": ["Pichincha", "Guayas", "Pichincha", "Azuay", "Guayas"],
            "segmento": ["medio", "medio-alto", "económico", "económico", "medio"],
            "pais_origen": ["Japón", "Japón", "Japón", "Corea del Sur", "Japón"],
            "Precio Final Editado": [15000, 22000, 12000, 18000, 9000],
        }
    )


def test_prepare_features_contains_required_columns(tmp_path):
    df = build_training_df()
    X, y = prepare_features(df, output_dir=str(tmp_path))

    assert REQUIRED_FEATURES.issubset(set(X.columns))
    assert X.columns.tolist() == FEATURE_ORDER
    assert len(X) == len(y)


def test_export_metadata_includes_feature_order_and_importance(tmp_path):
    df = build_training_df()

    class FakeModel:
        feature_importances_ = np.array([0.1] * len(FEATURE_ORDER), dtype=float)
        training_metrics = {"mae": 1000.0, "r2": 0.7}

    model = FakeModel()
    previous_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        metadata_path = export_metadata(df, model, FEATURE_ORDER)
        metadata_full_path = os.path.join(tmp_path, metadata_path)
    finally:
        os.chdir(previous_cwd)

    assert os.path.exists(metadata_full_path)

    with open(metadata_full_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    assert metadata["feature_order"] == FEATURE_ORDER
    assert REQUIRED_FEATURES.issubset(set(metadata["feature_importances"].keys()))
    assert "categorical_values" in metadata
    assert "metrics" in metadata
