"""Shared feature contract for training and inference."""

from datetime import datetime

CURRENT_YEAR = datetime.now().year

CATEGORICAL_COLUMNS = [
    "marca",
    "modelo",
    "carroceria",
    "transmision",
    "tipo_combustible",
    "provincia",
    "traccion",
    "segmento",
    "pais_origen",
    "color",
]

NUMERIC_COLUMNS = [
    "anio",
    "antiguedad",
    "kilometraje",
    "motor_cc",
    "potencia_hp",
]

FEATURE_ORDER = [*NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS]

VALIDATION_LIMITS = {
    "anio": {"min": 1980, "max": CURRENT_YEAR + 1},
    "kilometraje": {"min": 0, "max": 1_000_000},
    "motor_cc": {"min": 600, "max": 8000},
    "potencia_hp": {"min": 30, "max": 800},
}
