"""Shared feature contract for training and inference."""

from datetime import datetime

CURRENT_YEAR = datetime.now().year

CATEGORICAL_COLUMNS = [
    "marca",
    "modelo",
    "tipo",
    "transmision",
    "combustible",
    "provincia",
    "color",
    "estado_motor",
    "estado_carroceria",
]

NUMERIC_COLUMNS = [
    "anio",
    "antiguedad",
    "kilometraje",
    "cilindrada",
]

FEATURE_ORDER = [*NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS]

VALIDATION_LIMITS = {
    "anio": {"min": 1980, "max": CURRENT_YEAR + 1},
    "kilometraje": {"min": 0, "max": 1_000_000},
    "cilindrada": {"min": 0.6, "max": 8.0},
}
