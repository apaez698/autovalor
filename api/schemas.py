"""Marshmallow schemas for vehicle input validation."""

import marshmallow as ma
from marshmallow import ValidationError, fields, validate, validates

try:
    from src.feature_config import VALIDATION_LIMITS
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.feature_config import VALIDATION_LIMITS

CATEGORICAL_FIELDS = [
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


class VehicleInputSchema(ma.Schema):
    """Validate a structured vehicle prediction request body.

    Pass ``valid_options`` dict at instantiation to enable categorical value
    validation against encoder-derived or metadata-derived allowed values::

        schema = VehicleInputSchema(valid_options={"marca": ["TOYOTA", ...]})
    """

    class Meta:
        # Silently ignore keys that are not declared on the schema
        # (e.g. legacy clients sending extra fields).
        unknown = ma.EXCLUDE

    def __init__(self, *args, valid_options: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._valid_options: dict = valid_options or {}

    anio = fields.Integer(
        required=True,
        metadata={"description": "Model year"},
        validate=validate.Range(
            min=VALIDATION_LIMITS["anio"]["min"],
            max=VALIDATION_LIMITS["anio"]["max"],
            error=(
                f"Must be between {VALIDATION_LIMITS['anio']['min']} "
                f"and {VALIDATION_LIMITS['anio']['max']}."
            ),
        ),
    )
    kilometraje = fields.Integer(
        required=True,
        metadata={"description": "Odometer reading in km (>= 0)"},
        validate=validate.Range(
            min=VALIDATION_LIMITS["kilometraje"]["min"],
            max=VALIDATION_LIMITS["kilometraje"]["max"],
            error=(
                f"Must be between {VALIDATION_LIMITS['kilometraje']['min']} "
                f"and {VALIDATION_LIMITS['kilometraje']['max']}."
            ),
        ),
    )
    cilindrada = fields.Float(
        required=True,
        metadata={"description": "Engine displacement in litres"},
        validate=validate.Range(
            min=VALIDATION_LIMITS["cilindrada"]["min"],
            max=VALIDATION_LIMITS["cilindrada"]["max"],
            error=(
                f"Must be between {VALIDATION_LIMITS['cilindrada']['min']} "
                f"and {VALIDATION_LIMITS['cilindrada']['max']}."
            ),
        ),
    )
    marca = fields.String(required=True)
    modelo = fields.String(required=True)
    tipo = fields.String(required=True)
    transmision = fields.String(required=True)
    combustible = fields.String(required=True)
    provincia = fields.String(required=True)
    color = fields.String(required=True)
    estado_motor = fields.String(required=True)
    estado_carroceria = fields.String(required=True)

    # ------------------------------------------------------------------
    # Categorical helpers
    # ------------------------------------------------------------------

    def _check_categorical(self, field_name: str, value: str) -> None:
        """Raise ValidationError when *value* is not in the allowed set."""
        allowed = self._valid_options.get(field_name)
        if not allowed:
            return
        allowed_upper = {str(v).strip().upper() for v in allowed}
        if str(value).strip().upper() not in allowed_upper:
            raise ValidationError(
                f"Invalid value '{value}'. "
                f"Valid options: {', '.join(sorted(allowed_upper))}."
            )

    @validates("marca", "modelo", "tipo", "transmision", "combustible",
               "provincia", "color", "estado_motor", "estado_carroceria")
    def validate_categorical(self, value: str, *, data_key: str, **kwargs) -> None:
        self._check_categorical(data_key, value)
