"""Prediction routes."""

from fastapi import APIRouter

from app.schemas import VehicleInput
from app.services.model_service import predict

router = APIRouter()


@router.post("/api/predict")
def predict_price(vehicle: VehicleInput):
    """Predice el precio de compra de un vehículo usando XGBoost."""
    return predict(vehicle)
