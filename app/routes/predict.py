"""Prediction routes."""

from fastapi import APIRouter, HTTPException

from app.schemas import VehicleInput, PredictRequest
from app.services.model_service import predict

router = APIRouter()


@router.post("/api/predict")
def predict_price(vehicle: VehicleInput):
    """Predice el precio de compra de un vehículo usando XGBoost."""
    return predict(vehicle)


@router.post("/predict")
async def predict_legacy(body: PredictRequest):
    """Legacy route — accepts both { vehicle: {...} } and flat field format."""
    if body.vehicle:
        return predict(body.vehicle)
    if body.marca:
        v = VehicleInput(
            marca=body.marca, modelo=body.modelo, anio=body.anio,
            kilometraje=body.kilometraje, motor_cc=body.motor_cc,
            potencia_hp=body.potencia_hp, carroceria=body.carroceria,
            transmision=body.transmision, tipo_combustible=body.tipo_combustible,
            provincia=body.provincia, traccion=body.traccion,
            segmento=body.segmento, pais_origen=body.pais_origen,
            color=body.color,
        )
        return predict(v)
    raise HTTPException(status_code=422, detail="Provide 'vehicle' object or flat fields")
