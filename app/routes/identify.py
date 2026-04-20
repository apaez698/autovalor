"""Identify & evaluate routes — photo-based vehicle identification."""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.schemas import VehicleInput, PhotoIDResponse
from app.services import vision_service
from app.services.model_service import predict

router = APIRouter()


@router.post("/api/identify", response_model=PhotoIDResponse)
async def identify_vehicle(
    foto: UploadFile = File(...),
    incluir_specs: bool = Form(default=True),
):
    """Identifica marca/modelo/año de un vehículo a partir de una foto usando Claude Vision."""
    result = await vision_service.identify_vehicle(foto, incluir_specs)
    return PhotoIDResponse(**result)


@router.post("/api/evaluate")
async def evaluate_vehicle(
    foto: UploadFile = File(...),
    precio_compra: float = Form(...),
    kilometraje: Optional[int] = Form(default=None),
    provincia: Optional[str] = Form(default="PICHINCHA"),
):
    """Flujo completo: foto → identifica → predice precio → calcula score."""
    id_result = await vision_service.identify_vehicle(foto, incluir_specs=True)

    marca = id_result["marca"].upper()
    modelo = id_result["modelo"].upper()
    anio = id_result.get("anio_estimado") or datetime.now().year - 3
    carroceria = (id_result.get("carroceria") or "SUV").upper()
    color = (id_result.get("color") or "BLANCO").upper()
    specs = id_result.get("specs_sugeridas") or {}

    precio_estimado = None
    try:
        vehicle = VehicleInput(
            marca=marca,
            modelo=modelo,
            anio=anio,
            kilometraje=kilometraje or 50000,
            motor_cc=specs.get("motor_cc", 1600),
            potencia_hp=specs.get("potencia_hp", 120),
            carroceria=specs.get("carroceria", carroceria),
            transmision=specs.get("transmision", "AUTOMATICA"),
            tipo_combustible=specs.get("tipo_combustible", "GASOLINA"),
            provincia=(provincia or "PICHINCHA").upper(),
            traccion=specs.get("traccion", "4X2"),
            segmento=specs.get("segmento", "MEDIO"),
            pais_origen=specs.get("pais_origen", "DESCONOCIDO"),
            color=color,
        )
        pred_result = predict(vehicle)
        precio_estimado = pred_result["precio_estimado"]
    except HTTPException:
        pass

    score_result = None
    if precio_estimado:
        precio_venta_est = precio_estimado * 0.88
        margen_bruto = precio_venta_est - precio_compra
        margen_pct = (margen_bruto / precio_compra) * 100 if precio_compra > 0 else 0

        if margen_pct >= 20:
            score, rec = "EXCELENTE", "Compra inmediata — margen alto"
        elif margen_pct >= 12:
            score, rec = "BUENO", "Buen negocio — margen sano para reventa"
        elif margen_pct >= 5:
            score, rec = "REGULAR", "Negocio justo — negocia el precio"
        elif margen_pct >= 0:
            score, rec = "BAJO", "Margen muy bajo — solo si rotas rápido"
        else:
            score, rec = "MALO", "No es negocio — precio sobre el mercado"

        score_result = {
            "score": score,
            "recomendacion": rec,
            "margen_bruto": round(margen_bruto, 2),
            "margen_pct": round(margen_pct, 2),
        }

    return {
        "identificacion": {
            "marca": marca,
            "modelo": modelo,
            "anio_estimado": anio,
            "carroceria": carroceria,
            "color": color,
            "confianza": id_result["confianza"],
            "notas": id_result.get("notas"),
        },
        "precio_compra": precio_compra,
        "precio_estimado": round(precio_estimado, 2) if precio_estimado else None,
        "score": score_result,
        "specs_usadas": specs or None,
    }
