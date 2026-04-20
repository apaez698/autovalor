"""Score route — business opportunity scoring."""

import os
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException

from app.schemas import ScoreInput, VehicleInput
from app.services.model_service import model, vehicle_catalog, predict
from app.services.supabase import supabase_headers, supabase_url

router = APIRouter()


@router.post("/api/score")
async def score_negocio(data: ScoreInput):
    """Calcula score de oportunidad de negocio combinando predicción ML + mercado."""
    # Precio ML
    precio_ml = None
    if model is not None:
        marca_upper = data.marca.strip().upper()
        modelo_upper = data.modelo.strip().upper()
        entry = next(
            (e for e in vehicle_catalog
             if e["marca"] == marca_upper and e["modelo"] == modelo_upper),
            None,
        )
        if entry and entry.get("specs"):
            specs = entry["specs"]
            try:
                vehicle = VehicleInput(
                    marca=marca_upper,
                    modelo=modelo_upper,
                    anio=data.anio,
                    kilometraje=data.km or 50000,
                    motor_cc=specs.get("motor_cc", 1600),
                    potencia_hp=specs.get("potencia_hp", 120),
                    carroceria=specs.get("carroceria", "SUV"),
                    transmision=specs.get("transmision", "MANUAL"),
                    tipo_combustible=specs.get("tipo_combustible", "GASOLINA"),
                    provincia="PICHINCHA",
                    traccion=specs.get("traccion", "4X2"),
                    segmento=specs.get("segmento", "MEDIO"),
                    pais_origen=specs.get("pais_origen", "DESCONOCIDO"),
                    color="BLANCO",
                )
                result = predict(vehicle)
                precio_ml = result["precio_estimado"]
            except Exception:
                pass

    # Precio mercado (Supabase)
    precio_mercado = None
    if os.getenv("SUPABASE_KEY") and os.getenv("SUPABASE_URL"):
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(
                    supabase_url("precios_mercado"),
                    headers=supabase_headers(),
                    params={
                        "marca": f"eq.{data.marca.lower()}",
                        "modelo_nombre": f"ilike.%{data.modelo}%",
                        "año": f"eq.{data.anio}",
                        "order": "fecha_scrape.desc",
                        "limit": "1",
                    },
                    timeout=10,
                )
                if r.status_code == 200 and r.json():
                    precio_mercado = r.json()[0].get("precio_ideal")
        except Exception:
            pass

    # Determine reference price
    precio_ref = None
    fuente_precio = None
    if precio_mercado and precio_ml:
        precio_ref = (precio_mercado * 0.6 + precio_ml * 0.4)
        fuente_precio = "mercado+ml"
    elif precio_mercado:
        precio_ref = precio_mercado
        fuente_precio = "mercado"
    elif precio_ml:
        precio_ref = precio_ml
        fuente_precio = "ml"

    if precio_ref is None:
        raise HTTPException(
            status_code=404,
            detail=f"Sin datos de precio para {data.marca} {data.modelo} {data.anio}",
        )

    # Calculate score
    precio_venta_est = precio_ref * 0.88
    margen_bruto = precio_venta_est - data.precio_compra
    margen_pct = (margen_bruto / data.precio_compra) * 100 if data.precio_compra > 0 else 0

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

    return {
        "marca": data.marca,
        "modelo": data.modelo,
        "anio": data.anio,
        "precio_compra": data.precio_compra,
        "precio_referencia": round(precio_ref, 2),
        "fuente_precio": fuente_precio,
        "precio_venta_estimado": round(precio_venta_est, 2),
        "margen_bruto": round(margen_bruto, 2),
        "margen_pct": round(margen_pct, 2),
        "score": score,
        "recomendacion": rec,
        "precio_ml": round(precio_ml, 2) if precio_ml else None,
        "precio_mercado": round(precio_mercado, 2) if precio_mercado else None,
    }
