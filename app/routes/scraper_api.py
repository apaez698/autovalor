"""Scraper ingestion and retrain routes (Supabase)."""

import os
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Header

from app.schemas import PrecioMercado, PrecioMercadoBatch
from app.services.model_service import load_resources
from app.services.supabase import (
    supabase_headers, supabase_url, supabase_upsert_headers, supabase_upsert_url,
)

router = APIRouter()


def verificar_api_key(authorization: Optional[str] = Header(None)):
    api_key = os.getenv("API_KEY")
    if not api_key:
        return True
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="API Key requerida")
    if authorization.replace("Bearer ", "") != api_key:
        raise HTTPException(status_code=403, detail="API Key inválida")
    return True


@router.post("/api/scraper/precios", status_code=201)
async def recibir_precio(
    precio: PrecioMercado,
    _: bool = Depends(verificar_api_key),
):
    data = precio.model_dump()
    async with httpx.AsyncClient() as client:
        r = await client.post(
            supabase_upsert_url("precios_mercado"),
            headers=supabase_upsert_headers(),
            json=data,
            timeout=10,
        )
    if r.status_code not in (200, 201):
        raise HTTPException(status_code=500, detail=f"Supabase error: {r.text[:200]}")
    result = r.json()
    inserted = result[0] if isinstance(result, list) and result else data
    return {
        "ok": True,
        "id": inserted.get("id"),
        "marca": precio.marca,
        "modelo": precio.modelo_nombre,
    }


@router.post("/api/scraper/precios/batch", status_code=201)
async def recibir_precios_batch(
    batch: PrecioMercadoBatch,
    _: bool = Depends(verificar_api_key),
):
    if len(batch.registros) > 500:
        raise HTTPException(status_code=400, detail="Máximo 500 registros por batch")

    data = [r.model_dump() for r in batch.registros]
    async with httpx.AsyncClient() as client:
        r = await client.post(
            supabase_upsert_url("precios_mercado"),
            headers=supabase_upsert_headers(),
            json=data,
            timeout=30,
        )
    if r.status_code not in (200, 201):
        raise HTTPException(status_code=500, detail=f"Supabase error: {r.text[:300]}")
    result = r.json()
    insertados = len(result) if isinstance(result, list) else len(data)
    return {"ok": True, "insertados": insertados, "enviados": len(data)}


@router.get("/api/precios/{marca}/{modelo}/{anio}")
async def consultar_precio(marca: str, modelo: str, anio: int):
    async with httpx.AsyncClient() as client:
        r = await client.get(
            supabase_url("precios_mercado"),
            headers=supabase_headers(),
            params={
                "marca": f"eq.{marca}",
                "modelo_slug": f"eq.{modelo}",
                "año": f"eq.{anio}",
                "order": "fecha_scrape.desc",
                "limit": "1",
            },
            timeout=10,
        )
    if r.status_code != 200 or not r.json():
        raise HTTPException(status_code=404, detail="Precio no encontrado")
    return r.json()[0]


@router.post("/api/retrain")
async def retrain_model_endpoint(
    scraped_csv: Optional[str] = None,
    use_supabase: bool = True,
    _: bool = Depends(verificar_api_key),
):
    from ml.retrain import retrain
    from fastapi import BackgroundTasks

    def _do_retrain():
        retrain(scraped_csv=scraped_csv or "", use_supabase=use_supabase)
        load_resources()

    import asyncio
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _do_retrain)

    return {
        "ok": True,
        "message": "Retraining iniciado en background. El modelo se recargará al terminar.",
    }
