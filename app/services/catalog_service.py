"""Catalog service — search, list marcas/modelos, get specs."""

from fastapi import HTTPException

from app.services.model_service import vehicle_catalog


def list_marcas() -> dict:
    marcas = sorted(set(e["marca"] for e in vehicle_catalog))
    return {"marcas": marcas, "total": len(marcas)}


def list_modelos(marca: str) -> dict:
    marca_upper = marca.strip().upper()
    modelos = [
        {
            "modelo": e["modelo"],
            "versiones": e["versiones"],
            "año_min": e["año_min"],
            "año_max": e["año_max"],
            "carroceria": e["specs"].get("carroceria", ""),
            "count": e["count"],
        }
        for e in vehicle_catalog
        if e["marca"] == marca_upper
    ]
    if not modelos:
        raise HTTPException(status_code=404, detail=f"Marca '{marca}' no encontrada")
    modelos.sort(key=lambda x: (-x["count"], x["modelo"]))
    return {"marca": marca_upper, "modelos": modelos, "total": len(modelos)}


def get_specs(marca: str, modelo: str) -> dict:
    marca_upper = marca.strip().upper()
    modelo_upper = modelo.strip().upper()

    entry = next(
        (e for e in vehicle_catalog
         if e["marca"] == marca_upper and e["modelo"] == modelo_upper),
        None,
    )
    if not entry:
        raise HTTPException(
            status_code=404,
            detail=f"No se encontró {marca}/{modelo} en el catálogo",
        )
    return {
        "marca": entry["marca"],
        "modelo": entry["modelo"],
        "versiones": entry["versiones"],
        "año_min": entry["año_min"],
        "año_max": entry["año_max"],
        "specs": entry["specs"],
    }


def search_catalog(q: str) -> dict:
    query = q.strip().upper()
    if len(query) < 2:
        raise HTTPException(status_code=400, detail="Query debe tener al menos 2 caracteres")

    results = [
        {
            "marca": e["marca"],
            "modelo": e["modelo"],
            "carroceria": e["specs"].get("carroceria", ""),
            "año_min": e["año_min"],
            "año_max": e["año_max"],
        }
        for e in vehicle_catalog
        if query in e["marca"] or query in e["modelo"]
    ]
    return {"query": query, "results": results[:20], "total": len(results)}
