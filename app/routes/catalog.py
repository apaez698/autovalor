"""Catalog routes — marcas, modelos, specs, search."""

from fastapi import APIRouter

from app.services import catalog_service

router = APIRouter(prefix="/api/catalog")


@router.get("/marcas")
def marcas():
    return catalog_service.list_marcas()


@router.get("/modelos/{marca}")
def modelos(marca: str):
    return catalog_service.list_modelos(marca)


@router.get("/specs/{marca}/{modelo}")
def specs(marca: str, modelo: str):
    return catalog_service.get_specs(marca, modelo)


@router.get("/search")
def search(q: str):
    return catalog_service.search_catalog(q)
