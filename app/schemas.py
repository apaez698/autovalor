"""Pydantic schemas shared across routes."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class VehicleInput(BaseModel):
    marca: str
    modelo: str
    anio: int = Field(..., ge=1950, le=datetime.now().year + 1)
    kilometraje: int = Field(..., ge=0, le=1_000_000)
    motor_cc: float = Field(..., ge=600, le=8000)
    potencia_hp: float = Field(..., ge=30, le=800)
    carroceria: str
    transmision: str
    tipo_combustible: str
    provincia: str
    traccion: str
    segmento: str
    pais_origen: str
    color: str


class ScoreInput(BaseModel):
    marca: str
    modelo: str
    anio: int
    precio_compra: float
    km: Optional[int] = None


class PhotoIDResponse(BaseModel):
    marca: str
    modelo: str
    anio_estimado: Optional[int] = None
    carroceria: Optional[str] = None
    color: Optional[str] = None
    confianza: str
    specs_sugeridas: Optional[dict] = None
    notas: Optional[str] = None


class PrecioMercado(BaseModel):
    marca: str
    modelo_slug: str
    modelo_nombre: str
    año: int
    precio_ideal: float
    precio_min: Optional[float] = None
    precio_max: Optional[float] = None
    url: str = ""
    enlace_listado: str = ""
    fecha_scrape: str = ""
    fuente: str = "patiotuerca"


class PrecioMercadoBatch(BaseModel):
    registros: list[PrecioMercado]


class ScraperRunRequest(BaseModel):
    scraper: str = "patiotuerca"
    años: list[int] = list(range(2018, 2027))
    marcas: Optional[list[str]] = None
    max_modelos: int = 999


class RetrainRequest(BaseModel):
    scraped_csv: Optional[str] = None
    use_supabase: bool = True
