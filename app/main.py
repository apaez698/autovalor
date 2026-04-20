"""
AutoValor — Unified FastAPI application.
Integrates: XGBoost prediction, vehicle catalog, business scoring, photo ID.
"""

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI(
    title="AutoValor API",
    version="3.0.0",
    description="Sistema de IA para compra/reventa de vehículos en Ecuador",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ML resources on startup
from app.services.model_service import load_resources  # noqa: E402
load_resources()

# Register routers
from app.routes import predict, catalog, score, identify, health, metadata, scraper_api  # noqa: E402

app.include_router(predict.router)
app.include_router(catalog.router)
app.include_router(score.router)
app.include_router(identify.router)
app.include_router(health.router)
app.include_router(metadata.router)
app.include_router(scraper_api.router)
