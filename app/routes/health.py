"""Health check route."""

import os
import datetime

from fastapi import APIRouter

from app.services.model_service import model, label_encoders, vehicle_catalog

router = APIRouter()


@router.get("/health")
def health():
    return {
        "status": "ok",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_loaded": model is not None,
        "encoders_loaded": sorted(label_encoders.keys()),
        "catalog_entries": len(vehicle_catalog),
        "supabase_configured": bool(os.getenv("SUPABASE_KEY")),
        "anthropic_configured": bool(os.getenv("ANTHROPIC_API_KEY")),
    }
