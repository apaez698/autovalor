"""Metadata and info routes."""

from fastapi import APIRouter

from app.services.model_service import model_metadata
from ml.feature_config import FEATURE_ORDER, VALIDATION_LIMITS

router = APIRouter()


@router.get("/api/metadata")
def metadata():
    categorical_values = model_metadata.get("categorical_values", {})
    return {
        "feature_order": model_metadata.get("feature_order", FEATURE_ORDER),
        "categorical_values": categorical_values,
        "feature_importances": model_metadata.get("feature_importances", {}),
        "metrics": model_metadata.get("metrics", {}),
        "validation_limits": VALIDATION_LIMITS,
    }


@router.get("/metadata")
def metadata_legacy():
    return metadata()


@router.get("/feature-importance")
def feature_importance_legacy():
    importance = model_metadata.get("feature_importances", {})
    sorted_importance = [
        {"feature": f, "importance": v}
        for f, v in sorted(importance.items(), key=lambda x: x[1], reverse=True)
    ]
    return {"feature_importance": sorted_importance}


@router.get("/info")
def info_legacy():
    return {
        "name": "AutoValor API",
        "version": "3.0.0",
        "endpoints": [
            "/health", "/metadata", "/predict", "/feature-importance",
            "/api/predict", "/api/metadata", "/api/catalog/marcas",
            "/api/catalog/modelos/{marca}", "/api/catalog/specs/{marca}/{modelo}",
            "/api/catalog/search", "/api/score", "/api/identify", "/api/evaluate",
            "/api/scraper/precios", "/api/scraper/precios/batch",
            "/api/precios/{marca}/{modelo}/{anio}",
        ],
    }
