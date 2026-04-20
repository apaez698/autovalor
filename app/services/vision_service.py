"""Vision service — Claude Vision vehicle identification."""

import base64
import json
import os

import anthropic
from fastapi import HTTPException, UploadFile

from app.services.model_service import vehicle_catalog

VISION_SYSTEM_PROMPT = """Eres un experto en identificación de vehículos del mercado ecuatoriano.
Analiza la foto del vehículo y devuelve SOLO un JSON válido (sin markdown, sin explicaciones) con:
{
  "marca": "string — marca del vehículo",
  "modelo": "string — modelo específico",
  "anio_estimado": int o null,
  "carroceria": "sedán|SUV|hatchback|pickup|van|coupé|convertible",
  "color": "string",
  "confianza": "alta|media|baja",
  "notas": "string — observaciones relevantes sobre el estado visible"
}
Usa nombres de marca y modelo tal como se usan en Ecuador (ej: CHEVROLET, TOYOTA, KIA).
Si no puedes identificar el vehículo, pon confianza "baja" y explica en notas."""

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB


async def identify_vehicle(foto: UploadFile, incluir_specs: bool = True) -> dict:
    """Identify a vehicle from a photo using Claude Vision."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY no configurada")

    if foto.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de imagen no soportado: {foto.content_type}. Usa JPEG, PNG, GIF o WebP.",
        )

    image_bytes = await foto.read()
    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=400, detail="Imagen demasiado grande (max 10MB)")

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        system=VISION_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": foto.content_type,
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": "Identifica este vehículo."},
                ],
            }
        ],
    )

    raw_text = response.content[0].text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=502,
            detail=f"Respuesta de Claude no es JSON válido: {raw_text[:200]}",
        )

    # Enrich with catalog specs
    specs_sugeridas = None
    if incluir_specs and result.get("marca") and result.get("modelo"):
        marca_upper = result["marca"].strip().upper()
        modelo_upper = result["modelo"].strip().upper()
        entry = next(
            (e for e in vehicle_catalog
             if e["marca"] == marca_upper and e["modelo"] == modelo_upper),
            None,
        )
        if not entry:
            entry = next(
                (e for e in vehicle_catalog
                 if e["marca"] == marca_upper and modelo_upper in e["modelo"]),
                None,
            )
        if entry:
            specs_sugeridas = entry["specs"]

    return {
        "marca": result.get("marca", "DESCONOCIDO"),
        "modelo": result.get("modelo", "DESCONOCIDO"),
        "anio_estimado": result.get("anio_estimado"),
        "carroceria": result.get("carroceria"),
        "color": result.get("color"),
        "confianza": result.get("confianza", "baja"),
        "specs_sugeridas": specs_sugeridas,
        "notas": result.get("notas"),
    }
