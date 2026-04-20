"""
Retraining pipeline — Combina data histórica con precios de mercado scrapeados.

Los precios de Patiotuerca son precios de VENTA en el mercado.
Los datos de training son precios de COMPRA (avalúo). La relación es:
   precio_compra ≈ precio_mercado × factor_descuento

Este script:
1. Carga la data histórica de training (data_limpia_entrenamiento.csv)
2. Carga precios scrapeados (CSV o Supabase)
3. Normaliza nombres de marca/modelo para que coincidan con los del training set
4. Enriquece los precios scrapeados con specs del catálogo
5. Aplica factor de descuento (compra ≈ 88% del precio mercado)
6. Combina ambas fuentes
7. Reentrena el modelo XGBoost
8. Exporta modelo y metadata actualizados
"""

import json
import os
import re
from pathlib import Path

import httpx
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
CATALOG_PATH = MODELS_DIR / "vehicle_catalog.json"

# Factor de descuento: precio_compra ≈ precio_mercado × FACTOR
COMPRA_FACTOR = 0.88

# ─── Normalization: scraped model names → training model names ───────────────

# Prefixes used by Patiotuerca that should be stripped
_STRIP_PREFIXES = ["AWT", "BRT", "ASA", "NKR", "SRT", "SVM", "UNI"]

# Suffixes to strip (trim levels, editions)
_STRIP_SUFFIXES = [
    "HIGH", "MID", "LOW", "ACTIVE", "HIBRID", "HIBRIDO",
    "4X4", "4X2", "CD", "AC", "TM", "AT", "MT", "CVT",
    "SEG", "XEI", "GLX", "GL", "GLS", "GT", "SR", "SV",
    "ADVANCE", "PREMIUM", "LIMITED", "SPORT", "BASE",
    "FULL", "STD", "DLX", "COMFORT", "STYLE",
]

# Explicit mapping for known mismatches: scraped_upper → training name
_MODEL_ALIASES = {
    # Toyota
    "COROLLA CROSS HIGH": "COROLLA CROSS",
    "COROLLA CROSS MID": "COROLLA CROSS",
    "COROLLA CROSS LOW": "COROLLA CROSS",
    "COROLLA 1.8": "COROLLA",
    "COROLLA HV HYBRID 1.8": "COROLLA",
    "COROLLA HV XEI": "COROLLA",
    "COROLLA HV SEG AC": "COROLLA",
    "C HR": "C-HR",
    "C HR ACTIVE HIBRID": "C-HR",
    "CAMRY HIBRIDO": "CAMRY",
    "HILUX 4X4 CD AC": "HILUX",
    "AWT FORTUNER": "FORTUNER",
    "BRT HILUX 4X4 CD AC": "HILUX",
    "AVEO FAMILY": "AVEO",
    # Kia
    "CARNIVAL": "CARNIVAL R",
    "RIO": "RIO",
    "BESTA": "BESTA",
    "BRISA": "BRISA",
    # Hyundai
    "GRAND I10": "GRAND I-10",
    "I10": "I-10",
    # Nissan
    "X TRAIL": "X TRAIL",
    # Chevrolet
    "D MAX": "D-MAX",
    "DMAX": "D-MAX",
    "TRAIL BLAZER": "TRAIL BLAZER",
    "GRAND VITARA SZ": "GRAND VITARA",
}

# Brand normalization: scraped → training
_BRAND_ALIASES = {
    "toyota": "TOYOTA",
    "chevrolet": "CHEVROLET",
    "kia": "KIA",
    "hyundai": "HYUNDAI",
    "nissan": "NISSAN",
    "mazda": "MAZDA",
    "suzuki": "SUZUKI",
    "ford": "FORD",
    "volkswagen": "VOLKSWAGEN",
    "honda": "HONDA",
    "peugeot": "PEUGEOT",
    "renault": "RENAULT",
    "mitsubishi": "MITSUBISHI",
    "chery": "CHERY",
    "byd": "BYD",
    "mg": "MG MOTOR",
    "great+wall": "GREAT WALL",
    "jac": "JAC MOTORS",
    "jeep": "JEEP",
    "isuzu": "ISUZU",
    "changan": "CHANGAN",
    "haval": "HAVAL",
    "jetour": "JETOUR",
    "fiat": "FIAT",
    "ram": "RAM",
    "mini": "MINI",
    "audi": "AUDI",
    "bmw": "BMW",
    "mercedes+benz": "MERCEDES BENZ",
    "land+rover": "LAND ROVER",
    "lexus": "LEXUS",
    "subaru": "SUBARU",
    "volvo": "VOLVO",
    "geely": "GEELY",
    "dongfeng": "DONGFENG",
    "dfsk": "D.F.S.K. (DONGFENG)",
    "foton": "FOTON",
    "jmc": "JMC",
}


def _build_training_model_index(df_hist: pd.DataFrame) -> dict[str, set[str]]:
    """Build marca → set(modelos) from historical training data."""
    index: dict[str, set[str]] = {}
    for _, row in df_hist[["Marca", "Modelo"]].drop_duplicates().iterrows():
        marca = str(row["Marca"]).strip().upper()
        modelo = str(row["Modelo"]).strip().upper()
        index.setdefault(marca, set()).add(modelo)
    return index


def normalize_marca(marca_raw: str) -> str:
    """Normalize a scraped brand name to training format."""
    key = marca_raw.strip().lower().replace(" ", "+")
    if key in _BRAND_ALIASES:
        return _BRAND_ALIASES[key]
    return marca_raw.strip().upper()


def normalize_modelo(modelo_raw: str, marca_norm: str, model_index: dict[str, set[str]]) -> str:
    """Normalize a scraped model name to match training data model names.
    
    Strategy:
    1. Check explicit alias map
    2. Try stripping prefixes/suffixes to match training models
    3. Try fuzzy containment match against training models
    4. Return cleaned name as-is if no match found
    """
    nombre = modelo_raw.strip().upper()
    nombre = re.sub(r"\s+", " ", nombre)  # collapse whitespace

    # 1. Exact alias
    if nombre in _MODEL_ALIASES:
        return _MODEL_ALIASES[nombre]

    # Get known models for this brand
    known = model_index.get(marca_norm, set())

    # 2. Direct match
    if nombre in known:
        return nombre

    # 3. Strip prefix (e.g., "AWT FORTUNER" → "FORTUNER")
    parts = nombre.split()
    if len(parts) > 1 and parts[0] in _STRIP_PREFIXES:
        stripped = " ".join(parts[1:])
        if stripped in _MODEL_ALIASES:
            return _MODEL_ALIASES[stripped]
        if stripped in known:
            return stripped

    # 4. Progressive suffix stripping (e.g., "HILUX 4X4 CD AC" → "HILUX")
    candidate = nombre
    # First, strip known prefixes
    cparts = candidate.split()
    if len(cparts) > 1 and cparts[0] in _STRIP_PREFIXES:
        cparts = cparts[1:]
    # Then strip suffixes one by one from the right
    while len(cparts) > 1:
        if cparts[-1] in _STRIP_SUFFIXES or re.match(r"^\d+(\.\d+)?$", cparts[-1]):
            cparts = cparts[:-1]
            trial = " ".join(cparts)
            if trial in _MODEL_ALIASES:
                return _MODEL_ALIASES[trial]
            if trial in known:
                return trial
        else:
            break

    # 5. Containment: find training model contained in or containing scraped name
    cleaned = " ".join(cparts)
    best_match = None
    best_len = 0
    for km in known:
        if km in cleaned or cleaned in km:
            if len(km) > best_len:
                best_match = km
                best_len = len(km)
    if best_match:
        return best_match

    # 6. Hyphen tolerance: "C HR" ↔ "C-HR", "D MAX" ↔ "D-MAX"
    cleaned_hyphen = cleaned.replace(" ", "-")
    for km in known:
        if km == cleaned_hyphen:
            return km

    return cleaned


def load_catalog() -> dict[tuple[str, str], dict]:
    """Load vehicle catalog as a (marca, modelo) → specs lookup."""
    if not CATALOG_PATH.exists():
        return {}
    with CATALOG_PATH.open("r", encoding="utf-8") as f:
        catalog = json.load(f)
    return {
        (e["marca"], e["modelo"]): e["specs"]
        for e in catalog
    }


def load_scraped_from_csv(csv_path: str) -> pd.DataFrame:
    """Load scraped market prices from a CSV file."""
    if not os.path.exists(csv_path):
        print(f"CSV no encontrado: {csv_path}")
        return pd.DataFrame()
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    # Handle encoding issue with año column
    año_col = next((c for c in df.columns if "a" in c.lower() and "o" in c.lower() and c not in ["marca", "modelo_nombre", "modelo_slug"]), None)
    if año_col and año_col != "año":
        df = df.rename(columns={año_col: "año"})
    print(f"Cargados {len(df)} registros de {csv_path}")
    return df


def load_scraped_from_supabase() -> pd.DataFrame:
    """Load all market prices from Supabase."""
    url = os.getenv("SUPABASE_URL", "").rstrip("/")
    key = os.getenv("SUPABASE_KEY", "")
    if not url or not key:
        print("Supabase no configurado — skip")
        return pd.DataFrame()

    headers = {"apikey": key, "Authorization": f"Bearer {key}"}
    all_rows = []
    offset = 0
    limit = 1000

    while True:
        r = httpx.get(
            f"{url}/rest/v1/precios_mercado",
            headers=headers,
            params={
                "select": "marca,modelo_slug,modelo_nombre,año,precio_ideal,precio_min,precio_max,fecha_scrape,fuente",
                "order": "id.asc",
                "offset": str(offset),
                "limit": str(limit),
            },
            timeout=30,
        )
        if r.status_code != 200:
            print(f"Error Supabase: {r.status_code}")
            break
        rows = r.json()
        if not rows:
            break
        all_rows.extend(rows)
        offset += limit

    if all_rows:
        df = pd.DataFrame(all_rows)
        print(f"Cargados {len(df)} registros desde Supabase")
        return df
    return pd.DataFrame()


def enrich_scraped_data(df_scraped: pd.DataFrame, model_index: dict[str, set[str]]) -> pd.DataFrame:
    """Enrich scraped market prices with normalized names + catalog specs."""
    if df_scraped.empty:
        return df_scraped

    catalog = load_catalog()

    records = []
    matched = 0
    unmatched_models = set()

    for _, row in df_scraped.iterrows():
        marca_raw = str(row.get("marca", "")).strip()
        modelo_nombre = str(row.get("modelo_nombre", "")).strip()

        # Normalize brand and model
        marca_norm = normalize_marca(marca_raw)
        modelo_norm = normalize_modelo(modelo_nombre, marca_norm, model_index)

        # Lookup specs from catalog
        specs = catalog.get((marca_norm, modelo_norm), {})

        # If no exact match, try partial
        if not specs:
            for (cat_marca, cat_modelo), cat_specs in catalog.items():
                if cat_marca == marca_norm and (
                    modelo_norm in cat_modelo or cat_modelo in modelo_norm
                ):
                    specs = cat_specs
                    break

        año = int(row.get("año", 2023))
        precio_mercado = float(row.get("precio_ideal", 0))
        if precio_mercado <= 0:
            continue

        # Track matching
        if modelo_norm in model_index.get(marca_norm, set()):
            matched += 1
        else:
            unmatched_models.add(f"{marca_norm}/{modelo_norm}")

        # Estimated purchase price (what a dealer would pay)
        precio_compra = precio_mercado * COMPRA_FACTOR

        record = {
            "Marca": marca_norm,
            "Modelo": modelo_norm,
            "Año": año,
            "Recorrido": 50000.0,  # Estimated avg km for market price
            "Precio Final Editado": precio_compra,
            "Color": "BLANCO",
            "provincia": "PICHINCHA",
            "carroceria": specs.get("carroceria", "DESCONOCIDO"),
            "motor_cc": specs.get("motor_cc"),
            "potencia_hp": specs.get("potencia_hp"),
            "transmision": specs.get("transmision", "DESCONOCIDO"),
            "traccion": specs.get("traccion", "DESCONOCIDO"),
            "tipo_combustible": specs.get("tipo_combustible", "DESCONOCIDO"),
            "segmento": specs.get("segmento", "DESCONOCIDO"),
            "pais_origen": specs.get("pais_origen", "DESCONOCIDO"),
            "_fuente": "scraper_mercado",
            "_modelo_original": modelo_nombre,  # Keep original for debugging
        }
        records.append(record)

    df_enriched = pd.DataFrame(records)
    print(f"Registros enriquecidos: {len(df_enriched)} (de {len(df_scraped)} scrapeados)")
    print(f"  Coincidencias con training set: {matched}")
    if unmatched_models:
        print(f"  Sin match en training ({len(unmatched_models)}): {sorted(unmatched_models)[:15]}")
    return df_enriched


def retrain(
    historical_csv: str = "",
    scraped_csv: str = "",
    use_supabase: bool = True,
):
    """Combine historical + scraped data and retrain the model."""
    from ml.train import (
        load_and_clean_data,
        prepare_features,
        train_model,
        export_metadata,
    )

    frames = []

    # 1. Historical data
    hist_path = historical_csv or str(DATA_DIR / "data_limpia_entrenamiento.csv")
    if os.path.exists(hist_path):
        df_hist = load_and_clean_data(hist_path)
        df_hist["_fuente"] = "historico"
        frames.append(df_hist)
        print(f"\n📚 Data histórica: {len(df_hist)} registros")
    else:
        df_hist = pd.DataFrame()

    # Build model index from historical data for normalization
    model_index = _build_training_model_index(df_hist) if not df_hist.empty else {}
    if model_index:
        total_models = sum(len(v) for v in model_index.values())
        print(f"📋 Índice de normalización: {len(model_index)} marcas, {total_models} modelos")

    # 2. Scraped data from CSV
    if scraped_csv and os.path.exists(scraped_csv):
        df_csv = load_scraped_from_csv(scraped_csv)
        df_enriched_csv = enrich_scraped_data(df_csv, model_index)
        if not df_enriched_csv.empty:
            frames.append(df_enriched_csv)
            print(f"🌐 Data scrapeada (CSV): {len(df_enriched_csv)} registros")

    # 3. Scraped data from Supabase
    if use_supabase:
        df_supa = load_scraped_from_supabase()
        df_enriched_supa = enrich_scraped_data(df_supa, model_index)
        if not df_enriched_supa.empty:
            # Avoid duplicating CSV and Supabase data 
            if scraped_csv:
                # Supabase should have same data, just use one
                print(f"⚠️  Supabase data ({len(df_enriched_supa)}) omitida — ya tenemos CSV")
            else:
                frames.append(df_enriched_supa)
                print(f"🌐 Data scrapeada (Supabase): {len(df_enriched_supa)} registros")

    if not frames:
        print("❌ Sin datos para entrenar")
        return

    df_combined = pd.concat(frames, ignore_index=True)
    print(f"\n📊 Dataset combinado: {len(df_combined)} registros")

    # Show distribution by source
    if "_fuente" in df_combined.columns:
        print("\nRegistros por fuente:")
        print(df_combined["_fuente"].value_counts().to_string())

    # 4. Prepare features and train
    X, y = prepare_features(df_combined)
    print(f"\nFeature matrix: {X.shape}")
    model, metrics = train_model(X, y)

    # 5. Export metadata
    export_metadata(df_combined, model, X.columns.tolist())

    print("\n" + "=" * 60)
    print("✅ Modelo reentrenado exitosamente")
    print(f"   MAE:  ${metrics['mae']:,.0f}")
    print(f"   R²:   {metrics['r2']:.4f}")
    print(f"   MAPE: {metrics['mape']:.2f}%")
    print(f"   Registros totales: {len(df_combined)}")
    print("=" * 60)

    return model, metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Retrain con datos históricos + scrapeados")
    parser.add_argument(
        "--historical",
        default=str(DATA_DIR / "data_limpia_entrenamiento.csv"),
        help="CSV con data histórica de avalúos",
    )
    parser.add_argument(
        "--scraped",
        default="",
        help="CSV con precios scrapeados del mercado",
    )
    parser.add_argument(
        "--no-supabase",
        action="store_true",
        help="No cargar datos desde Supabase",
    )
    args = parser.parse_args()

    retrain(
        historical_csv=args.historical,
        scraped_csv=args.scraped,
        use_supabase=not args.no_supabase,
    )
