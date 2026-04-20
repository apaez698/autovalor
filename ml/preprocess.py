"""
Parser de Descripciones de Vehículos
Usa Claude API para extraer campos estructurados desde el campo Descripción
y genera un catálogo limpio listo para entrenar el modelo de precio.

Uso:
    pip install anthropic pandas tqdm
    export ANTHROPIC_API_KEY=sk-ant-...
    python parser_vehiculos.py --input data.csv --output catalogo_limpio.csv
"""

import os
import json
import time
import argparse
import pandas as pd
from tqdm import tqdm
import anthropic

# ── Configuración ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Eres un parser especializado en fichas técnicas de vehículos del mercado ecuatoriano.
Tu única tarea es extraer datos estructurados desde descripciones de avalúos.
Responde ÚNICAMENTE con un objeto JSON válido, sin texto adicional, sin markdown, sin explicaciones."""

USER_PROMPT_TEMPLATE = """Extrae los campos del siguiente vehículo y devuelve solo JSON.

Datos del registro:
- Marca: {marca}
- Modelo: {modelo}
- Año: {anio}
- Color: {color}
- Recorrido: {recorrido} {tipo_recorrido}
- Descripción técnica: {descripcion}

Devuelve exactamente este JSON (usa null si no puedes determinar el valor):
{{
  "version": null,
  "carroceria": null,
  "motor_cc": null,
  "motor_label": null,
  "cilindros": null,
  "potencia_hp": null,
  "torque_nm": null,
  "transmision": null,
  "velocidades": null,
  "traccion": null,
  "puertas": null,
  "aire_acondicionado": null,
  "aire_bizona": null,
  "pantalla_pulgadas": null,
  "apple_carplay": null,
  "android_auto": null,
  "camaras": null,
  "airbags": null,
  "aros_pulgadas": null,
  "luces_led": null,
  "norma_euro": null,
  "tipo_combustible": null,
  "es_hibrido": null,
  "pais_origen": null,
  "segmento": null
}}

Reglas:
- carroceria: "sedán", "SUV", "pickup", "hatchback", "van", "coupé", "convertible"
- transmision: "manual" o "automática"
- traccion: "4x2", "4x4", "AWD"
- tipo_combustible: "gasolina", "diesel", "híbrido", "eléctrico"
- segmento: "económico", "medio", "medio-alto", "alto", "lujo"
- es_hibrido: true/false
- apple_carplay, android_auto, luces_led, aire_acondicionado, aire_bizona, camaras: true/false/null
- Para números usa solo el valor numérico (sin unidades)
- pais_origen: deduce si puedes (BRASIL, JAPÓN, COREA, etc.)"""


# ── Mapeo de Agencias → Ciudad / Provincia ────────────────────────────────────
# Completa las agencias que conozcas con: {"NOMBRE": ("Ciudad", "Provincia")}
# Las marcadas con None quedarán como "Desconocido" hasta que las definas tú.

AGENCIA_UBICACION = {
    # ── Quito (Pichincha) ──────────────────────────────────────────────────────
    "CUMBAYA":                  ("Quito",         "Pichincha"),
    "LOS CHILLOS":              ("Quito",         "Pichincha"),
    "CONDADO":                  ("Quito",         "Pichincha"),
    "SZK LABRADOR":             ("Quito",         "Pichincha"),
    "SZK CHILLOS":              ("Quito",         "Pichincha"),
    "SZK CUMBAYA":              ("Quito",         "Pichincha"),
    "SZK GRANADOS":             ("Quito",         "Pichincha"),
    "SZK JTM":                  ("Quito",         "Pichincha"),
    "SZK":                      ("Quito",         "Pichincha"),
    "PEUGEOT LION":             ("Quito",         "Pichincha"),
    "AMAZONAS":                 ("Quito",         "Pichincha"),
    "BICENTENARIO":             ("Quito",         "Pichincha"),
    "AUTOCONSA ORELLANA":       ("Quito",         "Pichincha"),
    "MARESA JTM":               ("Quito",         "Pichincha"),
    "PEUGEOT VILLAGE PLAZA":    ("Quito",         "Pichincha"),

    # ── Quito confirmados ──────────────────────────────────────────────────────
    "GRANADOS":                 ("Quito",         "Pichincha"),   # Av. Granados
    "ORELLANA":                 ("Quito",         "Pichincha"),   # Av. Orellana
    "SUR":                      ("Quito",         "Pichincha"),   # Sector Quitumbe/Guamaní
    "CJA. AROSEMENA":           ("Quito",         "Pichincha"),   # Av. Carlos Julio Arosemena
    "CARRION":                  ("Quito",         "Pichincha"),   # Zona Mariscal/centro-norte
    "MATRIZ":                   ("Quito",         "Pichincha"),   # Oficina principal
    "LA MARTINA":               ("Quito",         "Pichincha"),   # Sector comercial
    "JARDIN":                   ("Quito",         "Pichincha"),   # Zona El Jardín
    "EL DORADO":                ("Quito",         "Pichincha"),   # Sector El Dorado
    "CHANGAN EL DORADO":        ("Quito",         "Pichincha"),
    "CHANGAN EL JARDIN":        ("Quito",         "Pichincha"),
    "OPEL EL DORADO":           ("Quito",         "Pichincha"),
    "PEUGEOT":                  ("Quito",         "Pichincha"),

    # ── Guayaquil (Guayas) ────────────────────────────────────────────────────
    "LA PIAZZA SAMBORONDÓN":    ("Guayaquil",     "Guayas"),
    "GBV DAULE":                ("Daule",         "Guayas"),
    "DAULE":                    ("Daule",         "Guayas"),
    "URDENOR":                  ("Guayaquil",     "Guayas"),
    "GBV ALCIVAR":              ("Guayaquil",     "Guayas"),
    "GUAYAQUIL":                ("Guayaquil",     "Guayas"),
    "SZK SUR":                  ("Guayaquil",     "Guayas"),
    "PEUGEOT JUAN TANCA MARENGO": ("Guayaquil",  "Guayas"),
    "PEUGEOT SCALA SHOPPING":   ("Guayaquil",     "Guayas"),
    "AUTOCONSA JTM":            ("Guayaquil",     "Guayas"),

    # ── Cuenca (Azuay) ────────────────────────────────────────────────────────
    "PEUGEOT CUENCA":           ("Cuenca",        "Azuay"),
    "IMPORTADORA TOMEBAMBA CUENCA": ("Cuenca",    "Azuay"),
    "GBV CUENCA":               ("Cuenca",        "Azuay"),
    "SZK CUENCA":               ("Cuenca",        "Azuay"),
    "CUENCA":                   ("Cuenca",        "Azuay"),
    "OPEL CUENCA":              ("Cuenca",        "Azuay"),

    # ── Otras ciudades ────────────────────────────────────────────────────────
    "SANTO DOMINGO":            ("Santo Domingo", "Santo Domingo de los Tsáchilas"),
    "STO. DOMINGO":             ("Santo Domingo", "Santo Domingo de los Tsáchilas"),
    "AG. MACHALA":              ("Machala",       "El Oro"),
    "MANTA":                    ("Manta",         "Manabí"),
    "QUEVEDO":                  ("Quevedo",       "Los Ríos"),
    "GBV- Orellana Autos Ok":   ("Francisco de Orellana", "Orellana"),
}


def asignar_ubicacion(agencia: str) -> tuple:
    """Retorna (ciudad, provincia) para una agencia. Desconocido si no está mapeada."""
    if pd.isna(agencia):
        return ("Desconocido", "Desconocido")
    agencia_norm = str(agencia).strip()
    ubicacion = AGENCIA_UBICACION.get(agencia_norm)
    if ubicacion:
        return ubicacion
    return ("Desconocido", "Desconocido")


def parse_descripcion(client: anthropic.Anthropic, row: pd.Series) -> dict:
    """Llama a Claude para parsear una descripción y retorna dict con campos extraídos."""
    
    prompt = USER_PROMPT_TEMPLATE.format(
        marca=row.get("Marca", ""),
        modelo=row.get("Modelo", ""),
        anio=row.get("Año", ""),
        color=row.get("Color", ""),
        recorrido=row.get("Recorrido", ""),
        tipo_recorrido=row.get("Tipo Recorrido", "KM"),
        descripcion=row.get("Descripción", "")
    )
    
    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )
    
    raw = message.content[0].text.strip()
    
    # Limpiar por si viene con markdown
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    
    return json.loads(raw)


def convertir_millas_a_km(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte recorrido en millas a km."""
    mask = df["Tipo Recorrido"] == "MI"
    df.loc[mask, "Recorrido"] = (df.loc[mask, "Recorrido"] * 1.60934).round(0)
    df.loc[mask, "Tipo Recorrido"] = "KM"
    return df


def limpiar_dataset_base(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza básica antes del parsing."""
    
    # Filtrar filas sin datos esenciales
    df = df.dropna(subset=["Marca", "Modelo", "Año", "Precio Final Editado"])
    
    # Eliminar precios cero o negativos
    df = df[df["Precio Final Editado"] > 0]
    
    # Convertir millas a km
    df = convertir_millas_a_km(df)
    
    # Normalizar recorrido
    df["Recorrido"] = pd.to_numeric(df["Recorrido"], errors="coerce")
    
    # Eliminar outliers de recorrido (> 500,000 km es sospechoso)
    df = df[df["Recorrido"] < 500_000]
    
    # Año como entero
    df["Año"] = df["Año"].astype(int)
    
    # Asignar ciudad y provincia desde la agencia
    ubicaciones = df["Agencia"].apply(asignar_ubicacion)
    df["ciudad"]    = ubicaciones.apply(lambda x: x[0])
    df["provincia"] = ubicaciones.apply(lambda x: x[1])

    mapeadas = (df["ciudad"] != "Desconocido").sum()
    print(f"   📍 {mapeadas}/{len(df)} agencias con ubicación mapeada "
          f"({mapeadas/len(df)*100:.1f}%)")

    # Resetear índice
    df = df.reset_index(drop=True)

    print(f"✅ Dataset limpio: {len(df)} registros válidos")
    return df


def procesar_csv(input_path: str, output_path: str, batch_size: int = 50, delay: float = 0.3):
    """
    Pipeline completo:
    1. Carga y limpia el CSV original
    2. Parsea cada descripción con Claude
    3. Guarda CSV enriquecido
    """
    
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("❌ Define ANTHROPIC_API_KEY como variable de entorno")
    
    client = anthropic.Anthropic(api_key=api_key)
    
    print(f"\n📂 Cargando {input_path}...")
    df = pd.read_csv(input_path)
    print(f"   {len(df)} filas encontradas")
    
    print("\n🧹 Limpiando dataset base...")
    df = limpiar_dataset_base(df)
    
    # Si ya existe un output parcial, retomar desde donde quedó
    if os.path.exists(output_path):
        df_done = pd.read_csv(output_path)
        ya_procesados = len(df_done)
        print(f"\n⏩ Retomando desde fila {ya_procesados} (procesamiento parcial detectado)")
        df_pending = df.iloc[ya_procesados:].copy()
    else:
        df_done = pd.DataFrame()
        df_pending = df.copy()
        ya_procesados = 0
    
    if len(df_pending) == 0:
        print("✅ Todo ya fue procesado.")
        return
    
    print(f"\n🤖 Parseando {len(df_pending)} descripciones con Claude...\n")
    
    results = []
    errors = 0
    
    for i, (idx, row) in enumerate(tqdm(df_pending.iterrows(), total=len(df_pending), desc="Parsing")):
        try:
            parsed = parse_descripcion(client, row)
            results.append(parsed)
        except Exception as e:
            # En caso de error, guardar campos nulos para no perder el registro
            results.append({})
            errors += 1
            if errors <= 5:
                print(f"\n⚠️  Error en fila {ya_procesados + i}: {e}")
        
        # Rate limit: pequeña pausa entre llamadas
        time.sleep(delay)
        
        # Guardado incremental cada batch_size registros
        if (i + 1) % batch_size == 0:
            df_batch = df_pending.iloc[:i+1].copy()
            parsed_df = pd.DataFrame(results)
            df_batch = df_batch.reset_index(drop=True)
            df_combined = pd.concat([df_batch, parsed_df], axis=1)
            
            if len(df_done) > 0:
                df_final = pd.concat([df_done, df_combined], ignore_index=True)
            else:
                df_final = df_combined
            
            df_final.to_csv(output_path, index=False)
            tqdm.write(f"💾 Guardado parcial: {ya_procesados + i + 1} registros")
    
    # Guardar resultado final
    parsed_df = pd.DataFrame(results)
    df_pending_reset = df_pending.reset_index(drop=True)
    df_combined = pd.concat([df_pending_reset, parsed_df], axis=1)
    
    if len(df_done) > 0:
        df_final = pd.concat([df_done, df_combined], ignore_index=True)
    else:
        df_final = df_combined
    
    df_final.to_csv(output_path, index=False)
    
    print(f"\n✅ Parser completo!")
    print(f"   Registros procesados : {len(df_final)}")
    print(f"   Errores de parsing   : {errors}")
    print(f"   Output guardado en   : {output_path}")
    
    # Resumen de campos extraídos
    campos_nuevos = [c for c in parsed_df.columns if c in df_final.columns]
    print(f"\n📊 Campos extraídos ({len(campos_nuevos)}):")
    for campo in campos_nuevos:
        pct = df_final[campo].notna().mean() * 100
        print(f"   {campo:<25} {pct:5.1f}% completado")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parser de vehículos con Claude API")
    parser.add_argument("--input",  default="data.csv",           help="CSV de entrada")
    parser.add_argument("--output", default="catalogo_limpio.csv", help="CSV de salida")
    parser.add_argument("--batch",  type=int, default=50,          help="Guardado cada N registros")
    parser.add_argument("--delay",  type=float, default=0.3,       help="Pausa entre llamadas (segundos)")
    args = parser.parse_args()
    
    procesar_csv(args.input, args.output, args.batch, args.delay)