"""Build vehicle catalog JSON for autocomplete from training data."""

import json
import pandas as pd


def build_catalog(csv_path: str, output_path: str) -> list[dict]:
    df = pd.read_csv(csv_path)
    catalog = []

    for (marca, modelo), group in df.groupby(["Marca", "Modelo"]):
        versions = group["version"].dropna().unique().tolist()

        specs = {}
        spec_cols = [
            "carroceria", "motor_cc", "potencia_hp", "transmision",
            "traccion", "tipo_combustible", "segmento", "pais_origen",
        ]
        for col in spec_cols:
            vals = group[col].dropna()
            if len(vals) > 0:
                mode = vals.mode()
                specs[col] = mode.iloc[0] if len(mode) > 0 else vals.iloc[0]

        year_min = int(group["Año"].min())
        year_max = int(group["Año"].max())

        entry = {
            "marca": str(marca).strip().upper(),
            "modelo": str(modelo).strip().upper(),
            "versiones": sorted(set(str(v).strip().upper() for v in versions)),
            "año_min": year_min,
            "año_max": year_max,
            "count": len(group),
            "specs": {},
        }

        for k, v in specs.items():
            if hasattr(v, "item"):
                entry["specs"][k] = v.item()
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                entry["specs"][k] = float(v)
            else:
                entry["specs"][k] = str(v).strip().upper()

        catalog.append(entry)

    catalog.sort(key=lambda x: (-x["count"], x["marca"], x["modelo"]))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)

    return catalog


if __name__ == "__main__":
    catalog = build_catalog(
        "data/data_limpia_entrenamiento.csv",
        "models/vehicle_catalog.json",
    )
    marcas = set(e["marca"] for e in catalog)
    print(f"Catalog: {len(catalog)} entries, {len(marcas)} marcas")
    print("\nTop 15 by frequency:")
    for e in catalog[:15]:
        v_count = len(e["versiones"])
        print(f"  {e['marca']:12s} {e['modelo']:20s}  {e['count']:3d} records  {v_count:2d} versions  {e['año_min']}-{e['año_max']}")
    print(f"\nSaved to models/vehicle_catalog.json")
