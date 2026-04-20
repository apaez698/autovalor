# AutoValor - Vehicle Valuation ML API

FastAPI-based REST API for vehicle valuation predictions using XGBoost, with vehicle photo identification via Claude Vision and market price scraping from PatioTuerca.

## Project Structure

```
autovalor/
├── app/                        # FastAPI application
│   ├── main.py                # Entry point, middleware, router registration
│   ├── schemas.py             # Pydantic models (VehicleInput, ScoreInput, etc.)
│   ├── routes/
│   │   ├── predict.py         # POST /api/predict — XGBoost price prediction
│   │   ├── catalog.py         # GET /api/catalog/* — Vehicle catalog queries
│   │   ├── score.py           # POST /api/score — Business opportunity scoring
│   │   ├── identify.py        # POST /api/identify, /api/evaluate — Photo ID (Claude Vision)
│   │   ├── health.py          # GET /health
│   │   ├── metadata.py        # GET /api/metadata, /feature-importance, /info
│   │   └── scraper_api.py     # Scraper ingestion + retrain endpoints
│   └── services/
│       ├── model_service.py   # ML model loading & inference
│       ├── catalog_service.py # In-memory catalog queries
│       ├── vision_service.py  # Claude Vision integration
│       └── supabase.py        # Supabase REST helpers
├── ml/                         # Training pipeline
│   ├── feature_config.py      # Feature contract (15 features)
│   ├── train.py               # XGBoost training, metadata export
│   ├── preprocess.py          # Claude-powered description parsing
│   ├── build_catalog.py       # Generates vehicle_catalog.json
│   └── retrain.py             # Incremental retraining with scraped data
├── scraper/                    # Market price scrapers
│   ├── base.py                # Abstract scraper framework + Playwright engine
│   ├── cli.py                 # CLI runner
│   ├── patiotuerca_precios.py # Price predictions scraper
│   └── patiotuerca_usados.py  # Used listings scraper (JSON-LD)
├── models/                     # Trained model artifacts
│   ├── vehicle_model.pkl      # XGBoost model
│   ├── *_label_encoder.joblib # Categorical encoders (10)
│   ├── model_metadata.json    # Feature order, metrics, categorical values
│   └── vehicle_catalog.json   # 254 marca/modelo specs
├── data/                       # Training data and datasets
├── tests/                      # pytest suite
├── requirements.txt
├── pyproject.toml
└── render.yaml                 # Render.com deployment config
```

## Setup

```bash
# Create environment with uv
uv venv
source .venv/bin/activate       # Linux/Mac
.venv\Scripts\Activate.ps1      # Windows

# Install dependencies
uv pip install -r requirements.txt
```

## Usage

### Run the API

```bash
uv run uvicorn app.main:app --port 8001
```

### Train the model

```bash
uv run python -m ml.train
```

### Build vehicle catalog

```bash
uv run python -m ml.build_catalog
```

### Run scrapers

```bash
# Price predictions
uv run python -m scraper.cli patiotuerca --anos 2020 2021 2022

# Used listings
uv run python -m scraper.cli patiotuerca_usados --marcas toyota kia --anos 2023 2024
```

### Run tests

```bash
uv run pytest tests/ -v
```

## API Endpoints

| Method | Endpoint                              | Description                                   |
| ------ | ------------------------------------- | --------------------------------------------- |
| POST   | `/api/predict`                        | XGBoost price prediction                      |
| GET    | `/api/catalog/marcas`                 | All brands                                    |
| GET    | `/api/catalog/modelos/{marca}`        | Models by brand                               |
| GET    | `/api/catalog/specs/{marca}/{modelo}` | Vehicle specs                                 |
| GET    | `/api/catalog/search?q=`              | Search catalog                                |
| POST   | `/api/score`                          | Business opportunity scoring                  |
| POST   | `/api/identify`                       | Photo → vehicle ID (Claude Vision)            |
| POST   | `/api/evaluate`                       | Full flow: photo → identify → predict → score |
| GET    | `/health`                             | Service health check                          |
| GET    | `/api/metadata`                       | Feature order, categorical values, metrics    |

### Example Prediction

```bash
curl -X POST http://localhost:8001/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "marca": "TOYOTA",
    "modelo": "RAV4",
    "anio": 2022,
    "kilometraje": 35000,
    "motor_cc": 2000,
    "potencia_hp": 170,
    "carroceria": "SUV",
    "transmision": "AUTOMATICA",
    "tipo_combustible": "GASOLINA",
    "provincia": "PICHINCHA",
    "traccion": "4X2",
    "segmento": "SUV MEDIANO",
    "pais_origen": "JAPON",
    "color": "BLANCO"
  }'
```

## Deployment

Configured for [Render.com](https://render.com/) via `render.yaml`.

---

## Guía del Scraper

Guía paso a paso para ejecutar los scrapers de precios de PatioTuerca Ecuador.

### Requisitos previos

1. **Python 3.9+** instalado
2. **uv** (gestor de paquetes) — instalar con:

   ```bash
   # Windows (PowerShell)
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

   # Linux/Mac
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Google Chrome** instalado (solo si vas a usar modo CDP para el scraper de usados)

### Instalación

```bash
# 1. Clonar el repositorio
git clone <url-del-repo>
cd autovalor

# 2. Crear entorno virtual e instalar dependencias
uv venv
uv pip install -r requirements.txt

# 3. Instalar browsers de Playwright (necesario la primera vez)
uv run playwright install chromium
```

### Scrapers disponibles

| Scraper              | Comando               | Qué hace                                                                   | Velocidad                   |
| -------------------- | --------------------- | -------------------------------------------------------------------------- | --------------------------- |
| `patiotuerca`        | Precios de referencia | Scrapa precio ideal/min/max por marca/modelo/año desde `/precio/`          | Rápido (~1-2 min por marca) |
| `patiotuerca_usados` | Listados de usados    | Extrae anuncios reales con precios publicados desde `/usados/` vía JSON-LD | Medio (~2-5 min por marca)  |

Para ver los scrapers registrados:

```bash
uv run python -m scraper.cli --list
```

### Uso básico

#### Scraper de precios (recomendado para empezar)

```bash
# Todas las marcas, años 2018-2026
uv run python -m scraper.cli patiotuerca

# Solo Toyota y Kia, años 2022-2025
uv run python -m scraper.cli patiotuerca --marcas toyota kia --anos 2022 2023 2024 2025

# Limitar a 3 modelos por marca (test rápido)
uv run python -m scraper.cli patiotuerca --marcas toyota --anos 2024 --max-modelos 3

# Guardar en archivo específico
uv run python -m scraper.cli patiotuerca --output precios_toyota.csv
```

#### Scraper de usados

```bash
# Todas las marcas, años 2020-2026
uv run python -m scraper.cli patiotuerca_usados --anos 2020 2021 2022 2023 2024 2025

# Solo algunas marcas
uv run python -m scraper.cli patiotuerca_usados --marcas toyota chevrolet --anos 2023 2024

# Con enriquecimiento de detalle (abre cada ficha individual — más lento pero más datos)
uv run python -m scraper.cli patiotuerca_usados --marcas toyota --anos 2024 --enrich

# Con Chrome real vía CDP (necesario si Cloudflare bloquea)
uv run python -m scraper.cli patiotuerca_usados --cdp --anos 2024
```

### Opciones del CLI

| Opción          | Descripción                                                   | Default                 |
| --------------- | ------------------------------------------------------------- | ----------------------- |
| `--anos`        | Años a scrapear                                               | 2018-2026               |
| `--marcas`      | Marcas específicas (slugs)                                    | Todas (~50)             |
| `--max-modelos` | Máximo de modelos por marca                                   | 999 (sin límite)        |
| `--output`      | Archivo CSV de salida                                         | Auto-generado con fecha |
| `--cdp`         | Usar Chrome real vía CDP (anti-Cloudflare)                    | Desactivado             |
| `--enrich`      | Enriquecer con detalle individual (solo `patiotuerca_usados`) | Desactivado             |

### Marcas disponibles (slugs)

```
toyota, chevrolet, kia, hyundai, nissan, mazda, suzuki, ford, volkswagen,
honda, peugeot, renault, mitsubishi, jeep, chery, great+wall, jac, jetour,
byd, mg, haval, dfsk, jmc, isuzu, ram, changan, gac, geely, gwm, audi,
bmw, mercedes+benz, land+rover, lexus, subaru, volvo, porsche, maserati,
seat, citroen, ds+automobiles, fiat, opel, alfa+romeo, mini, dongfeng,
foton, faw, livan, maxus, baic, bestune, shineray, soueast, zx+auto,
lifan, neta, brilliance
```

### Modo CDP (Chrome real)

Si Cloudflare bloquea el scraping headless, usa `--cdp`:

1. Google Chrome debe estar instalado
2. Al ejecutar con `--cdp`, se abre Chrome con una ventana visible
3. **Resuelve el captcha/challenge de Cloudflare manualmente** en la ventana que se abre
4. Una vez pasado, el scraper continúa automáticamente

```bash
uv run python -m scraper.cli patiotuerca_usados --cdp --anos 2024
```

### Salida

El scraper genera un archivo CSV con los datos scrapeados. El nombre por defecto incluye la fecha:

```
precios_patiotuerca_20260419_195730.csv
```

**Columnas del scraper de precios:**
`marca, modelo_slug, modelo_nombre, año, precio_ideal, precio_min, precio_max, url, enlace_listado, fecha_scrape, fuente`

**Columnas del scraper de usados:**
`fecha_scrape, fuente, listing_id, brand, model, subtype, year, price, currency, mileage_km, body_type, color, transmission, fuel, engine_cc, traction, city, province, url, marca_slug`

### Reanudación automática

Si el scraper se interrumpe, **vuelve a ejecutar el mismo comando con `--output` apuntando al CSV existente**. El scraper detecta los registros ya scrapeados y continúa desde donde se quedó:

```bash
# Se interrumpió:
uv run python -m scraper.cli patiotuerca --output precios.csv

# Reanudar (mismo archivo):
uv run python -m scraper.cli patiotuerca --output precios.csv
```

### Troubleshooting

| Problema                         | Solución                                                                       |
| -------------------------------- | ------------------------------------------------------------------------------ |
| `playwright install` falla       | Ejecutar `uv run playwright install --with-deps chromium`                      |
| Cloudflare bloquea               | Usar `--cdp` para Chrome real                                                  |
| Timeout en muchas páginas        | Verificar conexión a internet, PatioTuerca puede estar caído                   |
| `UnicodeEncodeError` en Windows  | Ya está manejado internamente, pero puedes usar `chcp 65001` antes de ejecutar |
| Pocas marcas/modelos encontrados | Normal para marcas poco comunes en Ecuador                                     |
