"""Scraper Patiotuerca Usados.

Estrategia:
- Listado: HTTP requests + BeautifulSoup extrae JSON-LD de cada pagina (35 autos/pagina).
  No requiere Playwright para el listado. Maneja paginacion base64.
- Detalle (opcional con --enrich): Playwright abre ficha individual para campos extra
  (color, transmision, combustible, motor, ciudad).
- Itera marcas de MARCAS_SLUGS o --marcas, todas las paginas del listado por marca.
"""

import asyncio
import base64
import json
import re
import sys
import time
from datetime import datetime
from typing import Optional

import requests as http_requests
from bs4 import BeautifulSoup
from playwright.async_api import TimeoutError as PWTimeout, Error as PWError
import pandas as pd

from scraper.base import BaseScraper, log, register_scraper
from scraper.patiotuerca_precios import MARCAS_SLUGS

BASE = "https://ecuador.patiotuerca.com"
BASE_URL = f"{BASE}/usados/-/autos"
TIMEOUT = 30_000
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "es-EC,es;q=0.9",
}
BROWSER_HEADERS = {
    **HEADERS,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
    "Referer": "https://ecuador.patiotuerca.com/",
}


def parse_int(value) -> Optional[int]:
    if value is None:
        return None
    nums = re.sub(r"[^\d]", "", str(value))
    return int(nums) if nums else None


def _encode_page(page_number: int) -> str:
    """PatioTuerca usa base64 en el param de pagina. Pagina 2 -> encode('1') = 'MQ=='"""
    return base64.b64encode(str(page_number - 1).encode()).decode()


def _extract_id(url: str) -> str:
    m = re.search(r"/(\d+)(?:[/?#]|$)", url)
    return m.group(1) if m else ""


async def _scrape_listing_page_jsonld_playwright(page, url: str, max_retries: int = 3):
    """Extrae autos via JSON-LD del HTML renderizado en Playwright.
    Devuelve (lista_autos, total_paginas).
    """
    for attempt in range(max_retries):
        try:
            await page.goto(url, timeout=TIMEOUT, wait_until="domcontentloaded")
            await page.wait_for_timeout(2000)
        except Exception as e:
            log(f"  Playwright error en {url}: {e}")
            if attempt < max_retries - 1:
                await page.wait_for_timeout(3000)
                continue
            return [], 0

        # Leer el HTML renderizado desde Playwright
        try:
            html_content = await page.content()
        except Exception as e:
            log(f"  page.content() error (intento {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                await page.wait_for_timeout(3000)
                continue
            return [], 0

        # Detectar Cloudflare challenge y esperar
        cf_terms = ["just a moment", "un momento", "challenge", "verificaci"]
        if any(term in html_content.lower() for term in cf_terms):
            log(f"  ⚠️ Cloudflare detectado en {url} (intento {attempt+1}/{max_retries}), esperando...")
            await page.wait_for_timeout(8000)
            if attempt < max_retries - 1:
                continue
            return [], 0

        break

    soup = BeautifulSoup(html_content, "html.parser")

    # Detectar total de paginas desde paginacion
    total_pages = 1
    for a in soup.select(".pagination a, nav.pagination a, [class*='pagination'] a"):
        try:
            num = int(a.get_text(strip=True))
            if num > total_pages:
                total_pages = num
        except ValueError:
            continue

    cars = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            raw = script.string or ""
            if not raw.strip():
                continue
            data = json.loads(raw)
            items = data if isinstance(data, list) else [data]
            for item in items:
                if item.get("@type") != "Car":
                    continue
                offer = item.get("offers") or {}
                if isinstance(offer, list):
                    offer = offer[0] if offer else {}
                mileage = item.get("mileageFromOdometer") or {}
                if isinstance(mileage, dict):
                    mileage_val = parse_int(mileage.get("value"))
                else:
                    mileage_val = parse_int(mileage)
                car_url = item.get("url") or ""
                if car_url and not car_url.startswith("http"):
                    car_url = BASE + car_url
                brand_raw = item.get("brand") or ""
                brand = brand_raw.get("name") if isinstance(brand_raw, dict) else brand_raw
                cars.append({
                    "source":       "patiotuerca",
                    "listing_id":   _extract_id(car_url),
                    "brand":        brand,
                    "model":        item.get("model"),
                    "year":         parse_int(item.get("vehicleModelDate")),
                    "price":        parse_int(offer.get("price")),
                    "currency":     offer.get("priceCurrency", "USD"),
                    "mileage_km":   mileage_val,
                    "body_type":    item.get("bodyType"),
                    "color":        item.get("color"),
                    "name":         item.get("name"),
                    "url":          car_url,
                    "transmission": None,
                    "fuel":         None,
                    "engine_cc":    None,
                    "traction":     None,
                    "city":         None,
                    "province":     None,
                    "subtype":      None,
                })
        except Exception:
            continue

    log(f"  JSON-LD: {len(cars)} autos | paginas detectadas: {total_pages} | url: {url}")
    return cars, total_pages


def _norm_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def _fmt_duration(seconds: float) -> str:
    """Formatea segundos a HH:MM:SS o MM:SS."""
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


async def enviar_batch_registros(registros: list, endpoint: str, api_key: str) -> tuple[bool, str]:
    """Envía batch de registros al endpoint de Vercel.
    
    Args:
        registros: Lista de dicts con datos de autos
        endpoint: URL del endpoint
        api_key: Bearer token
        
    Returns:
        (success, mensaje)
    """
    if not registros or not endpoint:
        return True, "Sin registros o endpoint"
    
    try:
        import httpx
        async with httpx.AsyncClient(timeout=30) as client:
            payload = {"registros": registros}
            response = await client.post(
                endpoint,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
            )
            response.raise_for_status()
            log(f"  ✓ Enviados {len(registros)} registros al endpoint ({response.status_code})")
            return True, f"Enviados {len(registros)}"
    except Exception as e:
        log(f"  ✗ Error al enviar batch: {e}")
        return False, str(e)


@register_scraper
class PatiotuercaUsadosScraper(BaseScraper):
    name = "patiotuerca_usados"
    fuente = "patiotuerca_usados"

    def default_marcas(self) -> list:
        return MARCAS_SLUGS

    def __init__(self, marcas=None, max_modelos: int = 999):
        super().__init__(marcas=marcas, max_modelos=max_modelos)
        # max_modelos actua como limite de paginas por marca (999 = sin limite practico)
        self.max_pages_per_brand = max(1, max_modelos)

    async def obtener_modelos(self, page, marca: str) -> dict:
        return {}

    async def scrape_precio(self, page, marca, modelo_slug, modelo_nombre, anio):
        return None

    def _brand_listing_url(self, marca: str, page_num: int) -> str:
        base = f"{BASE_URL}/-/{marca}"
        if page_num <= 1:
            return base
        return f"{base}?page={_encode_page(page_num)}"

    async def _scrape_marca_async(self, page, marca: str) -> list:
        """Recorre todas las paginas de una marca extrayendo JSON-LD por Playwright.
        Continúa hasta que no haya nuevos registros (carga dinámica de paginación)."""
        all_cars: list = []
        seen_ids: set = set()
        detected_pages_history = []

        for pnum in range(1, self.max_pages_per_brand + 1):
            url = self._brand_listing_url(marca, pnum)
            cars, detected_pages = await _scrape_listing_page_jsonld_playwright(page, url)
            detected_pages_history.append(detected_pages)

            # Log sobre paginación dinámica
            if pnum == 1:
                log(f"  marca={marca} pages_detectadas=[inicialmente: {detected_pages}] (dinámica)")

            new = [c for c in cars if c["listing_id"] not in seen_ids]
            for c in new:
                seen_ids.add(c["listing_id"])
                c["marca_slug"] = marca
            all_cars.extend(new)

            # Mostrar rango de páginas detectadas
            min_pages = min(detected_pages_history)
            max_pages = max(detected_pages_history)
            pages_info = f"{min_pages}-{max_pages}" if min_pages != max_pages else f"{max_pages}"
            log(f"  pagina {pnum} (progresión: {pages_info}): +{len(new)} autos (acumulado: {len(all_cars)})")

            if not new:
                log(f"  >> Sin nuevos autos en página {pnum}, finalizando {marca}")
                break

            time.sleep(0.8)

        return all_cars

    async def _enrich_detail(self, ctx, car: dict) -> dict:
        """Abre la ficha individual con Playwright para extraer campos enriquecidos."""
        url = car.get("url") or ""
        if not url:
            return car
        page = await ctx.new_page()
        try:
            await page.goto(url, timeout=TIMEOUT, wait_until="domcontentloaded")
            await page.wait_for_selector("#technicalData", timeout=12_000)

            fields: dict = {}
            for div in await page.query_selector_all("#technicalData div.col"):
                label_el = await div.query_selector("small")
                if not label_el:
                    continue
                label = (await label_el.inner_text()).strip()
                full = (await div.inner_text()).strip().replace("\n", " ")
                value = full.replace(label, "", 1).strip()
                if label:
                    fields[label] = value

            norm = {_norm_text(k): v for k, v in fields.items()}

            def gf(*labels):
                for lbl in labels:
                    v = norm.get(_norm_text(lbl))
                    if v:
                        return v
                return None

            crumbs = await page.query_selector_all(
                ".breadcrumb li a span[itemprop='name']"
            )
            city = (await crumbs[-1].inner_text()).strip() if crumbs else None
            province = (await crumbs[-2].inner_text()).strip() if len(crumbs) > 1 else None

            car.update({
                "transmission": gf("Transmision", "Transmisi\u00f3n"),
                "fuel":         gf("Combustible"),
                "engine_cc":    parse_int(gf("Motor(cilindraje)", "Motor (cilindraje)", "Motor")),
                "traction":     gf("Traccion", "Tracci\u00f3n"),
                "color":        car.get("color") or gf("Color"),
                "subtype":      gf("Subtipo"),
                "city":         city,
                "province":     province,
                "listing_id":   car.get("listing_id") or (
                    gf("Publicacion", "Publicaci\u00f3n") or ""
                ).replace("#", "").strip(),
            })
        except Exception:
            pass
        finally:
            await page.close()
        return car

    async def run(
        self,
        años: list = None,
        output_csv: str = "",
        endpoint: str = "",
        api_key: str = "",
        use_cdp: bool = False,
        enrich_detail: bool = False,
        close_endpoint: str = "",
    ) -> pd.DataFrame:
        years = list(años or range(2015, 2027))

        payload_preview: list = []
        ad_rows: list = []
        debug_rows: list = []
        seen_ids: set = set()
        years_set = set(years)
        resumed_count = 0

        if output_csv:
            try:
                prev = pd.read_csv(output_csv, encoding="utf-8")
                if "listing_id" in prev.columns:
                    seen_ids = set(prev["listing_id"].dropna().astype(str).tolist())
                    ad_rows = prev.to_dict("records")
                    resumed_count = len(ad_rows)
                    log(f"Reanudando — {len(prev)} filas previas, {len(seen_ids)} IDs vistos")
            except Exception:
                pass

        debug_path = ""
        if output_csv:
            debug_path = (
                output_csv.replace(".csv", "_debug.csv")
                if output_csv.lower().endswith(".csv")
                else f"{output_csv}_debug.csv"
            )

        log(f"[{self.name}] Iniciando — {len(self.marcas)} marcas | anos: {sorted(years_set)}")
        if enrich_detail:
            log("  Modo enriquecimiento: Playwright abrira fichas individuales")
        if output_csv:
            log(f"  Output: {output_csv}")
        print()

        ctx = None
        browser = None
        chrome_proc = None
        pw_instance = None

        # Siempre abrimos Playwright para leer el HTML renderizado con JSON-LD
        # Esto evita el 403 que recibe requests directo.
        from playwright.async_api import async_playwright as _apw
        pw_instance = _apw()
        pw = await pw_instance.__aenter__()
        if use_cdp:
            browser, _page, ctx, chrome_proc = await self._launch_chrome_cdp(pw)
        else:
            log("  Abriendo Playwright para leer JSON-LD del HTML renderizado...")
            browser = await pw.chromium.launch(headless=True)
            _page, ctx = await self.nueva_pagina(browser)
            # Navegar al listado para inicializar sesion
            try:
                await _page.goto(BASE_URL, timeout=TIMEOUT, wait_until="domcontentloaded")
                await _page.wait_for_timeout(2500)
            except Exception as e:
                log(f"  Advertencia al navegar para sesion: {e}")

        log(f"  Playwright listo para extraer JSON-LD")

        try:
            total_enviados = resumed_count
            t_start = time.time()
            marcas_tiempos = []  # segundos por marca

            for i, marca in enumerate(self.marcas, 1):
                t_marca_start = time.time()

                # --- Barra de progreso ---
                pct = (i - 1) / len(self.marcas)
                elapsed = time.time() - t_start
                bar_len = 30
                filled = int(bar_len * pct)
                bar = '█' * filled + '░' * (bar_len - filled)
                elapsed_str = _fmt_duration(elapsed)
                if marcas_tiempos:
                    avg = sum(marcas_tiempos) / len(marcas_tiempos)
                    remaining = avg * (len(self.marcas) - (i - 1))
                    eta_str = _fmt_duration(remaining)
                    rpm = (len(ad_rows) / elapsed * 60) if elapsed > 0 else 0
                    progress_line = f"  [{bar}] {i}/{len(self.marcas)} ({pct*100:.0f}%) | {elapsed_str} elapsed | ETA: {eta_str} | {len(ad_rows)} registros ({rpm:.0f}/min)"
                else:
                    progress_line = f"  [{bar}] {i}/{len(self.marcas)} ({pct*100:.0f}%) | {elapsed_str} elapsed | calculando ETA..."
                log(progress_line)
                log(f"[{i}/{len(self.marcas)}] {marca}")

                cars = await self._scrape_marca_async(_page, marca)
                if not cars:
                    log("  Sin autos encontrados via JSON-LD")
                    debug_rows.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "marca": marca, "url": "", "stage": "listing",
                        "status": "discarded", "reason": "no_jsonld_found", "note": "",
                    })
                    continue

                # Filtrar por anos solicitados
                filtered = [
                    c for c in cars
                    if c.get("year") is None or c["year"] in years_set
                ]
                log(f"  {len(cars)} autos totales -> {len(filtered)} en anos {sorted(years_set)}")

                for car in filtered:
                    lid = str(car.get("listing_id") or car.get("url") or "")
                    if lid and lid in seen_ids:
                        continue

                    if enrich_detail and ctx:
                        car = await self._enrich_detail(ctx, car)
                        await self.delay()

                    row = {
                        "fecha_scrape":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "fuente":        "patiotuerca",
                        "source":        "patiotuerca",
                        "listing_id":    car.get("listing_id"),
                        "brand":         car.get("brand"),
                        "model":         car.get("model"),
                        "subtype":       car.get("subtype"),
                        "year":          car.get("year"),
                        "price":         car.get("price"),
                        "currency":      car.get("currency", "USD"),
                        "mileage_km":    car.get("mileage_km"),
                        "body_type":     car.get("body_type"),
                        "color":         car.get("color"),
                        "transmission":  car.get("transmission"),
                        "fuel":          car.get("fuel"),
                        "engine_cc":     car.get("engine_cc"),
                        "traction":      car.get("traction"),
                        "city":          car.get("city"),
                        "province":      car.get("province"),
                        "url":           car.get("url"),
                        "marca_slug":    marca,
                    }
                    ad_rows.append(row)
                    payload_preview.append(row)
                    if lid:
                        seen_ids.add(lid)

                    log(
                        "  >> payload: "
                        + json.dumps(
                            {k: row[k] for k in [
                                "listing_id", "brand", "model", "year",
                                "price", "mileage_km", "url",
                            ]},
                            ensure_ascii=False,
                        )
                    )

                    debug_rows.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "marca": marca, "url": car.get("url", ""),
                        "stage": "listing", "status": "ok",
                        "reason": "jsonld_extracted",
                        "note": f"price={car.get('price')} year={car.get('year')}",
                    })

                # Guardar progreso despues de cada marca
                if output_csv and ad_rows:
                    pd.DataFrame(ad_rows).to_csv(output_csv, index=False, encoding="utf-8-sig")
                if debug_path and debug_rows:
                    pd.DataFrame(debug_rows).to_csv(debug_path, index=False, encoding="utf-8-sig")

                # Enviar registros de esta marca al endpoint (batches de 20)
                nuevos_marca = ad_rows[total_enviados:]
                if endpoint and api_key and nuevos_marca:
                    log(f"  [ENVÍO] Enviando {len(nuevos_marca)} registros de {marca}...")
                    for batch_idx in range(0, len(nuevos_marca), 20):
                        batch = nuevos_marca[batch_idx : batch_idx + 20]
                        success, msg = await enviar_batch_registros(batch, endpoint, api_key)
                        if not success:
                            log(f"  ⚠️ Batch {batch_idx//20 + 1} falló: {msg}")
                        await self.delay()
                    total_enviados = len(ad_rows)

                t_marca_elapsed = time.time() - t_marca_start
                marcas_tiempos.append(t_marca_elapsed)
                log(f"  marca completada: {len(filtered)} filas | acumulado total: {len(ad_rows)} | enviados: {total_enviados} | marca en {_fmt_duration(t_marca_elapsed)}")

            # Resumen final
            total_elapsed = time.time() - t_start
            log(f"\n{'='*60}")
            log(f"  SCRAPING COMPLETO")
            log(f"  Marcas: {len(self.marcas)} | Registros: {len(ad_rows)} | Enviados: {total_enviados}")
            log(f"  Tiempo total: {_fmt_duration(total_elapsed)}")
            log(f"{'='*60}")

            # Llamar al endpoint de cierre (actualizar estado de carros vendidos)
            if close_endpoint and api_key:
                try:
                    log(f"\n[CIERRE] Actualizando estado de carros (últimas 3 días)...")
                    import httpx
                    async with httpx.AsyncClient(timeout=30) as client:
                        response = await client.post(
                            close_endpoint,
                            json={"days": 3},
                            headers={
                                "Content-Type": "application/json",
                                "Authorization": f"Bearer {api_key}",
                            },
                        )
                        response.raise_for_status()
                        log(f"  ✓ Endpoint de cierre ejecutado ({response.status_code})")
                except Exception as e:
                    log(f"  ✗ Error en endpoint de cierre: {e}")

        finally:
            if ctx:
                await ctx.close()
            if browser:
                await browser.close()
            if chrome_proc:
                chrome_proc.kill()
            await pw_instance.__aexit__(None, None, None)

        df = pd.DataFrame(ad_rows)

        if output_csv and len(df) > 0:
            df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        if debug_path and debug_rows:
            pd.DataFrame(debug_rows).to_csv(debug_path, index=False, encoding="utf-8-sig")

        # Payload preview final completo
        if payload_preview:
            payload_to_send = {
                "endpoint": endpoint or "(envio_deshabilitado)",
                "registros": payload_preview,
                "total_registros": len(payload_preview),
            }
            payload_json = json.dumps(payload_to_send, ensure_ascii=False, indent=2)
            log("\n=== PAYLOAD COMPLETO (preview, no enviado) ===")
            print(payload_json)

            if output_csv:
                preview_path = (
                    output_csv.replace(".csv", "_payload_preview.json")
                    if output_csv.lower().endswith(".csv")
                    else f"{output_csv}_payload_preview.json"
                )
                with open(preview_path, "w", encoding="utf-8") as f:
                    f.write(payload_json)
                log(f"   Payload guardado en: {preview_path}")

        log(f"\n[OK] [{self.name}] Completado: {len(df)} autos scrapeados")
        if debug_path:
            log(f"   Debug: {debug_path}")

        if len(df) > 0 and "brand" in df.columns:
            log("\nAutos por marca:")
            print(df.groupby("brand")["price"].count().sort_values(ascending=False).to_string())

        return df

