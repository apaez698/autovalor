"""
Base scraper framework — Diseño extensible para scrapers de precios vehiculares.

Para crear un nuevo scraper:
1. Heredar de BaseScraper
2. Implementar obtener_modelos() y scrape_precio()
3. Registrarlo en SCRAPERS_REGISTRY
"""

import asyncio
import os
import random
import re
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


class ScraperResult:
    """Resultado estandarizado de un scrape de precio."""

    __slots__ = (
        "marca", "modelo_slug", "modelo_nombre", "año",
        "precio_ideal", "precio_min", "precio_max",
        "url", "enlace_listado", "fecha_scrape", "fuente",
    )

    def __init__(
        self,
        marca: str,
        modelo_slug: str,
        modelo_nombre: str,
        año: int,
        precio_ideal: float,
        precio_min: Optional[float] = None,
        precio_max: Optional[float] = None,
        url: str = "",
        enlace_listado: str = "",
        fuente: str = "unknown",
    ):
        self.marca = marca
        self.modelo_slug = modelo_slug
        self.modelo_nombre = modelo_nombre
        self.año = año
        self.precio_ideal = precio_ideal
        self.precio_min = precio_min
        self.precio_max = precio_max
        self.url = url
        self.enlace_listado = enlace_listado
        self.fecha_scrape = datetime.now().strftime("%Y-%m-%d")
        self.fuente = fuente

    def to_dict(self) -> dict:
        return {
            "marca": self.marca,
            "modelo_slug": self.modelo_slug,
            "modelo_nombre": self.modelo_nombre,
            "año": self.año,
            "precio_ideal": self.precio_ideal,
            "precio_min": self.precio_min,
            "precio_max": self.precio_max,
            "url": self.url,
            "enlace_listado": self.enlace_listado,
            "fecha_scrape": self.fecha_scrape,
            "fuente": self.fuente,
        }


def log(msg: str):
    import sys
    text = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        print(text.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8"), flush=True)


def limpiar_precio(texto: Optional[str]) -> Optional[float]:
    if not texto:
        return None
    nums = re.sub(r"[^\d]", "", str(texto))
    return float(nums) if nums else None


class BaseScraper(ABC):
    """Clase base para scrapers de precios vehiculares."""

    name: str = "base"
    fuente: str = "unknown"

    # Anti-blocking config
    delay_min: float = 1.5
    delay_max: float = 3.5
    pause_every_n_brands: int = 10
    pause_min: float = 10.0
    pause_max: float = 20.0
    batch_size: int = 50

    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    ]

    def __init__(self, marcas: Optional[list[str]] = None, max_modelos: int = 999):
        self.marcas = marcas or self.default_marcas()
        self.max_modelos = max_modelos

    @abstractmethod
    def default_marcas(self) -> list[str]:
        """Return the default list of brand slugs for this scraper."""
        ...

    @abstractmethod
    async def obtener_modelos(self, page, marca: str) -> dict[str, str]:
        """Return {slug: display_name} for a given brand."""
        ...

    @abstractmethod
    async def scrape_precio(
        self, page, marca: str, modelo_slug: str, modelo_nombre: str, año: int
    ) -> Optional[ScraperResult | dict]:
        """Scrape data for a specific brand/model/year. Return None if not found."""
        ...

    async def setup(self, browser):
        """Optional hook: called once before scraping starts."""
        pass

    async def teardown(self, browser):
        """Optional hook: called once after scraping finishes."""
        pass

    async def delay(self):
        await asyncio.sleep(random.uniform(self.delay_min, self.delay_max))

    @staticmethod
    def _find_chrome() -> Optional[str]:
        """Find the real Chrome installation on Windows."""
        import os as _os
        candidates = [
            _os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
            _os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
            _os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
        ]
        for p in candidates:
            if _os.path.exists(p):
                return p
        return None

    async def _launch_chrome_cdp(self, pw):
        """Launch real Chrome via subprocess and connect Playwright over CDP.
        
        This bypasses Cloudflare because Chrome runs without automation flags.
        The user must solve the Cloudflare challenge once in the visible browser.
        """
        import subprocess, os as _os
        from playwright.async_api import Error as PWError

        chrome = self._find_chrome()
        if not chrome:
            raise RuntimeError("Chrome not found. Install Google Chrome.")

        profile_dir = _os.path.abspath(".chrome_profile")
        port = 9222

        proc = None
        try:
            browser = await pw.chromium.connect_over_cdp(f"http://localhost:{port}")
            log("Conectado a instancia CDP existente de Chrome.")
        except Exception:
            log("Lanzando Chrome real (sin flag de automatización)...")
            proc = subprocess.Popen([
                chrome,
                f"--remote-debugging-port={port}",
                f"--user-data-dir={profile_dir}",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-extensions",
                "about:blank",
            ])
            await asyncio.sleep(3)
            browser = await pw.chromium.connect_over_cdp(f"http://localhost:{port}")

        ctx = browser.contexts[0]
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()

        # Navegar a usados para que el challenge aplique justo al flujo objetivo.
        log("Navegando a Patiotuerca usados — resuelve Cloudflare en el browser...")
        await page.goto(
            "https://ecuador.patiotuerca.com/usados/autos",
            timeout=60000,
        )

        for i in range(90):
            await asyncio.sleep(2)
            try:
                # Cloudflare puede recargar la pestaña y destruir contexto temporalmente.
                title = await page.title()
                current_url = page.url.lower()
            except PWError:
                continue

            challenge_terms = ["un momento", "just a moment", "verificaci", "challenge"]
            if not any(term in title.lower() for term in challenge_terms):
                log(f"[OK] Cloudflare superado — {title[:50]}")
                break
            # Si ya estamos en dominio correcto sin challenge explícito, continuar.
            if "ecuador.patiotuerca.com" in current_url and "__cf_chl" not in current_url:
                log("[OK] Cloudflare aparentemente superado (URL estable).")
                break
            if i % 10 == 0:
                log(f"  Esperando challenge... ({i*2}s)")
        else:
            raise RuntimeError("Timeout esperando Cloudflare. Intenta de nuevo.")

        await asyncio.sleep(2)
        return browser, page, ctx, proc

    async def nueva_pagina(self, browser):
        ua = random.choice(self.user_agents)
        ctx = await browser.new_context(
            user_agent=ua,
            locale="es-EC",
            viewport={"width": random.choice([1280, 1366, 1440, 1920]), "height": 800},
        )
        page = await ctx.new_page()
        return page, ctx

    async def enviar_batch(
        self, client: httpx.AsyncClient, batch: list[dict], endpoint: str, api_key: str
    ) -> tuple[int, int]:
        if not endpoint or not batch:
            return 0, 0
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            r = await client.post(
                endpoint,
                json={"registros": batch},
                headers=headers,
                timeout=30,
            )
            if r.status_code in (200, 201):
                return r.json().get("insertados", len(batch)), 0
            log(f"  API ERR {r.status_code}: {r.text[:150]}")
            return 0, len(batch)
        except Exception as e:
            log(f"  API ERR: {e}")
            return 0, len(batch)

    async def run(
        self,
        años: list[int],
        output_csv: str = "",
        endpoint: str = "",
        api_key: str = "",
        use_cdp: bool = False,
    ) -> pd.DataFrame:
        """Execute the full scrape pipeline."""
        from playwright.async_api import async_playwright

        results: list[dict] = []
        done_keys: set[tuple] = set()
        api_ok = api_err = 0
        pending_batch: list[dict] = []

        # Resume from previous CSV
        if output_csv and Path(output_csv).exists():
            try:
                df_prev = pd.read_csv(output_csv, encoding="utf-8")
                results = df_prev.to_dict("records")
                done_keys = {
                    (r["marca"], r["modelo_slug"], str(r.get("año", r.get("aÃ±o", ""))))
                    for r in results
                }
                log(f"Reanudando — {len(done_keys)} registros previos")
            except Exception:
                pass

        log(f"[{self.name}] Iniciando: {len(self.marcas)} marcas × {len(años)} años")
        if use_cdp:
            log("Modo: Chrome CDP (anti-Cloudflare)")
        if endpoint:
            log(f"Endpoint: {endpoint}")
        if output_csv:
            log(f"Output: {output_csv}")
        print()

        async with async_playwright() as pw:
            if use_cdp:
                browser, page, ctx, chrome_proc = await self._launch_chrome_cdp(pw)
            else:
                chrome_proc = None
                browser = await pw.chromium.launch(headless=True)
                await self.setup(browser)
                page, ctx = await self.nueva_pagina(browser)

            async with httpx.AsyncClient() as http:
                for i, marca in enumerate(self.marcas, 1):
                    log(f"[{i}/{len(self.marcas)}] {marca}")
                    modelos = await self.obtener_modelos(page, marca)

                    if not modelos:
                        log(f"  Sin modelos — skip")
                        continue

                    items = list(modelos.items())[: self.max_modelos]
                    log(f"  {len(items)} modelos")
                    marca_ok = 0

                    for modelo_slug, modelo_nombre in items:
                        for año in años:
                            key = (marca, modelo_slug, str(año))
                            if key in done_keys:
                                continue

                            result = await self.scrape_precio(
                                page, marca, modelo_slug, modelo_nombre, año
                            )

                            if result:
                                if isinstance(result, dict):
                                    data = result
                                    precio_log = float(
                                        data.get("precio_ideal")
                                        or data.get("precio_publicado")
                                        or 0
                                    )
                                else:
                                    data = result.to_dict()
                                    precio_log = float(result.precio_ideal)

                                results.append(data)
                                done_keys.add(key)
                                pending_batch.append(data)
                                marca_ok += 1
                                log(
                                    f"  ✓ {modelo_nombre} {año} → ${precio_log:,.0f}"
                                )

                                if endpoint and len(pending_batch) >= self.batch_size:
                                    ok, err = await self.enviar_batch(
                                        http, pending_batch, endpoint, api_key
                                    )
                                    api_ok += ok
                                    api_err += err
                                    log(f"  📤 Batch: {ok} OK | Total API: {api_ok}")
                                    pending_batch = []

                            await self.delay()

                    # Save CSV and send remaining batch per brand
                    if output_csv and results:
                        pd.DataFrame(results).to_csv(
                            output_csv, index=False, encoding="utf-8-sig"
                        )
                    if endpoint and pending_batch:
                        ok, err = await self.enviar_batch(
                            http, pending_batch, endpoint, api_key
                        )
                        api_ok += ok
                        api_err += err
                        pending_batch = []
                    log(
                        f"  💾 {marca_ok} nuevos | Total: {len(results)} | API: {api_ok}\n"
                    )

                    # Anti-blocking pause
                    if i % self.pause_every_n_brands == 0 and i < len(self.marcas):
                        pausa = random.uniform(self.pause_min, self.pause_max)
                        log(f"⏸  Pausa anti-bloqueo: {pausa:.0f}s...")
                        await asyncio.sleep(pausa)

            await ctx.close()
            if not use_cdp:
                await self.teardown(browser)
            await browser.close()
            if chrome_proc:
                chrome_proc.kill()

        df = pd.DataFrame(results)
        if output_csv and len(df) > 0:
            df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        log(f"\n[OK] [{self.name}] Completado: {len(df)} registros")
        if endpoint:
            log(f"   API: {api_ok} OK | {api_err} ERR")
        if len(df) > 0:
            log("\n📊 Registros por marca:")
            if "marca" in df.columns:
                if "precio_ideal" in df.columns:
                    metric_col = "precio_ideal"
                elif "precio_publicado" in df.columns:
                    metric_col = "precio_publicado"
                else:
                    metric_col = df.columns[0]
                print(
                    df.groupby("marca")[metric_col]
                    .count()
                    .sort_values(ascending=False)
                    .to_string()
                )

        return df


# ── Scraper registry ──────────────────────────────────────────────────────────

SCRAPERS_REGISTRY: dict[str, type[BaseScraper]] = {}


def register_scraper(cls: type[BaseScraper]) -> type[BaseScraper]:
    """Decorator to register a scraper class."""
    SCRAPERS_REGISTRY[cls.name] = cls
    return cls


def get_scraper(name: str) -> type[BaseScraper]:
    if name not in SCRAPERS_REGISTRY:
        available = ", ".join(SCRAPERS_REGISTRY.keys()) or "(none)"
        raise ValueError(f"Scraper '{name}' no encontrado. Disponibles: {available}")
    return SCRAPERS_REGISTRY[name]


def list_scrapers() -> list[str]:
    return list(SCRAPERS_REGISTRY.keys())
