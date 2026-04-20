"""
Scraper Patiotuerca Ecuador v3 — Basado en el framework extensible.
Scrapa precios de ecuador.patiotuerca.com/precio/autos/{marca}/{modelo}/{año}
"""

import asyncio
import argparse
import os
import re
from datetime import datetime
from typing import Optional

from playwright.async_api import TimeoutError as PWTimeout
from dotenv import load_dotenv

from scraper.base import (
    BaseScraper,
    ScraperResult,
    limpiar_precio,
    log,
    register_scraper,
)

load_dotenv()

BASE = "https://ecuador.patiotuerca.com"
TIMEOUT = 15_000

MARCAS_SLUGS = [
    "toyota", "chevrolet", "kia", "hyundai", "nissan", "mazda", "suzuki",
    "ford", "volkswagen", "honda", "peugeot", "renault", "mitsubishi", "jeep",
    "chery", "great+wall", "jac", "jetour", "byd", "mg", "haval", "dfsk", "jmc",
    "isuzu", "ram", "changan", "gac", "geely", "gwm", "audi", "bmw", "mercedes+benz",
    "land+rover", "lexus", "subaru", "volvo", "porsche", "maserati", "seat",
    "citroen", "ds+automobiles", "fiat", "opel", "alfa+romeo", "mini",
    "dongfeng", "foton", "faw", "livan", "maxus", "baic", "bestune",
    "shineray", "soueast", "zx+auto", "lifan", "neta", "brilliance",
]


@register_scraper
class PatiotuercaScraper(BaseScraper):
    name = "patiotuerca"
    fuente = "patiotuerca"

    def default_marcas(self) -> list[str]:
        return MARCAS_SLUGS

    async def obtener_modelos(self, page, marca: str) -> dict[str, str]:
        url = f"{BASE}/precio/autos/{marca}"
        try:
            await page.goto(url, timeout=TIMEOUT, wait_until="domcontentloaded")
            await page.wait_for_timeout(2000)
            titulo = await page.title()
            if "comprar carros" in titulo.lower() or "404" in titulo:
                return {}
            links = await page.query_selector_all(
                f"a[href*='/precio/autos/{marca}/']"
            )
            modelos = {}
            for link in links:
                href = await link.get_attribute("href") or ""
                partes = href.split(f"/precio/autos/{marca}/")
                if len(partes) > 1:
                    slug = partes[1].split("/")[0].split("?")[0]
                    if slug and not re.match(r"^\d{4}$", slug) and len(slug) > 1:
                        texto = (await link.inner_text()).strip()
                        if texto:
                            modelos[slug] = texto
            return modelos
        except Exception as e:
            log(f"  ERR modelos {marca}: {e}")
            return {}

    async def scrape_precio(
        self,
        page,
        marca: str,
        modelo_slug: str,
        modelo_nombre: str,
        año: int,
    ) -> Optional[ScraperResult]:
        url = f"{BASE}/precio/autos/{marca}/{modelo_slug}/{año}"
        try:
            await page.goto(url, timeout=TIMEOUT, wait_until="domcontentloaded")
            await page.wait_for_timeout(2000)
            titulo = await page.title()
            if "comprar carros" in titulo.lower() or "404" in titulo:
                return None

            ideal_el = await page.query_selector(".price-valuator h5.value")
            min_el = await page.query_selector(".price-down h5.value")
            max_el = await page.query_selector(".price-up h5.value")

            ideal_txt = await ideal_el.inner_text() if ideal_el else None
            precio_ideal = limpiar_precio(ideal_txt)
            if not precio_ideal:
                return None

            # Capturar enlace al listado de anuncios en Patiotuerca
            enlace_listado = ""
            try:
                listado_el = await page.query_selector(
                    "a[href*='/usados/'], a[href*='/nuevos/']"
                )
                if listado_el:
                    href = await listado_el.get_attribute("href") or ""
                    if href.startswith("/"):
                        enlace_listado = f"{BASE}{href}"
                    elif href.startswith("http"):
                        enlace_listado = href
            except Exception:
                pass
            if not enlace_listado:
                enlace_listado = f"{BASE}/usados/autos/{marca}/{modelo_slug}"

            return ScraperResult(
                marca=marca,
                modelo_slug=modelo_slug,
                modelo_nombre=modelo_nombre,
                año=año,
                precio_ideal=precio_ideal,
                precio_min=limpiar_precio(
                    await min_el.inner_text() if min_el else None
                ),
                precio_max=limpiar_precio(
                    await max_el.inner_text() if max_el else None
                ),
                url=url,
                enlace_listado=enlace_listado,
                fuente="patiotuerca",
            )
        except PWTimeout:
            return None
        except Exception as e:
            log(f"  ERR {modelo_nombre}/{año}: {e}")
            return None


async def main():
    parser = argparse.ArgumentParser(description="Scraper Patiotuerca Ecuador")
    parser.add_argument(
        "--anos", nargs="+", type=int, default=list(range(2018, 2027))
    )
    parser.add_argument(
        "--output",
        default=f"precios_mercado_{datetime.now().strftime('%Y%m%d')}.csv",
    )
    parser.add_argument("--endpoint", default=os.getenv("API_ENDPOINT", ""))
    parser.add_argument("--api-key", default=os.getenv("API_KEY", ""))
    parser.add_argument("--marcas", nargs="+")
    parser.add_argument("--max-modelos", type=int, default=999)
    args = parser.parse_args()

    scraper = PatiotuercaScraper(
        marcas=args.marcas,
        max_modelos=args.max_modelos,
    )
    await scraper.run(
        años=args.anos,
        output_csv=args.output,
        endpoint=args.endpoint,
        api_key=args.api_key,
    )


if __name__ == "__main__":
    asyncio.run(main())
