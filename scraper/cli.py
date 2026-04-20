"""
CLI to run scrapers.

Usage:
    uv run python -m scraper.cli patiotuerca_usados --anos 2020 2021 2022 2023 2024 2025
    uv run python -m scraper.cli --list
"""

import argparse
import asyncio
from datetime import datetime

from dotenv import load_dotenv

from scraper.base import get_scraper, list_scrapers
import scraper.patiotuerca_precios  # noqa: F401 — triggers @register_scraper
import scraper.patiotuerca_usados  # noqa: F401 — triggers @register_scraper

load_dotenv()


def main():
    ENDPOINT = "https://autocash-one.vercel.app/api/scraper-listings"
    API_KEY = "scrapper_feli_and"
    CLOSE_ENDPOINT = "https://autocash-one.vercel.app/api/scraper-close"

    parser = argparse.ArgumentParser(description="AutoValor — Scraper Runner")
    parser.add_argument("scraper", nargs="?", help="Nombre del scraper a ejecutar")
    parser.add_argument("--list", action="store_true", help="Listar scrapers disponibles")
    parser.add_argument("--anos", nargs="+", type=int, default=list(range(2018, 2027)))
    parser.add_argument("--output", default="")
    parser.add_argument("--marcas", nargs="+", help="Marcas a scrapear (default: todas)")
    parser.add_argument("--max-modelos", type=int, default=999)
    parser.add_argument("--cdp", action="store_true", help="Usar Chrome real vía CDP")
    parser.add_argument("--enrich", action="store_true", help="Enriquecer con detalle individual")
    args = parser.parse_args()

    if args.list:
        scrapers = list_scrapers()
        print("Scrapers disponibles:")
        for name in scrapers:
            print(f"  - {name}")
        return

    if not args.scraper:
        parser.error("Indica el nombre del scraper o usa --list")

    scraper_cls = get_scraper(args.scraper)
    scraper = scraper_cls(marcas=args.marcas, max_modelos=args.max_modelos)

    output = args.output or f"precios_{args.scraper}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    run_kwargs = dict(
        años=args.anos,
        output_csv=output,
        endpoint=ENDPOINT,
        api_key=API_KEY,
        use_cdp=args.cdp,
    )

    # Only pass extra kwargs if the scraper's run() accepts them
    import inspect
    run_params = inspect.signature(scraper.run).parameters
    if "enrich_detail" in run_params:
        run_kwargs["enrich_detail"] = args.enrich
    if "close_endpoint" in run_params:
        run_kwargs["close_endpoint"] = CLOSE_ENDPOINT

    asyncio.run(scraper.run(**run_kwargs))


if __name__ == "__main__":
    main()
