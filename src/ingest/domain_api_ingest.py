from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from src.utils.config import env_or_config, load_config, resolve_path
from src.utils.logging import setup_logging


LOGGER = setup_logging(__name__)


def request_with_retry(
    url: str,
    headers: Dict[str, str],
    params: Optional[Dict[str, Any]] = None,
    max_retries: int = 3,
    sleep_seconds: float = 1.0,
) -> requests.Response:
    """Request with basic retry and rate limiting."""
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            if resp.status_code in {429, 500, 502, 503, 504}:
                LOGGER.warning("HTTP %s on attempt %s; sleeping %.1fs", resp.status_code, attempt, sleep_seconds)
                time.sleep(sleep_seconds)
                continue
            resp.raise_for_status()
            return resp
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            LOGGER.warning("Request failed on attempt %s: %s", attempt, exc)
            time.sleep(sleep_seconds)
    raise RuntimeError(f"Failed after {max_retries} attempts") from last_exc


def normalize_domain_records(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Normalize Domain API listing records into canonical schema."""
    rows: List[Dict[str, Any]] = []
    for rec in records:
        address = rec.get("address", {}) if isinstance(rec.get("address"), dict) else {}
        details = rec.get("details", {}) if isinstance(rec.get("details"), dict) else {}
        geo = rec.get("geoLocation", {}) if isinstance(rec.get("geoLocation"), dict) else {}

        rows.append(
            {
                "price": rec.get("price") or rec.get("soldPrice"),
                "sale_date": rec.get("soldDate"),
                "listing_date": rec.get("listingDate"),
                "suburb": address.get("suburb"),
                "postcode": address.get("postcode"),
                "lga": address.get("lga"),
                "property_type": details.get("propertyType"),
                "bedrooms": details.get("bedrooms"),
                "bathrooms": details.get("bathrooms"),
                "parking": details.get("carspaces"),
                "land_size_sqm": details.get("landArea"),
                "building_size_sqm": details.get("buildingArea"),
                "lat": geo.get("latitude"),
                "lon": geo.get("longitude"),
                "source": "domain",
            }
        )
    df = pd.DataFrame(rows)
    df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce")
    df["listing_date"] = pd.to_datetime(df["listing_date"], errors="coerce")
    return df


def ingest_domain(output_raw: Path, output_clean: Path, cfg: Dict[str, Any]) -> None:
    """Fetch listings from Domain API when configured and normalize them."""
    api_key = os.environ.get("DOMAIN_API_KEY")
    if not api_key:
        LOGGER.info("DOMAIN_API_KEY not set; skipping Domain API ingestion")
        return

    base_url = env_or_config(cfg, "DOMAIN_API_BASE_URL", "domain_base_url")
    endpoint = env_or_config(cfg, "DOMAIN_API_ENDPOINT", "domain_endpoint")
    if not base_url or not endpoint:
        LOGGER.info("Domain API base URL or endpoint not configured; skipping")
        return

    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    headers = {"X-API-Key": api_key}
    params = cfg.get("ingest", {}).get("domain_params", {})

    LOGGER.info("Requesting Domain API listings")
    resp = request_with_retry(url, headers=headers, params=params)
    data = resp.json()

    output_raw.parent.mkdir(parents=True, exist_ok=True)
    with open(output_raw, "w", encoding="utf-8") as f:
        json.dump(data, f)

    records = data if isinstance(data, list) else data.get("listings", [])
    df = normalize_domain_records(records)

    output_clean.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_clean, index=False)
    LOGGER.info("Wrote %s records to %s", len(df), output_clean)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Domain API listings if configured")
    parser.add_argument("--output-raw", type=str, default=None)
    parser.add_argument("--output-clean", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config()
    raw_dir = resolve_path(cfg, "domain_raw")
    interim_dir = resolve_path(cfg, "interim")

    output_raw = Path(args.output_raw) if args.output_raw else raw_dir / "domain_listings.json"
    output_clean = Path(args.output_clean) if args.output_clean else interim_dir / "domain_clean.parquet"

    ingest_domain(output_raw, output_clean, cfg)


if __name__ == "__main__":
    main()
