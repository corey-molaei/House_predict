from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from src.utils.config import load_config, resolve_path
from src.utils.logging import setup_logging


LOGGER = setup_logging(__name__)


def _read_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".geojson", ".json"}:
        return pd.read_json(path)
    return pd.read_csv(path)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    col_map = {
        "price": ["price", "sale_price", "sold_price"],
        "sale_date": ["sale_date", "sold_date", "date"],
        "suburb": ["suburb", "locality"],
        "postcode": ["postcode", "post_code"],
        "lga": ["lga", "local_government_area"],
        "property_type": ["property_type", "property"],
        "land_size_sqm": ["land_size", "land_area_sqm", "area"],
        "lat": ["lat", "latitude"],
        "lon": ["lon", "longitude"],
    }

    out = {}
    for canonical, aliases in col_map.items():
        for name in aliases:
            if name in df.columns:
                out[canonical] = df[name]
                break
        if canonical not in out:
            out[canonical] = pd.Series([pd.NA] * len(df))

    out_df = pd.DataFrame(out)
    out_df["listing_date"] = pd.NaT
    out_df["bedrooms"] = pd.NA
    out_df["bathrooms"] = pd.NA
    out_df["parking"] = pd.NA
    out_df["building_size_sqm"] = pd.NA
    out_df["source"] = "nsw"
    out_df["sale_date"] = pd.to_datetime(out_df["sale_date"], errors="coerce")
    return out_df


def ingest_nsw(raw_dir: Path, output_clean: Path, allowed_exts: List[str]) -> None:
    files = [p for p in raw_dir.glob("**/*") if p.suffix.lower() in allowed_exts]
    if not files:
        LOGGER.info("No NSW data files found in %s; skipping", raw_dir)
        return

    frames = []
    for path in files:
        LOGGER.info("Reading NSW file %s", path)
        frames.append(_normalize(_read_file(path)))

    df = pd.concat(frames, ignore_index=True)
    output_clean.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_clean, index=False)
    LOGGER.info("Wrote %s records to %s", len(df), output_clean)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest NSW open data exports")
    parser.add_argument("--raw-dir", type=str, default=None)
    parser.add_argument("--output-clean", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config()
    raw_dir = Path(args.raw_dir) if args.raw_dir else resolve_path(cfg, "nsw_raw")
    interim_dir = resolve_path(cfg, "interim")
    output_clean = Path(args.output_clean) if args.output_clean else interim_dir / "nsw_clean.parquet"

    allowed_exts = cfg.get("ingest", {}).get("nsw_allowed_extensions", [".csv", ".geojson", ".json"])
    ingest_nsw(raw_dir, output_clean, allowed_exts)


if __name__ == "__main__":
    main()
