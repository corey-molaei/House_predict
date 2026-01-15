from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd

from src.utils.config import load_config, resolve_path
from src.utils.logging import setup_logging


LOGGER = setup_logging(__name__)


CANONICAL_COLUMNS = {
    "price": ["price", "sold_price", "sale_price", "soldPrice", "amount"],
    "sale_date": ["sale_date", "sold_date", "date", "soldDate"],
    "listing_date": ["listing_date", "listed_date", "listingDate"],
    "suburb": ["suburb", "locality"],
    "postcode": ["postcode", "post_code", "zip"],
    "lga": ["lga", "local_government_area"],
    "property_type": ["property_type", "type", "propertyType"],
    "bedrooms": ["bedrooms", "beds", "bed"],
    "bathrooms": ["bathrooms", "baths", "bath"],
    "parking": ["parking", "carspaces", "garage"],
    "land_size_sqm": ["land_size", "land_size_sqm", "landArea"],
    "building_size_sqm": ["building_size", "building_size_sqm", "buildingArea"],
    "lat": ["lat", "latitude"],
    "lon": ["lon", "lng", "longitude"],
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw Kaggle columns into canonical schema."""
    df = _normalize_columns(df)
    mapped: Dict[str, pd.Series] = {}
    for canonical, aliases in CANONICAL_COLUMNS.items():
        for name in aliases:
            if name in df.columns:
                mapped[canonical] = df[name]
                break
        if canonical not in mapped:
            mapped[canonical] = pd.Series([pd.NA] * len(df))
    out = pd.DataFrame(mapped)
    out["source"] = "kaggle"
    return out


def ingest_kaggle(input_path: Path, output_path: Path) -> None:
    """Read Kaggle data, standardize columns, and write parquet."""
    if input_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)
    df = standardize_columns(df)
    df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce")
    df["listing_date"] = pd.to_datetime(df["listing_date"], errors="coerce")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    LOGGER.info("Wrote %s rows to %s", len(df), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Kaggle Sydney house prices dataset")
    parser.add_argument("--input", type=str, default=None, help="Path to Kaggle CSV or parquet")
    parser.add_argument("--output", type=str, default=None, help="Output parquet path")
    args = parser.parse_args()

    cfg = load_config()
    raw_dir = resolve_path(cfg, "kaggle_raw")
    interim_dir = resolve_path(cfg, "interim")

    input_path = Path(args.input) if args.input else raw_dir / cfg["ingest"]["kaggle_filename"]
    output_path = Path(args.output) if args.output else interim_dir / "kaggle_clean.parquet"

    if not input_path.exists():
        raise FileNotFoundError(f"Kaggle dataset not found at {input_path}")

    ingest_kaggle(input_path, output_path)


if __name__ == "__main__":
    main()
