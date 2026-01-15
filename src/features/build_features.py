from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.utils.config import load_config, resolve_path
from src.utils.geo import distance_to_cbd_km
from src.utils.logging import setup_logging


LOGGER = setup_logging(__name__)


KEY_NUMERIC_FIELDS = [
    "bedrooms",
    "bathrooms",
    "parking",
    "land_size_sqm",
    "building_size_sqm",
    "lat",
    "lon",
]


def load_interim_frames(interim_dir: Path) -> List[pd.DataFrame]:
    files = list(interim_dir.glob("*_clean.parquet"))
    frames = []
    for path in files:
        LOGGER.info("Loading %s", path)
        frames.append(pd.read_parquet(path))
    return frames


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "sale_date" not in df.columns:
        df["sale_date"] = pd.NaT
    if "listing_date" in df.columns:
        df["sale_date"] = df["sale_date"].fillna(df["listing_date"])
    df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce")

    df["year"] = df["sale_date"].dt.year
    df["month"] = df["sale_date"].dt.month
    df["quarter"] = df["sale_date"].dt.quarter
    df["days_since_epoch"] = (df["sale_date"] - pd.Timestamp("1970-01-01")).dt.days

    if "lat" in df.columns and "lon" in df.columns:
        lat = pd.to_numeric(df["lat"], errors="coerce")
        lon = pd.to_numeric(df["lon"], errors="coerce")
        df["distance_to_cbd_km"] = distance_to_cbd_km(lat, lon)

    if "bedrooms" in df.columns and "bathrooms" in df.columns:
        df["bedrooms_per_bathroom"] = df["bedrooms"] / df["bathrooms"].replace({0: np.nan})

    if "building_size_sqm" in df.columns and "bedrooms" in df.columns:
        df["building_size_per_bedroom"] = df["building_size_sqm"] / df["bedrooms"].replace({0: np.nan})

    for col in KEY_NUMERIC_FIELDS:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isna().astype(int)

    return df


def apply_outlier_filter(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    if not cfg.get("features", {}).get("outlier_filter", False):
        return df
    multiplier = float(cfg.get("features", {}).get("outlier_iqr_multiplier", 1.5))

    def _iqr_filter(series: pd.Series) -> pd.Series:
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        return series.between(lower, upper) | series.isna()

    filtered = df.copy()
    for col in ["price", "land_size_sqm", "building_size_sqm"]:
        if col in filtered.columns:
            mask = _iqr_filter(filtered[col].astype(float))
            filtered = filtered[mask]
    return filtered


def build_features(cfg: Dict, output_path: Path) -> None:
    interim_dir = resolve_path(cfg, "interim")
    frames = load_interim_frames(interim_dir)
    if not frames:
        raise FileNotFoundError("No interim parquet files found. Run ingestion first.")

    df = pd.concat(frames, ignore_index=True)
    df = add_derived_features(df)
    df = apply_outlier_filter(df, cfg)

    if "price" not in df.columns:
        raise ValueError("Target column 'price' not found after ingestion")

    df = df[df["price"].notna()]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    LOGGER.info("Wrote processed dataset with %s rows to %s", len(df), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build processed features dataset")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config()
    processed_dir = resolve_path(cfg, "processed")
    output_path = Path(args.output) if args.output else processed_dir / "training.parquet"

    build_features(cfg, output_path)


if __name__ == "__main__":
    main()
