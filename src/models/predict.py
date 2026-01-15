from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd

from src.features.build_features import add_derived_features
from src.utils.config import load_config, resolve_path
from src.utils.logging import setup_logging


LOGGER = setup_logging(__name__)


COLUMN_ALIASES = {
    "price": ["price", "sold_price", "sale_price"],
    "sale_date": ["sale_date", "sold_date", "date"],
    "listing_date": ["listing_date", "listed_date"],
    "suburb": ["suburb"],
    "postcode": ["postcode", "post_code"],
    "lga": ["lga"],
    "property_type": ["property_type", "type"],
    "bedrooms": ["bedrooms", "beds"],
    "bathrooms": ["bathrooms", "baths"],
    "parking": ["parking", "carspaces"],
    "land_size_sqm": ["land_size_sqm", "land_size"],
    "building_size_sqm": ["building_size_sqm", "building_size"],
    "lat": ["lat", "latitude"],
    "lon": ["lon", "lng", "longitude"],
    "source": ["source"],
}


def standardize_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    out = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for name in aliases:
            if name in df.columns:
                out[canonical] = df[name]
                break
        if canonical not in out:
            out[canonical] = pd.Series([pd.NA] * len(df))
    return pd.DataFrame(out)


def predict(cfg: Dict, input_path: Path, output_path: Path, model_path: Path) -> None:
    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_cols = artifact["feature_columns"]

    df = pd.read_csv(input_path)
    df = standardize_input(df)
    df = add_derived_features(df)

    for col in feature_cols:
        if col not in df.columns:
            df[col] = pd.NA

    X = df[feature_cols]
    preds = model.predict(X)
    output = df.copy()
    output["predicted_price"] = preds

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    LOGGER.info("Wrote predictions to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict house prices using trained model")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    models_dir = resolve_path(cfg, "models")
    model_path = Path(args.model) if args.model else models_dir / "best_model.joblib"

    predict(cfg, Path(args.input), Path(args.output), model_path)


if __name__ == "__main__":
    main()
