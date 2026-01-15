from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error

from src.models.train import split_data
from src.utils.config import load_config, resolve_path
from src.utils.logging import setup_logging


LOGGER = setup_logging(__name__)


def evaluate(cfg: Dict, model_path: Path) -> Dict[str, float]:
    processed_dir = resolve_path(cfg, "processed")
    df = pd.read_parquet(processed_dir / "training.parquet")
    _, _, test_df = split_data(df, cfg)

    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_cols = artifact["feature_columns"]

    X_test = test_df[feature_cols]
    y_test = test_df["price"].astype(float)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mape = np.mean(np.abs((y_test - preds) / np.clip(y_test, 1.0, None))) * 100
    med_ae = median_absolute_error(y_test, preds)

    metrics = {"mae": mae, "rmse": rmse, "mape": mape, "median_abs_error": med_ae}
    LOGGER.info("Test metrics: %s", metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained model on test set")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    models_dir = resolve_path(cfg, "models")
    model_path = Path(args.model) if args.model else models_dir / "best_model.joblib"

    metrics = evaluate(cfg, model_path)
    reports_dir = resolve_path(cfg, "reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
