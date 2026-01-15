# Sydney House Price Prediction (NSW)

Predict Sydney/Greater Sydney house prices in AUD using tabular ML models. This repo supports Kaggle data, optional Domain API listings, and optional NSW Government open data exports.

## Quickstart

1) Place Kaggle data in `data/raw/kaggle/sydney_house_prices.csv` (or parquet)  
2) (Optional) Set `DOMAIN_API_KEY` and configure `domain_base_url` and `domain_endpoint` in `config.yaml`  
3) (Optional) Add NSW open data exports (CSV/GeoJSON) into `data/raw/nsw/`  
4) Run the pipeline:

```bash
python -m src.ingest.kaggle_ingest
python -m src.ingest.domain_api_ingest
python -m src.ingest.nsw_ingest
python -m src.features.build_features
python -m src.models.train
python -m src.models.evaluate
```

Predict new listings:

```bash
python -m src.models.predict --input data/raw/new_properties.csv --output reports/preds.csv
```

## Data Sources

- Kaggle dataset “Sydney House Prices” (manually downloaded).
- Domain Developer API (optional; requires your API key and configured endpoint/base URL).
- NSW Government open datasets (optional; provide local CSV/GeoJSON exports).

This repo does not scrape websites directly. For NSW datasets, use available open data downloads or map services, and respect their terms and acceptable-use guidance.

## Notes on Domain API

You must provide:
- `DOMAIN_API_KEY` in your environment
- `domain_base_url` and `domain_endpoint` in `config.yaml` (or `DOMAIN_API_BASE_URL` / `DOMAIN_API_ENDPOINT`)

If any are missing, Domain ingestion is skipped gracefully.

## Project Structure

```
data/
  raw/
    kaggle/
    domain/
    nsw/
  interim/
  processed/
models/
notebooks/
reports/
  figures/
src/
  ingest/
  features/
  models/
  utils/
```

## Modeling

- Baseline: Ridge regression
- Strong model: LightGBM if available, else XGBoost, else HistGradientBoosting
- Target transform: `log1p(price)` with inverse transform at prediction time
- Splitting: time-based if `sale_date` exists, else random

## Outputs

- `data/interim/*_clean.parquet`
- `data/processed/training.parquet`
- `models/best_model.joblib`
- `reports/metrics.json`
- `reports/figures/actual_vs_predicted.png`
- `reports/figures/residual_hist.png`

## Next improvements

- add more NSW open datasets
- add spatial joins with LGA boundaries
- add model monitoring & drift checks
- deploy as FastAPI service
