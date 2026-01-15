from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
)
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import TransformedTargetRegressor
from sklearn.inspection import permutation_importance

from src.features.target_encoding import TargetEncoderCV
from src.utils.config import load_config, resolve_path
from src.utils.logging import setup_logging


LOGGER = setup_logging(__name__)


TARGET_COLUMN = "price"
DATE_COLUMNS = ["sale_date", "listing_date"]
EXCLUDE_COLUMNS = [TARGET_COLUMN]


def load_processed(cfg: Dict) -> pd.DataFrame:
    processed_dir = resolve_path(cfg, "processed")
    path = processed_dir / "training.parquet"
    if not path.exists():
        raise FileNotFoundError("Processed dataset not found. Run build_features first.")
    return pd.read_parquet(path)


def split_data(df: pd.DataFrame, cfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strategy = cfg.get("split", {}).get("strategy", "time")
    train_frac = float(cfg.get("split", {}).get("train_frac", 0.7))
    val_frac = float(cfg.get("split", {}).get("val_frac", 0.15))
    seed = int(cfg.get("split", {}).get("random_seed", 42))

    if strategy == "time" and "sale_date" in df.columns:
        df = df.sort_values("sale_date")
        n = len(df)
        train_end = int(n * train_frac)
        val_end = int(n * (train_frac + val_frac))
        train = df.iloc[:train_end]
        val = df.iloc[train_end:val_end]
        test = df.iloc[val_end:]
    else:
        df = df.sample(frac=1.0, random_state=seed)
        n = len(df)
        train_end = int(n * train_frac)
        val_end = int(n * (train_frac + val_frac))
        train = df.iloc[:train_end]
        val = df.iloc[train_end:val_end]
        test = df.iloc[val_end:]

    return train, val, test


def build_preprocessor(df: pd.DataFrame, cfg: Dict) -> Tuple[ColumnTransformer, List[str]]:
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLUMNS]
    feature_cols = [c for c in feature_cols if c not in DATE_COLUMNS]

    numeric_cols = df[feature_cols].select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    suburb_threshold = int(cfg.get("features", {}).get("suburb_encoding_threshold", 50))
    use_target_encoding = False
    if "suburb" in df.columns:
        n_suburbs = df["suburb"].nunique(dropna=True)
        use_target_encoding = n_suburbs > suburb_threshold

    cat_cols = [c for c in categorical_cols if c != "suburb"]

    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ]
                ),
                numeric_cols,
            )
        )

    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            )
        )

    if "suburb" in categorical_cols:
        if use_target_encoding:
            transformers.append(("suburb_te", TargetEncoderCV(cv=5), ["suburb"]))
        else:
            transformers.append(
                (
                    "suburb_oh",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    ["suburb"],
                )
            )

    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor, feature_cols


def get_strong_model() -> Tuple[str, object, Dict[str, List]]:
    try:
        import lightgbm as lgb

        model = lgb.LGBMRegressor(random_state=42)
        param_grid = {
            "num_leaves": [31, 63],
            "learning_rate": [0.05, 0.1],
            "n_estimators": [200, 500],
            "max_depth": [-1, 10],
        }
        return "lightgbm", model, param_grid
    except Exception:
        pass

    try:
        import xgboost as xgb

        model = xgb.XGBRegressor(
            random_state=42,
            objective="reg:squarederror",
            n_jobs=4,
        )
        param_grid = {
            "max_depth": [4, 6],
            "learning_rate": [0.05, 0.1],
            "n_estimators": [300, 600],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }
        return "xgboost", model, param_grid
    except Exception:
        pass

    from sklearn.ensemble import HistGradientBoostingRegressor

    model = HistGradientBoostingRegressor(random_state=42)
    param_grid = {
        "learning_rate": [0.05, 0.1],
        "max_depth": [6, 10],
        "max_iter": [200, 400],
    }
    return "hist_gbdt", model, param_grid


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1.0, None))) * 100
    med_ae = median_absolute_error(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "mape": mape, "median_abs_error": med_ae}


def plot_diagnostics(y_true: np.ndarray, y_pred: np.ndarray, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.3)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "--", color="gray")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(output_dir / "actual_vs_predicted.png")
    plt.close()

    residuals = y_true - y_pred
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=40, alpha=0.7)
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title("Residual Histogram")
    plt.tight_layout()
    plt.savefig(output_dir / "residual_hist.png")
    plt.close()


def save_feature_importance(
    model, preprocessor: ColumnTransformer, X_val: pd.DataFrame, y_val: np.ndarray, output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = None

    estimator = model.regressor_ if isinstance(model, TransformedTargetRegressor) else model
    if hasattr(estimator, "feature_importances_") and feature_names is not None:
        importances = estimator.feature_importances_
        imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        imp_df.sort_values("importance", ascending=False).head(50).to_csv(
            output_dir / "feature_importance.csv", index=False
        )
        return

    if feature_names is not None:
        perm = permutation_importance(model, X_val, y_val, n_repeats=5, random_state=42)
        perm_df = pd.DataFrame({"feature": feature_names, "importance": perm.importances_mean})
        perm_df.sort_values("importance", ascending=False).head(50).to_csv(
            output_dir / "permutation_importance.csv", index=False
        )


def train_model(cfg: Dict) -> None:
    df = load_processed(cfg)
    train_df, val_df, test_df = split_data(df, cfg)

    preprocessor, feature_cols = build_preprocessor(train_df, cfg)

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COLUMN].astype(float)
    X_val = val_df[feature_cols]
    y_val = val_df[TARGET_COLUMN].astype(float)
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COLUMN].astype(float)

    baseline = Ridge(alpha=1.0)
    baseline_pipeline = Pipeline(
        steps=[("preprocess", preprocessor), ("model", baseline)]
    )
    baseline_model = TransformedTargetRegressor(regressor=baseline_pipeline, func=np.log1p, inverse_func=np.expm1)
    baseline_model.fit(X_train, y_train)
    baseline_pred = baseline_model.predict(X_val)
    baseline_metrics = evaluate_metrics(y_val, baseline_pred)

    model_name, strong_model, param_grid = get_strong_model()
    n_iter = int(cfg.get("model", {}).get("n_iter", 15))
    sampler = list(ParameterSampler(param_grid, n_iter=n_iter, random_state=42))

    best_score = float("inf")
    best_model = None
    best_params = None
    for params in sampler:
        strong_model.set_params(**params)
        pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", strong_model)])
        wrapped = TransformedTargetRegressor(regressor=pipeline, func=np.log1p, inverse_func=np.expm1)
        wrapped.fit(X_train, y_train)
        preds = wrapped.predict(X_val)
        metrics = evaluate_metrics(y_val, preds)
        if metrics["mae"] < best_score:
            best_score = metrics["mae"]
            best_model = wrapped
            best_params = params

    LOGGER.info("Baseline MAE: %.2f", baseline_metrics["mae"])
    if best_model is None:
        raise RuntimeError("Failed to train strong model")

    LOGGER.info("Best %s MAE: %.2f", model_name, best_score)

    if baseline_metrics["mae"] <= best_score:
        selected = baseline_model
        selected_name = "ridge"
        selected_params = {"alpha": 1.0}
    else:
        selected = best_model
        selected_name = model_name
        selected_params = best_params or {}

    # Retrain on train + val
    train_val = pd.concat([train_df, val_df], ignore_index=True)
    preprocessor, feature_cols = build_preprocessor(train_val, cfg)
    X_train_val = train_val[feature_cols]
    y_train_val = train_val[TARGET_COLUMN].astype(float)

    if selected_name == "ridge":
        final_model = TransformedTargetRegressor(
            regressor=Pipeline(
                steps=[("preprocess", preprocessor), ("model", Ridge(alpha=selected_params.get("alpha", 1.0)))]
            ),
            func=np.log1p,
            inverse_func=np.expm1,
        )
    else:
        strong_name, strong_reg, _ = get_strong_model()
        if strong_name != selected_name:
            LOGGER.warning("Model availability changed; retraining with %s", strong_name)
        strong_reg.set_params(**selected_params)
        final_model = TransformedTargetRegressor(
            regressor=Pipeline(steps=[("preprocess", preprocessor), ("model", strong_reg)]),
            func=np.log1p,
            inverse_func=np.expm1,
        )

    final_model.fit(X_train_val, y_train_val)
    test_preds = final_model.predict(X_test)
    test_metrics = evaluate_metrics(y_test, test_preds)

    reports_dir = resolve_path(cfg, "reports")
    figures_dir = reports_dir / "figures"
    plot_diagnostics(y_test.to_numpy(), test_preds, figures_dir)
    save_feature_importance(final_model, preprocessor, X_val, y_val.to_numpy(), figures_dir)

    metrics_path = reports_dir / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "selected_model": selected_name,
        "selected_params": selected_params,
        "baseline_val_metrics": baseline_metrics,
        "strong_val_mae": best_score,
        "test_metrics": test_metrics,
        "typical_error": f"Median absolute error is ±${test_metrics['median_abs_error']:,.0f}",
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    models_dir = resolve_path(cfg, "models")
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": final_model,
            "feature_columns": feature_cols,
            "selected_model": selected_name,
            "config": cfg,
        },
        models_dir / "best_model.joblib",
    )
    joblib.dump(
        {
            "model": baseline_model,
            "feature_columns": feature_cols,
            "selected_model": "ridge",
            "config": cfg,
        },
        models_dir / "baseline_model.joblib",
    )

    if best_model is not None:
        joblib.dump(
            {
                "model": best_model,
                "feature_columns": feature_cols,
                "selected_model": model_name,
                "config": cfg,
            },
            models_dir / "strong_model.joblib",
        )

    LOGGER.info("Saved model artifacts to %s", models_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train house price prediction model")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_model(cfg)


if __name__ == "__main__":
    main()
