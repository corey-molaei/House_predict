from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold


@dataclass
class TargetEncoderCV(BaseEstimator, TransformerMixin):
    """Target encoding with out-of-fold means to reduce leakage."""

    cv: int = 5
    smoothing: float = 10.0
    random_state: int = 42

    global_mean_: Optional[float] = None
    mapping_: Optional[pd.Series] = None

    def fit(self, X, y):
        x = self._to_series(X)
        y = pd.Series(y)
        self.global_mean_ = float(y.mean())
        stats = y.groupby(x).agg(["mean", "count"])
        smooth = (stats["mean"] * stats["count"] + self.global_mean_ * self.smoothing) / (
            stats["count"] + self.smoothing
        )
        self.mapping_ = smooth
        return self

    def fit_transform(self, X, y):
        x = self._to_series(X)
        y = pd.Series(y)
        self.global_mean_ = float(y.mean())
        encoded = pd.Series(index=x.index, dtype=float)

        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        for train_idx, valid_idx in kf.split(x):
            x_train = x.iloc[train_idx]
            y_train = y.iloc[train_idx]
            stats = y_train.groupby(x_train).agg(["mean", "count"])
            smooth = (stats["mean"] * stats["count"] + self.global_mean_ * self.smoothing) / (
                stats["count"] + self.smoothing
            )
            fold_map = smooth
            encoded.iloc[valid_idx] = x.iloc[valid_idx].map(fold_map).fillna(self.global_mean_)

        stats_full = y.groupby(x).agg(["mean", "count"])
        smooth_full = (stats_full["mean"] * stats_full["count"] + self.global_mean_ * self.smoothing) / (
            stats_full["count"] + self.smoothing
        )
        self.mapping_ = smooth_full

        return encoded.to_frame(name="suburb_te").values

    def transform(self, X):
        x = self._to_series(X)
        if self.mapping_ is None or self.global_mean_ is None:
            raise ValueError("TargetEncoderCV is not fitted")
        encoded = x.map(self.mapping_).fillna(self.global_mean_)
        return encoded.to_frame(name="suburb_te").values

    @staticmethod
    def _to_series(X) -> pd.Series:
        if isinstance(X, pd.DataFrame):
            if X.shape[1] != 1:
                raise ValueError("TargetEncoderCV expects a single column")
            return X.iloc[:, 0]
        if isinstance(X, pd.Series):
            return X
        return pd.Series(X)
