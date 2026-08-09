"""
Generic numerical feature scaling.

``FeatureScaler`` wraps scikit-learn scalers behind a single reusable
interface so the rest of the pipeline doesn't care which scaling strategy is
active. Fit on training data only, then applied consistently everywhere else.
"""

from __future__ import annotations

from typing import List, Literal

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from utils.logger import get_logger

logger = get_logger(__name__)

ScalerType = Literal["standard", "minmax", "robust", "none"]

_SCALER_REGISTRY = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


class FeatureScaler:
    """Fit/transform style scaler for numerical columns of any dataset."""

    def __init__(self, method: ScalerType = "standard") -> None:
        self.method = method
        self._scaler = _SCALER_REGISTRY[method]() if method != "none" else None
        self._columns: List[str] = []
        self._is_fitted = False

    def fit(self, df: pd.DataFrame, numerical_columns: List[str]) -> "FeatureScaler":
        """Learn scaling parameters from the training dataframe.

        Args:
            df: Training dataframe.
            numerical_columns: Numerical feature columns to scale.

        Returns:
            self, for chaining.
        """
        self._columns = [c for c in numerical_columns if c in df.columns]
        if self._scaler is not None and self._columns:
            self._scaler.fit(df[self._columns])
        self._is_fitted = True
        logger.info("FeatureScaler (%s) fitted on %d columns", self.method, len(self._columns))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the learned scaling to a dataframe.

        Args:
            df: Dataframe to scale.

        Returns:
            Copy of the dataframe with numerical columns scaled in place.
        """
        if not self._is_fitted:
            raise RuntimeError("FeatureScaler.fit() must be called before transform().")

        out = df.copy()
        if self._scaler is not None and self._columns:
            present = [c for c in self._columns if c in out.columns]
            out[present] = self._scaler.transform(out[present])
        return out

    def fit_transform(self, df: pd.DataFrame, numerical_columns: List[str]) -> pd.DataFrame:
        """Convenience wrapper for fit() followed by transform()."""
        return self.fit(df, numerical_columns).transform(df)
