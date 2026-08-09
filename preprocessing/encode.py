"""
Generic categorical-feature encoding.

``FeatureEncoder`` automatically chooses one-hot encoding for low-cardinality
columns and frequency/label encoding for high-cardinality columns, learning
all mappings on the training set and re-applying them consistently on any
future data (validation, test, or newly uploaded prediction data).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from utils.config import LOW_CARDINALITY_THRESHOLD
from utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEncoder:
    """Fit/transform style encoder for categorical columns of any dataset."""

    def __init__(self, low_cardinality_threshold: int = LOW_CARDINALITY_THRESHOLD) -> None:
        self.low_cardinality_threshold = low_cardinality_threshold

        self._onehot_columns: List[str] = []
        self._onehot_categories: Dict[str, List[str]] = {}
        self._label_encoders: Dict[str, LabelEncoder] = {}
        self._high_cardinality_columns: List[str] = []
        self._is_fitted = False

    def fit(self, df: pd.DataFrame, categorical_columns: List[str]) -> "FeatureEncoder":
        """Learn category mappings from the training dataframe.

        Args:
            df: Training dataframe (post-cleaning).
            categorical_columns: Categorical feature columns to encode.

        Returns:
            self, for chaining.
        """
        for col in categorical_columns:
            if col not in df.columns:
                continue
            n_unique = df[col].nunique(dropna=True)
            if n_unique <= self.low_cardinality_threshold:
                self._onehot_columns.append(col)
                self._onehot_categories[col] = sorted(df[col].dropna().unique().tolist())
            else:
                self._high_cardinality_columns.append(col)
                encoder = LabelEncoder()
                encoder.fit(df[col].astype(str).fillna("Unknown"))
                self._label_encoders[col] = encoder

        self._is_fitted = True
        logger.info(
            "FeatureEncoder fitted: %d one-hot columns, %d label-encoded columns",
            len(self._onehot_columns),
            len(self._high_cardinality_columns),
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the learned encodings to a dataframe.

        Unseen categories at prediction time are mapped to an "other"/unknown
        bucket instead of raising errors, keeping the pipeline robust for any
        new dataset slice.

        Args:
            df: Dataframe to encode.

        Returns:
            Fully numeric-encoded copy of the dataframe (categorical columns replaced).
        """
        if not self._is_fitted:
            raise RuntimeError("FeatureEncoder.fit() must be called before transform().")

        out = df.copy()

        for col in self._onehot_columns:
            if col not in out.columns:
                continue
            categories = self._onehot_categories[col]
            dummies = pd.get_dummies(out[col], prefix=col)
            expected_cols = [f"{col}_{cat}" for cat in categories]
            for expected_col in expected_cols:
                if expected_col not in dummies.columns:
                    dummies[expected_col] = 0
            dummies = dummies[expected_cols]
            out = out.drop(columns=[col]).join(dummies)

        for col, encoder in self._label_encoders.items():
            if col not in out.columns:
                continue
            known_classes = set(encoder.classes_)
            safe_values = out[col].astype(str).fillna("Unknown").apply(
                lambda v: v if v in known_classes else encoder.classes_[0]
            )
            out[col] = encoder.transform(safe_values)

        return out

    def fit_transform(self, df: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
        """Convenience wrapper for fit() followed by transform()."""
        return self.fit(df, categorical_columns).transform(df)


def encode_target(y: pd.Series) -> tuple:
    """Label-encode a classification target, returning encoded values + fitted encoder.

    Regression targets are passed through unchanged with a None encoder.

    Args:
        y: Raw target series.

    Returns:
        Tuple of (encoded_series, encoder_or_none).
    """
    if pd.api.types.is_numeric_dtype(y):
        return y, None
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(y.astype(str))
    return pd.Series(encoded, index=y.index, name=y.name), encoder
