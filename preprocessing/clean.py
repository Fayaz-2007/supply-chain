"""
Generic data-cleaning utilities.

``DataCleaner`` performs dataset-agnostic cleaning: missing value imputation,
duplicate removal, outlier capping, and constant-column removal. It only
relies on column *types* (numerical/categorical) detected at runtime, never
on hardcoded column names.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)

MissingStrategy = Literal["mean", "median", "mode", "constant", "drop"]

# Suffixes appended to a datetime column's name for each engineered numeric feature.
_DATETIME_FEATURE_SUFFIXES: List[str] = ["year", "month", "day", "dayofweek", "days_since"]


class DataCleaner:
    """Encapsulates a configurable, reusable cleaning pipeline for any tabular dataset."""

    def __init__(
        self,
        numerical_strategy: MissingStrategy = "median",
        categorical_strategy: MissingStrategy = "mode",
        constant_fill_value: str = "Unknown",
        outlier_method: Optional[Literal["iqr", "none"]] = "iqr",
        outlier_factor: float = 1.5,
        drop_constant_columns: bool = True,
        drop_duplicates: bool = True,
    ) -> None:
        self.numerical_strategy = numerical_strategy
        self.categorical_strategy = categorical_strategy
        self.constant_fill_value = constant_fill_value
        self.outlier_method = outlier_method
        self.outlier_factor = outlier_factor
        self.drop_constant_columns = drop_constant_columns
        self.drop_duplicates = drop_duplicates

        # Learned during fit(), reused at transform() time for consistency.
        self._numerical_fill_values: Dict[str, float] = {}
        self._categorical_fill_values: Dict[str, str] = {}
        self._outlier_bounds: Dict[str, tuple] = {}
        self._constant_columns: List[str] = []
        self._boolean_columns: List[str] = []
        self._boolean_fill_values: Dict[str, bool] = {}
        self._datetime_columns: List[str] = []
        self._datetime_reference: Dict[str, pd.Timestamp] = {}
        self._id_columns: List[str] = []
        self._is_fitted = False

    def fit(
        self,
        df: pd.DataFrame,
        numerical_columns: List[str],
        categorical_columns: List[str],
        boolean_columns: Optional[List[str]] = None,
        datetime_columns: Optional[List[str]] = None,
        id_columns: Optional[List[str]] = None,
    ) -> "DataCleaner":
        """Learn imputation values, outlier bounds, and smart-preprocessing rules.

        Args:
            df: Training dataframe.
            numerical_columns: Names of numerical feature columns.
            categorical_columns: Names of categorical feature columns.
            boolean_columns: Boolean feature columns, converted to 0/1 at transform time.
            datetime_columns: Datetime feature columns, replaced with engineered numeric
                features (year/month/day/day-of-week/days-since-earliest) at transform time.
                Any date column reserved for chronological time-series splitting should be
                excluded here so it survives untouched for the caller to split on.
            id_columns: Identifier-like columns, dropped entirely at transform time since
                they carry no generalizable signal for modeling.

        Returns:
            self, for chaining.
        """
        for col in numerical_columns:
            if col not in df.columns:
                continue
            if self.numerical_strategy == "mean":
                self._numerical_fill_values[col] = df[col].mean()
            elif self.numerical_strategy == "median":
                self._numerical_fill_values[col] = df[col].median()
            else:
                self._numerical_fill_values[col] = 0.0

            if self.outlier_method == "iqr":
                q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - self.outlier_factor * iqr
                upper = q3 + self.outlier_factor * iqr
                self._outlier_bounds[col] = (lower, upper)

        for col in categorical_columns:
            if col not in df.columns:
                continue
            if self.categorical_strategy == "mode" and not df[col].mode().empty:
                self._categorical_fill_values[col] = df[col].mode().iloc[0]
            else:
                self._categorical_fill_values[col] = self.constant_fill_value

        if self.drop_constant_columns:
            self._constant_columns = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]

        self._boolean_columns = [c for c in (boolean_columns or []) if c in df.columns]
        for col in self._boolean_columns:
            mode = df[col].mode()
            self._boolean_fill_values[col] = bool(mode.iloc[0]) if not mode.empty else False

        self._datetime_columns = [c for c in (datetime_columns or []) if c in df.columns]
        for col in self._datetime_columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            reference = parsed.min()
            self._datetime_reference[col] = reference if pd.notna(reference) else pd.Timestamp("1970-01-01")

        self._id_columns = [c for c in (id_columns or []) if c in df.columns]

        self._is_fitted = True
        logger.info(
            "DataCleaner fitted on %d numerical, %d categorical, %d boolean, "
            "%d datetime, %d id-like columns",
            len(numerical_columns),
            len(categorical_columns),
            len(self._boolean_columns),
            len(self._datetime_columns),
            len(self._id_columns),
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the learned cleaning steps to a (train or new) dataframe.

        Args:
            df: Dataframe to clean.

        Returns:
            Cleaned copy of the dataframe.
        """
        if not self._is_fitted:
            raise RuntimeError("DataCleaner.fit() must be called before transform().")

        out = df.copy()

        if self.drop_duplicates:
            before = len(out)
            out = out.drop_duplicates()
            if before != len(out):
                logger.info("Dropped %d duplicate rows", before - len(out))

        for col in self._constant_columns:
            if col in out.columns:
                out = out.drop(columns=[col])

        for col, fill_value in self._boolean_fill_values.items():
            if col in out.columns:
                out[col] = out[col].fillna(fill_value).astype(bool).astype(int)

        for col in self._datetime_columns:
            if col not in out.columns:
                continue
            parsed = pd.to_datetime(out[col], errors="coerce")
            reference = self._datetime_reference.get(col, pd.Timestamp("1970-01-01"))
            out[f"{col}_year"] = parsed.dt.year.fillna(reference.year).astype(int)
            out[f"{col}_month"] = parsed.dt.month.fillna(reference.month).astype(int)
            out[f"{col}_day"] = parsed.dt.day.fillna(reference.day).astype(int)
            out[f"{col}_dayofweek"] = parsed.dt.dayofweek.fillna(0).astype(int)
            out[f"{col}_days_since"] = (parsed - reference).dt.days.fillna(0).astype(int)
            out = out.drop(columns=[col])

        for col, fill_value in self._numerical_fill_values.items():
            if col in out.columns:
                out[col] = out[col].fillna(fill_value)
                if col in self._outlier_bounds:
                    lower, upper = self._outlier_bounds[col]
                    out[col] = out[col].clip(lower=lower, upper=upper)

        for col, fill_value in self._categorical_fill_values.items():
            if col in out.columns:
                out[col] = out[col].fillna(fill_value)

        for col in self._id_columns:
            if col in out.columns:
                out = out.drop(columns=[col])

        return out

    def fit_transform(
        self,
        df: pd.DataFrame,
        numerical_columns: List[str],
        categorical_columns: List[str],
        boolean_columns: Optional[List[str]] = None,
        datetime_columns: Optional[List[str]] = None,
        id_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Convenience wrapper for fit() followed by transform()."""
        return self.fit(
            df,
            numerical_columns,
            categorical_columns,
            boolean_columns=boolean_columns,
            datetime_columns=datetime_columns,
            id_columns=id_columns,
        ).transform(df)

    def get_engineered_numeric_columns(self) -> List[str]:
        """Return the numeric column names produced by datetime feature engineering.

        Useful for callers (e.g. the scaler step) that want to treat these new
        columns the same way as the originally detected numerical columns.
        """
        return [
            f"{col}_{suffix}"
            for col in self._datetime_columns
            for suffix in _DATETIME_FEATURE_SUFFIXES
        ]

    def get_removed_id_columns(self) -> List[str]:
        """Return the identifier-like columns dropped from the dataset during transform()."""
        return list(self._id_columns)


def get_missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build a tidy summary of missing values per column.

    Returns:
        DataFrame with columns: column, missing_count, missing_percent, dtype.
    """
    summary = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": df.isna().sum().values,
            "dtype": df.dtypes.astype(str).values,
        }
    )
    summary["missing_percent"] = (summary["missing_count"] / max(len(df), 1) * 100).round(2)
    return summary.sort_values("missing_count", ascending=False).reset_index(drop=True)
