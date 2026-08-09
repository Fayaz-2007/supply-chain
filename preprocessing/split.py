"""
Generic train/test splitting.

Handles classification, regression, and time-series problems uniformly.
For time series, the split is chronological (no shuffling, sorted by the
detected date column) instead of random.
"""

from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from utils.config import RANDOM_STATE
from utils.logger import get_logger

logger = get_logger(__name__)


def split_data(
    df: pd.DataFrame,
    target_column: str,
    problem_type: str = "classification",
    test_size: float = 0.2,
    date_column: Optional[str] = None,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features/target into train and test sets.

    Args:
        df: Full, already-cleaned/encoded dataframe (features + target).
        target_column: Name of the target column.
        problem_type: One of 'classification', 'regression', 'time_series'.
        test_size: Fraction of rows held out for testing.
        date_column: Required for chronological splitting when problem_type
            is 'time_series'. Ignored otherwise.
        random_state: Seed for reproducible random splits.

    Returns:
        X_train, X_test, y_train, y_test.
    """
    working = df.copy()

    if problem_type == "time_series" and date_column and date_column in working.columns:
        working = working.sort_values(date_column)
        split_idx = int(len(working) * (1 - test_size))
        train_df = working.iloc[:split_idx]
        test_df = working.iloc[split_idx:]

        feature_cols = [c for c in working.columns if c not in (target_column, date_column)]
        X_train, y_train = train_df[feature_cols], train_df[target_column]
        X_test, y_test = test_df[feature_cols], test_df[target_column]
        logger.info("Chronological split by '%s': train=%d, test=%d", date_column, len(X_train), len(X_test))
        return X_train, X_test, y_train, y_test

    feature_cols = [c for c in working.columns if c != target_column]
    X = working[feature_cols]
    y = working[target_column]

    stratify = y if problem_type == "classification" and y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    logger.info("Random split: train=%d, test=%d", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test
