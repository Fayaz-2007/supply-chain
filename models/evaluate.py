"""
Generic model evaluation metrics for classification and regression.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_classification(
    y_true: pd.Series, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Compute standard classification metrics.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities for the positive class (binary) or
            full probability matrix (multiclass), used for ROC-AUC if possible.

    Returns:
        Dict of metric_name -> value.
    """
    average = "binary" if pd.Series(y_true).nunique() == 2 else "weighted"

    metrics: Dict[str, float] = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }

    if y_proba is not None:
        try:
            if average == "binary":
                proba = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                metrics["ROC_AUC"] = roc_auc_score(y_true, proba)
            else:
                metrics["ROC_AUC"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
        except (ValueError, IndexError) as exc:
            logger.warning("Could not compute ROC_AUC: %s", exc)

    return {k: round(float(v), 4) for k, v in metrics.items()}


def evaluate_regression(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute standard regression metrics.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        Dict of metric_name -> value.
    """
    mse = float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))
    metrics: Dict[str, float] = {
        "RMSE": mse ** 0.5,
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }
    try:
        metrics["MAPE"] = mean_absolute_percentage_error(y_true, y_pred) * 100
    except ValueError as exc:
        logger.warning("Could not compute MAPE: %s", exc)

    return {k: round(float(v), 4) for k, v in metrics.items()}


def evaluate_model(
    problem_type: str, y_true: pd.Series, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Dispatch to the correct evaluation function based on problem type.

    Args:
        problem_type: 'classification', 'regression', or 'time_series'.
        y_true: Ground-truth values/labels.
        y_pred: Model predictions.
        y_proba: Optional predicted probabilities (classification only).

    Returns:
        Dict of metric_name -> value.
    """
    if problem_type == "classification":
        return evaluate_classification(y_true, y_pred, y_proba)
    return evaluate_regression(y_true, y_pred)
