"""
Generic model training.

``ModelTrainer`` exposes a single registry of estimators (classification and
regression) keyed by display name, so the dashboard and CLI can train any
supported algorithm without knowing anything about the underlying dataset.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Tuple

import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

from utils.config import RANDOM_STATE
from utils.logger import get_logger

logger = get_logger(__name__)

# Fixed model set trained automatically by the AutoML engine, per problem type.
# Deliberately a subset of each registry above -- one linear/baseline model plus
# three tree-ensemble models, giving a fast but representative comparison.
AUTOML_MODELS: Dict[str, list] = {
    "classification": ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "CatBoost"],
    "regression": ["Linear Regression", "Random Forest", "XGBoost", "LightGBM", "CatBoost"],
    "time_series": ["Linear Regression", "Random Forest", "XGBoost", "LightGBM", "CatBoost"],
}


def _build_classification_registry(random_state: int) -> Dict[str, Any]:
    """Construct fresh (untrained) classification estimators."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1),
        "XGBoost": XGBClassifier(random_state=random_state, eval_metric="logloss", verbosity=0),
        "LightGBM": LGBMClassifier(random_state=random_state, verbosity=-1),
        "CatBoost": CatBoostClassifier(random_state=random_state, verbose=False),
    }


def _build_regression_registry(random_state: int) -> Dict[str, Any]:
    """Construct fresh (untrained) regression estimators."""
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=random_state),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1),
        "XGBoost": XGBRegressor(random_state=random_state, verbosity=0),
        "LightGBM": LGBMRegressor(random_state=random_state, verbosity=-1),
        "CatBoost": CatBoostRegressor(random_state=random_state, verbose=False),
    }


class ModelTrainer:
    """Trains any registered model for a given problem type on plug-in data."""

    def __init__(self, problem_type: str, random_state: int = RANDOM_STATE) -> None:
        if problem_type not in ("classification", "regression", "time_series"):
            raise ValueError(f"Unsupported problem_type: {problem_type}")
        self.problem_type = problem_type
        self.random_state = random_state

        # time_series is modeled as regression over engineered/lagged features.
        effective_type = "regression" if problem_type == "time_series" else problem_type
        self._registry = (
            _build_classification_registry(random_state)
            if effective_type == "classification"
            else _build_regression_registry(random_state)
        )

    def available_models(self) -> list:
        """Return the display names of all models available for this problem type."""
        return list(self._registry.keys())

    def train(
        self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Tuple[Any, float]:
        """Train a single model by its display name.

        Args:
            model_name: Key from ``available_models()``.
            X_train: Training features.
            y_train: Training target.

        Returns:
            Tuple of (fitted_model, training_time_seconds).
        """
        if model_name not in self._registry:
            raise ValueError(f"Model '{model_name}' not available for {self.problem_type}")

        model = self._registry[model_name]
        logger.info("Training '%s' on %d rows / %d features", model_name, *X_train.shape)

        start = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - start

        logger.info("Finished training '%s' in %.2fs", model_name, elapsed)
        return model, elapsed
