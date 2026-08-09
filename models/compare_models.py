"""
Model comparison across all trained candidates for the current dataset.

``ModelComparator`` trains every requested model, scores it on the held-out
test set, and produces a ranked leaderboard -- the "Compare Models" feature
in the dashboard.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from models.evaluate import evaluate_model
from models.train import ModelTrainer
from utils.logger import get_logger

logger = get_logger(__name__)

# Metric used to rank models, per problem type. Higher-is-better vs lower-is-better
# is handled explicitly in `rank_models`.
_PRIMARY_METRIC = {
    "classification": "F1",
    "regression": "RMSE",
    "time_series": "RMSE",
}
_LOWER_IS_BETTER = {"RMSE", "MAE", "MAPE"}


class ModelComparator:
    """Trains and compares multiple models for the active problem type."""

    def __init__(self, problem_type: str) -> None:
        self.problem_type = problem_type
        self.trainer = ModelTrainer(problem_type)
        self.trained_models: Dict[str, Any] = {}

    def run(
        self,
        model_names: List[str],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> pd.DataFrame:
        """Train and evaluate each requested model.

        Args:
            model_names: Display names of models to train (subset of the trainer's registry).
            X_train, y_train: Training data.
            X_test, y_test: Held-out test data.
            progress_callback: Optional callback invoked with a short, human-readable
                status message as each stage completes (e.g. "Training XGBoost",
                "Evaluating models", "Selecting best model") -- lets callers (e.g. the
                Streamlit UI) surface live progress without this class knowing about UI.

        Returns:
            Leaderboard DataFrame, one row per model, ranked best-first.
        """
        rows = []
        for name in model_names:
            model, train_time = self.trainer.train(name, X_train, y_train)
            self.trained_models[name] = model

            y_pred = model.predict(X_test)
            y_proba = None
            if self.problem_type == "classification" and hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)

            metrics = evaluate_model(self.problem_type, y_test, y_pred, y_proba)
            row = {"Model": name, "Training Time (s)": round(train_time, 3), **metrics}
            rows.append(row)

            if progress_callback:
                progress_callback(f"Training {name}")

        if progress_callback:
            progress_callback("Evaluating models")

        leaderboard = pd.DataFrame(rows)

        if progress_callback:
            progress_callback("Selecting best model")

        return self.rank_models(leaderboard)

    def rank_models(self, leaderboard: pd.DataFrame) -> pd.DataFrame:
        """Sort the leaderboard by the primary metric and attach a Rank column.

        Args:
            leaderboard: Raw metrics table (one row per model).

        Returns:
            Leaderboard sorted best-first with an integer 'Rank' column.
        """
        primary_metric = _PRIMARY_METRIC.get(self.problem_type, "F1")
        if primary_metric not in leaderboard.columns:
            return leaderboard

        ascending = primary_metric in _LOWER_IS_BETTER
        sorted_board = leaderboard.sort_values(primary_metric, ascending=ascending).reset_index(drop=True)
        sorted_board.insert(0, "Rank", range(1, len(sorted_board) + 1))
        return sorted_board

    def get_best_model_name(self, leaderboard: pd.DataFrame) -> Optional[str]:
        """Return the model name ranked #1 in a leaderboard produced by rank_models()."""
        if leaderboard.empty or "Rank" not in leaderboard.columns:
            return None
        return leaderboard.loc[leaderboard["Rank"] == 1, "Model"].iloc[0]

    def get_best_model_reason(self, leaderboard: pd.DataFrame) -> List[str]:
        """Explain, in plain language, why the top-ranked model was selected.

        Reasons are built dynamically from whichever metric columns are present
        in the leaderboard -- never hardcoded to a specific dataset or model.

        Args:
            leaderboard: Ranked leaderboard produced by rank_models()/run().

        Returns:
            Short list of human-readable reason strings, best reason first.
        """
        if leaderboard.empty or "Rank" not in leaderboard.columns:
            return []

        primary_metric = _PRIMARY_METRIC.get(self.problem_type, "F1")
        best_row = leaderboard.loc[leaderboard["Rank"] == 1].iloc[0]
        reasons: List[str] = []

        if primary_metric in leaderboard.columns:
            direction = "Lowest" if primary_metric in _LOWER_IS_BETTER else "Highest"
            reasons.append(f"{direction} {primary_metric.replace('_', ' ')}: {best_row[primary_metric]}")

        other_metrics = [
            c for c in leaderboard.columns
            if c not in ("Rank", "Model", "Training Time (s)", primary_metric)
        ]
        for metric in other_metrics[:2]:
            reasons.append(f"{metric.replace('_', ' ')}: {best_row[metric]}")

        reasons.append(f"Ranked #1 of {len(leaderboard)} models evaluated on the held-out test set")
        return reasons

    def get_model(self, name: str) -> Any:
        """Retrieve a previously trained model instance by name."""
        return self.trained_models.get(name)
