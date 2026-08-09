"""
Generic prediction on new, unseen data.

``ModelPredictor`` wraps a saved artifact bundle (cleaner + encoder + scaler +
model + metadata) and reapplies the exact same preprocessing steps learned
during training before generating predictions -- so the "Predictions" page
works for any dataset without extra wiring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from utils.helpers import load_artifact, safe_column_list
from utils.logger import get_logger

logger = get_logger(__name__)


class ModelPredictor:
    """Loads a trained artifact bundle and produces predictions on new data."""

    def __init__(self, artifact_bundle: Dict[str, Any]) -> None:
        """
        Args:
            artifact_bundle: Dict produced by build_artifact_bundle(), containing
                keys: 'model', 'cleaner', 'encoder', 'scaler', 'feature_columns',
                'numerical_columns', 'categorical_columns', 'target_column',
                'problem_type', 'target_encoder'.
        """
        self.bundle = artifact_bundle

    @classmethod
    def from_file(cls, path: Path) -> "ModelPredictor":
        """Load an artifact bundle previously saved via joblib."""
        return cls(load_artifact(path))

    def predict(self, new_df: pd.DataFrame) -> pd.DataFrame:
        """Run the full preprocessing + inference pipeline on new raw data.

        Args:
            new_df: Raw dataframe with (at minimum) the same feature columns
                used at training time. Extra columns are ignored; missing
                columns are treated as missing values.

        Returns:
            Copy of new_df with an added 'Prediction' column (and
            'Prediction_Label' when a target encoder was used).
        """
        cleaner = self.bundle["cleaner"]
        encoder = self.bundle["encoder"]
        scaler = self.bundle["scaler"]
        model = self.bundle["model"]
        feature_columns = self.bundle["feature_columns"]
        numerical_columns = self.bundle["numerical_columns"]
        categorical_columns = self.bundle["categorical_columns"]
        target_encoder = self.bundle.get("target_encoder")

        working = new_df.copy()
        for col in feature_columns:
            if col not in working.columns:
                working[col] = np.nan

        # Coerce numeric feature columns defensively: an uploaded file may store a
        # numeric column as text or with stray non-numeric values. Bad values become
        # NaN here and are then imputed by the saved cleaner instead of crashing.
        for col in numerical_columns:
            if col in working.columns:
                working[col] = pd.to_numeric(working[col], errors="coerce")

        cleaned = cleaner.transform(working)
        encoded = encoder.transform(cleaned)
        scaled = scaler.transform(encoded)

        model_input = scaled.reindex(columns=self._model_feature_names(scaled), fill_value=0)

        predictions = model.predict(model_input)

        result = new_df.copy()
        result["Prediction"] = predictions

        if target_encoder is not None:
            try:
                result["Prediction_Label"] = target_encoder.inverse_transform(predictions.astype(int))
            except (ValueError, IndexError) as exc:
                logger.warning("Could not inverse-transform predictions: %s", exc)

        if self.bundle["problem_type"] == "classification" and hasattr(model, "predict_proba"):
            proba = model.predict_proba(model_input)
            result["Prediction_Confidence"] = proba.max(axis=1).round(4)

        return result

    def _model_feature_names(self, scaled_df: pd.DataFrame) -> list:
        """Return the exact feature-column order the model was trained on."""
        trained_columns = self.bundle.get("model_feature_columns")
        if trained_columns:
            return safe_column_list(scaled_df, trained_columns) or trained_columns
        return list(scaled_df.columns)


def validate_new_data(new_df: pd.DataFrame, artifact_bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Check whether newly uploaded data is structurally compatible with a trained model.

    Called before prediction so the UI can show user-friendly errors/warnings
    instead of letting a malformed upload crash the pipeline deep inside
    preprocessing.

    Args:
        new_df: Raw uploaded dataframe.
        artifact_bundle: Bundle produced by build_artifact_bundle() / loaded via joblib.

    Returns:
        Dict with:
            is_compatible: False if prediction should be blocked entirely
                (empty file, or every required feature column is missing).
            missing_columns: Required feature columns absent from new_df.
            extra_columns: Columns present in new_df but unused by the model.
            missing_ratio: Fraction of required feature columns that are missing.
    """
    feature_columns = artifact_bundle.get("feature_columns") or []
    target_column = artifact_bundle.get("target_column")
    required = [c for c in feature_columns if c != target_column]

    present = set(new_df.columns)
    missing = [c for c in required if c not in present]
    extra = [c for c in new_df.columns if c not in required and c != target_column]
    missing_ratio = round(len(missing) / len(required), 3) if required else 0.0

    is_compatible = new_df.shape[0] > 0 and missing_ratio < 1.0

    return {
        "is_compatible": is_compatible,
        "missing_columns": missing,
        "extra_columns": extra,
        "missing_ratio": missing_ratio,
    }


def build_artifact_bundle(
    model: Any,
    cleaner: Any,
    encoder: Any,
    scaler: Any,
    feature_columns: list,
    numerical_columns: list,
    categorical_columns: list,
    target_column: str,
    problem_type: str,
    model_feature_columns: list,
    target_encoder: Optional[Any] = None,
    model_name: Optional[str] = None,
    metrics: Optional[Dict[str, Any]] = None,
    leaderboard: Optional[Any] = None,
    dataset_name: Optional[str] = None,
    trained_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Bundle every artifact needed to reproduce predictions on new data.

    Keeping this in one dict (instead of many separate files) means the whole
    trained pipeline can be saved/loaded with a single joblib call.

    Args:
        model, cleaner, encoder, scaler: Fitted pipeline objects.
        feature_columns, numerical_columns, categorical_columns: Raw column-type
            lists used to (re)build features from new data.
        target_column, problem_type: Task metadata.
        model_feature_columns: Exact, ordered feature columns the model was
            trained on (post cleaning/encoding/scaling).
        target_encoder: Label encoder used on the target, if the target was categorical.
        model_name: Display name of the trained model (e.g. "CatBoost"), for UI/reporting.
        metrics: Evaluation metrics for this model on the held-out test set.
        leaderboard: Full model-comparison table from the training run, for reference.
        dataset_name: Name of the dataset the model was trained on.
        trained_at: ISO-8601 timestamp of when training completed.
    """
    return {
        "model": model,
        "cleaner": cleaner,
        "encoder": encoder,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "target_column": target_column,
        "problem_type": problem_type,
        "model_feature_columns": model_feature_columns,
        "target_encoder": target_encoder,
        "model_name": model_name,
        "metrics": metrics,
        "leaderboard": leaderboard,
        "dataset_name": dataset_name,
        "trained_at": trained_at,
    }
