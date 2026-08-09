"""
Central configuration for the datathon starter kit.

Every path, constant, and registry used across the project is defined here so
that nothing is hardcoded to a specific dataset or scattered across modules.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


# --------------------------------------------------------------------------- #
# Base paths
# --------------------------------------------------------------------------- #
BASE_DIR: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

OUTPUTS_DIR: Path = BASE_DIR / "outputs"
MODELS_OUTPUT_DIR: Path = OUTPUTS_DIR / "models"
REPORTS_OUTPUT_DIR: Path = OUTPUTS_DIR / "reports"
FIGURES_OUTPUT_DIR: Path = OUTPUTS_DIR / "figures"

ASSETS_DIR: Path = BASE_DIR / "assets"

# Ensure all working directories exist regardless of where the app is launched from.
for _dir in (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_OUTPUT_DIR,
    REPORTS_OUTPUT_DIR,
    FIGURES_OUTPUT_DIR,
    ASSETS_DIR,
):
    os.makedirs(_dir, exist_ok=True)


# --------------------------------------------------------------------------- #
# General constants
# --------------------------------------------------------------------------- #
RANDOM_STATE: int = 42
DEFAULT_TEST_SIZE: float = 0.2

# Columns whose cardinality is below this threshold are treated as low-cardinality
# categoricals (safe for one-hot encoding). Above it, label/frequency encoding is used.
LOW_CARDINALITY_THRESHOLD: int = 15

# A column is considered "likely an identifier" (and excluded from modeling) if the
# ratio of unique values to total rows exceeds this threshold.
ID_LIKE_UNIQUE_RATIO: float = 0.95

# Problem types supported by the pipeline.
PROBLEM_TYPES: List[str] = ["classification", "regression", "time_series"]

# Human-readable labels for the sidebar / dropdowns.
PROBLEM_TYPE_LABELS: Dict[str, str] = {
    "classification": "Classification",
    "regression": "Regression",
    "time_series": "Time Series",
}

# Model registry keys shown in the UI, grouped by problem type. The actual estimator
# construction lives in models/train.py -- this only defines the display names/order.
CLASSIFICATION_MODELS: List[str] = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "XGBoost",
    "LightGBM",
    "CatBoost",
]

REGRESSION_MODELS: List[str] = [
    "Linear Regression",
    "Decision Tree",
    "Random Forest",
    "XGBoost",
    "LightGBM",
    "CatBoost",
]

# Time series is treated as a regression problem over a lag/feature-engineered frame.
TIME_SERIES_MODELS: List[str] = REGRESSION_MODELS

# Streamlit session_state keys, centralized to avoid typos scattered across pages.
class StateKeys:
    RAW_DF = "raw_df"
    PROCESSED_DF = "processed_df"
    TARGET_COLUMN = "target_column"
    DATE_COLUMN = "date_column"
    PROBLEM_TYPE = "problem_type"
    FEATURE_COLUMNS = "feature_columns"
    CATEGORICAL_COLUMNS = "categorical_columns"
    NUMERICAL_COLUMNS = "numerical_columns"
    X_TRAIN = "X_train"
    X_TEST = "X_test"
    Y_TRAIN = "y_train"
    Y_TEST = "y_test"
    TRAINED_MODELS = "trained_models"
    COMPARISON_RESULTS = "comparison_results"
    BEST_MODEL_NAME = "best_model_name"
    ARTIFACT_BUNDLE = "artifact_bundle"
    PREDICTIONS = "predictions"
    DATASET_NAME = "dataset_name"
    DATASET_SOURCE = "dataset_source"
    PREPROCESSING_BUNDLE = "preprocessing_bundle"
    LAST_PREDICTION_SECONDS = "last_prediction_seconds"
    EXCLUDED_COLUMNS = "excluded_columns"
    LEAKAGE_REPORT = "leakage_report"

    # AI Assistant (Gemini) -- additive only, never read/written by the ML pipeline.
    AI_CHAT_HISTORY = "ai_chat_history"
    AI_EXECUTIVE_SUMMARY = "ai_executive_summary"
    AI_RECOMMENDATIONS = "ai_recommendations"
    AI_INSIGHTS_CACHE_KEY = "ai_insights_cache_key"
    AI_PREDICTION_EXPLANATION = "ai_prediction_explanation"
    AI_REPORT_NARRATIVE = "ai_report_narrative"


@dataclass
class AppConfig:
    """Lightweight, mutable snapshot of app-wide settings (extend as needed)."""

    app_title: str = "Universal Business Analytics & ML Studio"
    random_state: int = RANDOM_STATE
    test_size: float = DEFAULT_TEST_SIZE
    theme_primary_color: str = "#6C5CE7"
    max_preview_rows: int = 200
    max_categorical_chart_levels: int = 20
    supported_upload_types: List[str] = field(default_factory=lambda: ["csv", "xlsx", "xls"])


CONFIG = AppConfig()
