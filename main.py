"""
CLI entry point: run the full pipeline (clean -> encode -> scale -> split ->
train -> compare -> save) against any CSV/Excel file, without the dashboard.

Useful for a quick sanity check the moment the datathon dataset is released,
before wiring it into the Streamlit UI.

Usage:
    python main.py --data path/to/dataset.csv --target target_column_name
    python main.py --data path/to/dataset.csv --target target_column_name --problem-type regression
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from models.compare_models import ModelComparator
from models.predict import build_artifact_bundle
from preprocessing.leakage import detect_leaky_columns, summarize_leakage
from utils.config import (
    CLASSIFICATION_MODELS,
    MODELS_OUTPUT_DIR,
    REGRESSION_MODELS,
    REPORTS_OUTPUT_DIR,
)
from utils.helpers import (
    detect_column_types,
    detect_target_column,
    infer_problem_type,
    run_preprocessing_pipeline,
    save_artifact,
)
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generic tabular ML pipeline runner.")
    parser.add_argument("--data", required=True, help="Path to a CSV or Excel dataset.")
    parser.add_argument("--target", default=None, help="Target column name (auto-detected if omitted).")
    parser.add_argument(
        "--problem-type",
        default=None,
        choices=["classification", "regression", "time_series"],
        help="Problem type (auto-inferred from the target if omitted).",
    )
    parser.add_argument("--date-column", default=None, help="Date column, required for time_series.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Held-out test fraction.")
    parser.add_argument(
        "--keep-leaky-columns",
        action="store_true",
        help="Train on suspected target-leak columns instead of excluding them (not recommended).",
    )
    return parser.parse_args()


def load_dataset(path: str) -> pd.DataFrame:
    """Load a CSV or Excel file into a DataFrame."""
    suffix = Path(path).suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def main() -> None:
    """Run the end-to-end pipeline and print a leaderboard."""
    args = parse_args()

    df = load_dataset(args.data)
    logger.info("Loaded dataset with shape %s", df.shape)

    target_column = args.target or detect_target_column(df)
    problem_type = args.problem_type or infer_problem_type(df, target_column)
    logger.info("Target column: %s | Problem type: %s", target_column, problem_type)

    column_types = detect_column_types(df, target_column=target_column)

    # Scan for columns that reproduce the target on their own -- training on them
    # yields unrealistically perfect scores that collapse on real data.
    leakage_report = detect_leaky_columns(df, target_column, problem_type)
    suspected_leaks = summarize_leakage(leakage_report)
    if suspected_leaks:
        print("\n=== Suspected Target Leakage ===")
        print(leakage_report[leakage_report["is_suspected_leak"]].to_string(index=False))
        if args.keep_leaky_columns:
            print("\n--keep-leaky-columns set: training on them anyway (scores will be inflated).")
        else:
            print(f"\nExcluding from training: {', '.join(suspected_leaks)}")
            print("Pass --keep-leaky-columns to override.")

    exclude_columns = [] if args.keep_leaky_columns else suspected_leaks

    pipeline = run_preprocessing_pipeline(
        df=df,
        target_column=target_column,
        numerical_columns=column_types["numerical"],
        categorical_columns=column_types["categorical"],
        problem_type=problem_type,
        date_column=args.date_column,
        test_size=args.test_size,
        exclude_columns=exclude_columns,
    )

    model_names = CLASSIFICATION_MODELS if problem_type == "classification" else REGRESSION_MODELS
    comparator = ModelComparator(problem_type)
    leaderboard = comparator.run(
        model_names, pipeline["X_train"], pipeline["y_train"], pipeline["X_test"], pipeline["y_test"]
    )

    print("\n=== Model Leaderboard ===")
    print(leaderboard.to_string(index=False))

    best_model_name = comparator.get_best_model_name(leaderboard)
    artifact_bundle = build_artifact_bundle(
        model=comparator.get_model(best_model_name),
        cleaner=pipeline["cleaner"],
        encoder=pipeline["encoder"],
        scaler=pipeline["scaler"],
        feature_columns=column_types["categorical"] + column_types["numerical"],
        numerical_columns=column_types["numerical"],
        categorical_columns=column_types["categorical"],
        target_column=target_column,
        problem_type=problem_type,
        model_feature_columns=pipeline["model_feature_columns"],
        target_encoder=pipeline["target_encoder"],
    )

    model_path = Path(MODELS_OUTPUT_DIR) / "best_model.joblib"
    save_artifact(artifact_bundle, model_path)

    report_path = Path(REPORTS_OUTPUT_DIR) / "leaderboard.csv"
    leaderboard.to_csv(report_path, index=False)

    print(f"\nBest model: {best_model_name}")
    print(f"Saved model artifact to: {model_path}")
    print(f"Saved leaderboard to: {report_path}")


if __name__ == "__main__":
    main()
