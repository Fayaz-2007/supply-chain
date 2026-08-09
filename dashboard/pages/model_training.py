"""
Model Training page: a self-contained AutoML engine.

Given a dataset and target column already confirmed on Dataset Explorer, this
page automatically prepares the data, trains a fixed set of candidate models
for the detected problem type, evaluates and ranks them, selects the best
one, and persists every artifact required for later prediction -- with no
manual model selection required.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st

from models.compare_models import ModelComparator
from models.predict import build_artifact_bundle
from models.train import AUTOML_MODELS
from preprocessing.leakage import detect_leaky_columns, summarize_leakage
from utils.config import MODELS_OUTPUT_DIR, PROBLEM_TYPE_LABELS, StateKeys
from utils.helpers import run_preprocessing_pipeline, save_artifact
from utils.logger import get_logger

logger = get_logger(__name__)

# Session-state keys local to this page (not shared with other pages, so they
# don't belong in the central StateKeys registry).
_TOTAL_TIME_KEY = "automl_total_training_time"
_BEST_REASONS_KEY = "automl_best_model_reasons"

_NON_METRIC_COLUMNS = ("Rank", "Model", "Training Time (s)")


def _ensure_preprocessed(status: Any) -> bool:
    """Fit (or reuse) the preprocessing pipeline for the current dataset/target.

    Reuses the exact same `run_preprocessing_pipeline` the rest of the app relies
    on, so cleaning/encoding/scaling behavior is identical everywhere -- this
    function only adds progress reporting around that single call.
    """
    df = st.session_state.get(StateKeys.RAW_DF)
    target_column = st.session_state.get(StateKeys.TARGET_COLUMN)
    if df is None or target_column is None:
        return False

    bundle = st.session_state.get(StateKeys.PREPROCESSING_BUNDLE)
    if bundle is not None:
        status.write("✅ Preparing dataset (already prepared -- reusing)")
        status.write("✅ Splitting train/test (already split -- reusing)")
        return True

    excluded = st.session_state.get(StateKeys.EXCLUDED_COLUMNS, [])
    bundle = run_preprocessing_pipeline(
        df=df,
        target_column=target_column,
        numerical_columns=st.session_state.get(StateKeys.NUMERICAL_COLUMNS, []),
        categorical_columns=st.session_state.get(StateKeys.CATEGORICAL_COLUMNS, []),
        problem_type=st.session_state.get(StateKeys.PROBLEM_TYPE, "classification"),
        date_column=st.session_state.get(StateKeys.DATE_COLUMN),
        exclude_columns=excluded,
    )
    if bundle["dropped_columns"]:
        status.write(f"✅ Excluding {len(bundle['dropped_columns'])} leaky column(s) from training")
    if bundle["rows_dropped_missing_target"]:
        status.write(f"✅ Dropping {bundle['rows_dropped_missing_target']:,} row(s) with a missing target")
    status.write(f"✅ Splitting train/test ({len(bundle['X_train']):,} train / {len(bundle['X_test']):,} test rows)")
    status.write("✅ Preparing dataset (cleaning, encoding, scaling -- fitted on train split only)")

    st.session_state[StateKeys.PREPROCESSING_BUNDLE] = bundle
    return True


def _render_leakage_panel(df, target_column: str, problem_type: str) -> None:
    """Warn about suspected target leaks and let the user choose what to exclude.

    Flagged columns are pre-ticked for removal because a column that alone
    reproduces the target is almost always a downstream consequence of it --
    but the choice stays with the user, since occasionally such a column is a
    legitimately dominant predictor.
    """
    cache_key = (target_column, problem_type, id(df))
    if st.session_state.get("_leakage_cache_key") != cache_key:
        with st.spinner("Scanning for target leakage..."):
            report = detect_leaky_columns(df, target_column, problem_type)
        st.session_state[StateKeys.LEAKAGE_REPORT] = report
        st.session_state["_leakage_cache_key"] = cache_key
        # Default the exclusion list to whatever the scan flagged.
        st.session_state[StateKeys.EXCLUDED_COLUMNS] = summarize_leakage(report)

    report = st.session_state.get(StateKeys.LEAKAGE_REPORT)
    if report is None or report.empty:
        return

    flagged = summarize_leakage(report)
    if not flagged:
        st.success("✅ No target leakage detected -- no column reproduces the target on its own.")
        return

    st.warning(
        f"⚠️ **{len(flagged)} suspected data-leak column(s) detected.** These predict the target almost "
        "perfectly on their own, which usually means they are recorded *after* the outcome is known. "
        "Leaving them in produces unrealistically perfect scores. They are ticked for removal below."
    )

    previous = set(st.session_state.get(StateKeys.EXCLUDED_COLUMNS, []))
    chosen = []
    for _, row in report[report["is_suspected_leak"]].iterrows():
        column = row["column"]
        if st.checkbox(
            f"Exclude **{column}** -- {row['reason']}",
            value=column in previous,
            key=f"leak_exclude_{column}",
        ):
            chosen.append(column)

    # Changing the exclusion set invalidates any cached split/transform.
    if set(chosen) != previous:
        st.session_state[StateKeys.EXCLUDED_COLUMNS] = chosen
        st.session_state.pop(StateKeys.PREPROCESSING_BUNDLE, None)

    with st.expander("View full leakage scan"):
        st.caption(
            "Score = how accurately each column predicts the target *by itself* "
            "(correlation for regression). Columns at or above 0.98 are flagged."
        )
        st.dataframe(report, use_container_width=True)


def _run_automl(problem_type: str, target_column: str) -> None:
    """Run the full AutoML engine and persist results + artifacts to session_state."""
    overall_start = time.perf_counter()

    with st.status("Running AutoML training engine...", expanded=True) as status:
        if not _ensure_preprocessed(status):
            status.update(label="Preprocessing failed", state="error")
            st.error("Preprocessing failed -- check the dataset and target column.")
            return

        bundle = st.session_state[StateKeys.PREPROCESSING_BUNDLE]
        model_names = AUTOML_MODELS[problem_type]
        comparator = ModelComparator(problem_type)

        def _on_progress(message: str) -> None:
            status.write(f"✅ {message}")

        leaderboard = comparator.run(
            model_names,
            bundle["X_train"],
            bundle["y_train"],
            bundle["X_test"],
            bundle["y_test"],
            progress_callback=_on_progress,
        )

        best_model_name = comparator.get_best_model_name(leaderboard)
        best_reasons = comparator.get_best_model_reason(leaderboard)
        best_metrics = (
            leaderboard.loc[leaderboard["Rank"] == 1]
            .drop(columns=[c for c in _NON_METRIC_COLUMNS if c in leaderboard.columns])
            .iloc[0]
            .to_dict()
        )

        # Persist results under the same session-state keys the rest of the app
        # already reads (Model Comparison, Business Insights, Predictions, Reports).
        st.session_state[StateKeys.TRAINED_MODELS] = comparator.trained_models
        st.session_state[StateKeys.COMPARISON_RESULTS] = leaderboard
        st.session_state[StateKeys.BEST_MODEL_NAME] = best_model_name

        # Save every artifact required for later prediction -- no manual "save" step.
        artifact_bundle = build_artifact_bundle(
            model=comparator.get_model(best_model_name),
            cleaner=bundle["cleaner"],
            encoder=bundle["encoder"],
            scaler=bundle["scaler"],
            feature_columns=st.session_state.get(StateKeys.FEATURE_COLUMNS, []),
            numerical_columns=st.session_state.get(StateKeys.NUMERICAL_COLUMNS, []),
            categorical_columns=st.session_state.get(StateKeys.CATEGORICAL_COLUMNS, []),
            target_column=target_column,
            problem_type=problem_type,
            model_feature_columns=bundle["model_feature_columns"],
            target_encoder=bundle["target_encoder"],
            model_name=best_model_name,
            metrics=best_metrics,
            leaderboard=leaderboard,
            dataset_name=st.session_state.get(StateKeys.DATASET_NAME),
            trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        save_path = Path(MODELS_OUTPUT_DIR) / "best_model.joblib"
        save_artifact(artifact_bundle, save_path)
        st.session_state[StateKeys.ARTIFACT_BUNDLE] = artifact_bundle

        st.session_state[_TOTAL_TIME_KEY] = round(time.perf_counter() - overall_start, 2)
        st.session_state[_BEST_REASONS_KEY] = best_reasons

        status.update(label="AutoML training complete", state="complete")


def _render_results(df, problem_type: str) -> None:
    """Render the leaderboard, best-model explanation, and training summary."""
    leaderboard = st.session_state.get(StateKeys.COMPARISON_RESULTS)
    if leaderboard is None or leaderboard.empty:
        return

    best_model_name = st.session_state.get(StateKeys.BEST_MODEL_NAME)
    reasons = st.session_state.get(_BEST_REASONS_KEY, [])

    st.divider()
    st.subheader("Best Model")
    st.success(f"**{best_model_name}**")
    if reasons:
        st.markdown("**Reason:**\n" + "\n".join(f"- {reason}" for reason in reasons))

    st.divider()
    st.subheader("Model Comparison")

    def _highlight_best(row) -> list:
        return ["background-color: #d4f4dd" if row["Rank"] == 1 else "" for _ in row]

    st.dataframe(leaderboard.style.apply(_highlight_best, axis=1), use_container_width=True)

    st.divider()
    st.subheader("Training Summary")
    bundle = st.session_state.get(StateKeys.PREPROCESSING_BUNDLE, {}) or {}
    model_names = AUTOML_MODELS[problem_type]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Dataset", st.session_state.get(StateKeys.DATASET_NAME, "--"))
        st.metric("Rows", f"{df.shape[0]:,}")
    with c2:
        st.metric("Problem Type", PROBLEM_TYPE_LABELS.get(problem_type, problem_type))
        st.metric("Model Input Features", f"{len(bundle.get('model_feature_columns', []))}")
    with c3:
        st.metric("Best Model", best_model_name)
        st.metric("Total Training Time (s)", st.session_state.get(_TOTAL_TIME_KEY, "--"))

    st.caption(f"Models trained: {', '.join(model_names)}")

    dropped = bundle.get("dropped_columns") or []
    if dropped:
        st.caption(f"Excluded as suspected leakage: {', '.join(dropped)}")
    if bundle.get("rows_dropped_missing_target"):
        st.caption(f"Rows dropped for missing target: {bundle['rows_dropped_missing_target']:,}")

    save_path = Path(MODELS_OUTPUT_DIR) / "best_model.joblib"
    st.caption(f"Artifacts saved to `{save_path}`")


def render() -> None:
    """Render the Model Training (AutoML) page."""
    st.title("Model Training")
    st.caption(
        "An AutoML engine: automatically trains, evaluates, compares, and saves the "
        "best model for the detected problem type -- no manual model selection required."
    )

    df = st.session_state.get(StateKeys.RAW_DF)
    target_column = st.session_state.get(StateKeys.TARGET_COLUMN)
    if df is None or target_column is None:
        st.info("Upload a dataset and confirm the target column on **Dataset Explorer** first.")
        return

    problem_type = st.selectbox(
        "Problem type (auto-detected -- override only if it looks wrong)",
        options=list(PROBLEM_TYPE_LABELS.keys()),
        format_func=lambda k: PROBLEM_TYPE_LABELS[k],
        index=list(PROBLEM_TYPE_LABELS.keys()).index(
            st.session_state.get(StateKeys.PROBLEM_TYPE, "classification")
        ),
    )
    if problem_type != st.session_state.get(StateKeys.PROBLEM_TYPE):
        st.session_state[StateKeys.PROBLEM_TYPE] = problem_type
        st.session_state.pop(StateKeys.PREPROCESSING_BUNDLE, None)

    st.divider()
    _render_leakage_panel(df, target_column, problem_type)
    st.divider()

    model_names = AUTOML_MODELS[problem_type]
    st.markdown(f"**Models the AutoML engine will train:** {', '.join(model_names)}")

    if st.button("🚀 Run AutoML Training", type="primary", use_container_width=True):
        _run_automl(problem_type, target_column)

    _render_results(df, problem_type)
