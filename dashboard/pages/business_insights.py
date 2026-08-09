"""
Business Insights page: the Executive Business Dashboard.

A business-user-facing view of the trained model and its predictions -- KPI
cards, a compact model information panel, a dynamic executive summary, and a
set of interactive business visualizations. Everything is derived generically
from the active artifact bundle and the most recent predictions; no dataset-
specific column names are ever assumed.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.charts import (
    categorical_bar_chart,
    category_ranking_chart,
    correlation_heatmap,
    feature_importance_chart,
    histogram,
    model_comparison_bar,
    pie_chart,
)
from dashboard.components.kpi import render_kpi_row
from utils.config import StateKeys
from utils.helpers import get_active_artifact_bundle

# Mirrors the ranking metric compare_models.py uses per problem type -- duplicated
# here only as a tiny naming lookup (not comparison logic) so this page can label
# "Evaluation Metric" without importing a private symbol from another module.
_PRIMARY_METRIC_BY_PROBLEM = {"classification": "F1", "regression": "RMSE", "time_series": "RMSE"}


def _format_timestamp(value: Optional[str]) -> str:
    """Render an ISO-8601 timestamp as a short, human-readable string."""
    if not value:
        return "--"
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        return value


def _get_feature_importance(model: Any, feature_names: list) -> pd.DataFrame:
    """Extract feature importances (tree models) or absolute coefficients (linear models)."""
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(np.ravel(model.coef_))
    else:
        return pd.DataFrame()

    n = min(len(feature_names), len(values))
    return pd.DataFrame({"feature": feature_names[:n], "importance": values[:n]}).sort_values(
        "importance", ascending=False
    )


def _build_kpis(
    bundle: dict, predictions_df: Optional[pd.DataFrame], problem_type: str, target_column: str
) -> List[Tuple[str, str, Optional[str]]]:
    """Build the executive KPI row, adapting to whatever predictions/metadata exist."""
    kpis: List[Tuple[str, str, Optional[str]]] = []

    total_records = len(predictions_df) if predictions_df is not None else None
    kpis.append(("Total Records Analysed", f"{total_records:,}" if total_records is not None else "--", None))
    kpis.append(("Dataset Used", bundle.get("dataset_name") or "--", None))
    kpis.append(("Best Performing Model", bundle.get("model_name") or "--", None))

    if predictions_df is not None and "Prediction_Confidence" in predictions_df.columns:
        kpis.append(("Prediction Confidence", f"{predictions_df['Prediction_Confidence'].mean():.1%}", None))
    else:
        kpis.append(("Prediction Confidence", "N/A", None))

    if predictions_df is not None:
        if problem_type == "classification":
            label_col = "Prediction_Label" if "Prediction_Label" in predictions_df.columns else "Prediction"
            counts = predictions_df[label_col].value_counts()
            for class_name, count in counts.head(2).items():
                pct = count / len(predictions_df) * 100
                kpis.append((f"Predicted: {class_name}", f"{count:,} ({pct:.1f}%)", None))
        else:
            avg_val = predictions_df["Prediction"].mean()
            kpis.append((f"Average Predicted {target_column}", f"{avg_val:,.2f}", None))

    kpis.append(("Model Type", problem_type.title(), None))
    kpis.append(("Last Training Time", _format_timestamp(bundle.get("trained_at")), None))
    return kpis


def _render_model_info_panel(bundle: dict) -> None:
    """Compact panel of model metadata, read directly from the saved artifact bundle."""
    metrics = bundle.get("metrics") or {}
    problem_type = bundle.get("problem_type", "classification")
    primary_metric = _PRIMARY_METRIC_BY_PROBLEM.get(problem_type)
    best_score = metrics.get(primary_metric) if primary_metric else None

    info_items: List[Tuple[str, str, Optional[str]]] = [
        ("Best Model", bundle.get("model_name") or "--", None),
        ("Problem Type", problem_type.title(), None),
        ("Training Date", _format_timestamp(bundle.get("trained_at")), None),
        ("Dataset Used", bundle.get("dataset_name") or "--", None),
        ("Number of Features", f"{len(bundle.get('model_feature_columns') or [])}", None),
        ("Evaluation Metric", primary_metric or "--", None),
        ("Best Score", f"{best_score}" if best_score is not None else "--", None),
    ]
    render_kpi_row(info_items[:4])
    render_kpi_row(info_items[4:])


def _build_executive_summary(
    bundle: dict,
    predictions_df: Optional[pd.DataFrame],
    raw_df: Optional[pd.DataFrame],
    problem_type: str,
    importance_df: pd.DataFrame,
) -> str:
    """Dynamic, plain-language summary -- wording adapts to the actual results, not a template fill-in."""
    lines: List[str] = []
    dataset_name = bundle.get("dataset_name") or "the dataset"
    target_column = bundle.get("target_column", "")
    # Only report a row count if the dataset currently in session actually looks like
    # the one the model was trained on (same target column present) -- otherwise the
    # user has a different/unrelated dataset loaded and the count would be misleading.
    raw_df_matches_model = raw_df is not None and target_column in raw_df.columns
    row_note = f" ({len(raw_df):,} rows)" if raw_df_matches_model else ""
    lines.append(f"- **Dataset analysed successfully**: {dataset_name}{row_note}.")

    if predictions_df is not None:
        lines.append(f"- **Total predictions generated**: {len(predictions_df):,} rows scored.")
    else:
        lines.append("- **No predictions generated yet** -- upload data on the Predictions page to see results here.")

    if predictions_df is not None and problem_type == "classification":
        label_col = "Prediction_Label" if "Prediction_Label" in predictions_df.columns else "Prediction"
        top_class = predictions_df[label_col].value_counts().idxmax()
        lines.append(f"- **Highest-frequency predicted category**: '{top_class}'.")

    if not importance_df.empty:
        top_features = importance_df["feature"].head(3).tolist()
        lines.append(f"- **Major contributing business factors**: {', '.join(top_features)}.")

    if predictions_df is not None and "Prediction_Confidence" in predictions_df.columns:
        avg_conf = predictions_df["Prediction_Confidence"].mean()
        lines.append(f"- **Overall prediction confidence**: {avg_conf:.1%}.")

    lines.append(f"- **Model used**: {bundle.get('model_name') or 'N/A'}.")

    if predictions_df is None:
        action = "Upload new data on the Predictions page to generate forecasts or classifications."
    elif problem_type == "classification" and "Prediction_Confidence" in predictions_df.columns \
            and predictions_df["Prediction_Confidence"].mean() < 0.7:
        action = "Confidence is moderate -- consider manual review of low-confidence predictions before acting on them."
    elif problem_type == "classification":
        action = "Prioritize review or outreach for records predicted in the higher-risk / minority class(es)."
    else:
        target_column = bundle.get("target_column", "the target")
        action = f"Monitor forecasted {target_column} values for outliers and retrain if incoming data shifts materially."
    lines.append(f"- **Suggested next action**: {action}")

    return "\n".join(lines)


def _render_visualizations(
    bundle: dict,
    predictions_df: Optional[pd.DataFrame],
    raw_df: Optional[pd.DataFrame],
    problem_type: str,
    target_column: str,
    importance_df: pd.DataFrame,
) -> None:
    """Interactive, tabbed business visualizations built from existing chart functions."""
    tabs = st.tabs(
        ["Predictions", "Feature Importance", "Training Data", "Category Analysis", "Model Performance"]
    )

    with tabs[0]:
        if predictions_df is None:
            st.info("Generate predictions on the **Predictions** page to see this chart.")
        elif problem_type == "classification":
            label_col = "Prediction_Label" if "Prediction_Label" in predictions_df.columns else "Prediction"
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(categorical_bar_chart(predictions_df, label_col), use_container_width=True, key="exec_pred_bar")
            with c2:
                st.plotly_chart(pie_chart(predictions_df, label_col), use_container_width=True, key="exec_pred_pie")
        else:
            st.plotly_chart(histogram(predictions_df, "Prediction"), use_container_width=True, key="exec_pred_hist")

    with tabs[1]:
        if importance_df.empty:
            st.info("This model type does not expose feature importances or coefficients.")
        else:
            st.plotly_chart(
                feature_importance_chart(importance_df["feature"].tolist(), importance_df["importance"].tolist()),
                use_container_width=True, key="exec_feat_imp",
            )
            st.dataframe(importance_df.head(10), use_container_width=True)

    with tabs[2]:
        if raw_df is None or target_column not in raw_df.columns:
            st.info("Upload/select a dataset on **Dataset Explorer** to see target distribution and correlations.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Target Distribution**")
                if problem_type == "classification":
                    st.plotly_chart(categorical_bar_chart(raw_df, target_column), use_container_width=True, key="exec_target_dist")
                else:
                    st.plotly_chart(histogram(raw_df, target_column), use_container_width=True, key="exec_target_dist")
            with c2:
                st.markdown("**Correlation Heatmap**")
                numeric_cols = raw_df.select_dtypes(include="number").columns.tolist()
                if len(numeric_cols) >= 2:
                    st.plotly_chart(correlation_heatmap(raw_df, numeric_cols), use_container_width=True, key="exec_corr_heatmap")
                else:
                    st.info("Not enough numerical columns for a correlation heatmap.")

    with tabs[3]:
        if predictions_df is None:
            st.info(
                "Generate predictions on the **Predictions** page to unlock category-level rankings "
                "(e.g. top categories by predicted outcome)."
            )
        else:
            categorical_columns = [c for c in bundle.get("categorical_columns", []) if c in predictions_df.columns]
            if not categorical_columns:
                st.info("No categorical columns available for category-level analysis.")
            else:
                category_col = st.selectbox("Group by category", categorical_columns, key="exec_category_col")
                if problem_type == "classification":
                    label_col = "Prediction_Label" if "Prediction_Label" in predictions_df.columns else "Prediction"
                    class_options = sorted(predictions_df[label_col].astype(str).unique().tolist())
                    focus_class = st.selectbox("Rank categories by rate of predicted class", class_options, key="exec_focus_class")
                    rate_col = f"Rate: {focus_class}"
                    rate_df = predictions_df.copy()
                    rate_df[rate_col] = (rate_df[label_col].astype(str) == focus_class).astype(float)
                    st.plotly_chart(
                        category_ranking_chart(rate_df, category_col, rate_col, agg="mean"),
                        use_container_width=True, key="exec_category_ranking",
                    )
                    st.caption(f"Share of records predicted as '{focus_class}', by {category_col}.")
                else:
                    st.plotly_chart(
                        category_ranking_chart(predictions_df, category_col, "Prediction", agg="mean"),
                        use_container_width=True, key="exec_category_ranking",
                    )
                    st.caption(f"Average predicted {target_column}, by {category_col}.")

    with tabs[4]:
        leaderboard = bundle.get("leaderboard")
        if leaderboard is None or getattr(leaderboard, "empty", True):
            st.info("No model comparison data was saved with this model.")
        else:
            primary_metric = _PRIMARY_METRIC_BY_PROBLEM.get(problem_type)
            if primary_metric and primary_metric in leaderboard.columns:
                st.plotly_chart(model_comparison_bar(leaderboard, primary_metric), use_container_width=True, key="exec_model_perf")
            st.dataframe(leaderboard, use_container_width=True)


def render() -> None:
    """Render the Executive Business Dashboard."""
    st.title("Executive Business Dashboard")
    st.caption("A business-friendly view of the trained model and its predictions -- no ML jargon required.")

    artifact_bundle = get_active_artifact_bundle(st.session_state)
    if artifact_bundle is None:
        st.info("No trained model found yet. Train a model on the **Model Training** page first.")
        return

    predictions_df = st.session_state.get(StateKeys.PREDICTIONS)
    raw_df = st.session_state.get(StateKeys.RAW_DF)
    problem_type = artifact_bundle.get("problem_type", "classification")
    target_column = artifact_bundle.get("target_column", "")

    model = artifact_bundle.get("model")
    feature_names = artifact_bundle.get("model_feature_columns") or []
    importance_df = _get_feature_importance(model, feature_names)

    st.subheader("Key Performance Indicators")
    kpis = _build_kpis(artifact_bundle, predictions_df, problem_type, target_column)
    render_kpi_row(kpis[:4])
    if len(kpis) > 4:
        render_kpi_row(kpis[4:8])

    st.divider()
    st.subheader("Model Information")
    _render_model_info_panel(artifact_bundle)

    st.divider()
    st.subheader("Executive Summary")
    st.markdown(_build_executive_summary(artifact_bundle, predictions_df, raw_df, problem_type, importance_df))

    st.divider()
    st.subheader("Business Visualizations")
    _render_visualizations(artifact_bundle, predictions_df, raw_df, problem_type, target_column, importance_df)
