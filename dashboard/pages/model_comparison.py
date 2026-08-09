"""
Model Comparison page: ranked leaderboard with the best model highlighted.
"""

from __future__ import annotations

import streamlit as st

from dashboard.charts import model_comparison_bar
from utils.config import StateKeys

_METRIC_BY_PROBLEM = {
    "classification": "F1",
    "regression": "RMSE",
    "time_series": "RMSE",
}


def render() -> None:
    """Render the Model Comparison page."""
    st.title("Model Comparison")

    leaderboard = st.session_state.get(StateKeys.COMPARISON_RESULTS)
    if leaderboard is None or leaderboard.empty:
        st.info("Train models on the **Model Training** page first.")
        return

    best_model_name = st.session_state.get(StateKeys.BEST_MODEL_NAME)
    st.success(f"Best model: **{best_model_name}**")

    def _highlight_best(row: "st.column_config") -> list:
        return ["background-color: #d4f4dd" if row["Rank"] == 1 else "" for _ in row]

    st.dataframe(leaderboard.style.apply(_highlight_best, axis=1), use_container_width=True)

    problem_type = st.session_state.get(StateKeys.PROBLEM_TYPE, "classification")
    metric = _METRIC_BY_PROBLEM.get(problem_type, "F1")
    if metric in leaderboard.columns:
        st.plotly_chart(model_comparison_bar(leaderboard, metric), use_container_width=True, key="primary_metric_chart")

    other_metrics = [c for c in leaderboard.columns if c not in ("Rank", "Model", "Training Time (s)")]
    metric_choice = st.selectbox("Compare on a different metric", other_metrics, index=other_metrics.index(metric) if metric in other_metrics else 0)
    if metric_choice != metric:
        st.plotly_chart(model_comparison_bar(leaderboard, metric_choice), use_container_width=True, key="secondary_metric_chart")
