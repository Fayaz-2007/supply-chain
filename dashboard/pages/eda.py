"""
EDA page: automatically generated exploratory charts for any dataset.
"""

from __future__ import annotations

import streamlit as st

from dashboard.charts import (
    boxplot,
    categorical_bar_chart,
    correlation_heatmap,
    histogram,
    missing_value_chart,
)
from preprocessing.clean import get_missing_value_summary
from utils.config import CONFIG, StateKeys


def render() -> None:
    """Render the EDA page."""
    st.title("Exploratory Data Analysis")

    df = st.session_state.get(StateKeys.RAW_DF)
    if df is None:
        st.info("Upload a dataset on the **Dataset Explorer** page first.")
        return

    numerical_columns = st.session_state.get(StateKeys.NUMERICAL_COLUMNS, [])
    categorical_columns = st.session_state.get(StateKeys.CATEGORICAL_COLUMNS, [])

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Numerical Distributions", "Categorical Charts", "Correlation", "Missing Values"]
    )

    with tab1:
        if not numerical_columns:
            st.info("No numerical columns detected.")
        else:
            col = st.selectbox("Select numerical column", numerical_columns, key="eda_num_col")
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(histogram(df, col), use_container_width=True)
            with c2:
                group_by = st.selectbox(
                    "Group boxplot by (optional)",
                    ["(none)"] + categorical_columns[: CONFIG.max_categorical_chart_levels],
                    key="eda_box_group",
                )
                group_by = None if group_by == "(none)" else group_by
                st.plotly_chart(boxplot(df, col, group_by), use_container_width=True)

            with st.expander("View all numerical distributions"):
                for other_col in numerical_columns:
                    st.plotly_chart(
                        histogram(df, other_col),
                        use_container_width=True,
                        key=f"eda_hist_all_{other_col}",
                    )

    with tab2:
        if not categorical_columns:
            st.info("No categorical columns detected.")
        else:
            cat_col = st.selectbox("Select categorical column", categorical_columns, key="eda_cat_col")
            st.plotly_chart(categorical_bar_chart(df, cat_col), use_container_width=True)

    with tab3:
        if len(numerical_columns) < 2:
            st.info("Need at least two numerical columns for a correlation heatmap.")
        else:
            st.plotly_chart(correlation_heatmap(df, numerical_columns), use_container_width=True)

    with tab4:
        missing_summary = get_missing_value_summary(df)
        st.plotly_chart(missing_value_chart(missing_summary), use_container_width=True)
