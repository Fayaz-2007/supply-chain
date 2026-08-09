"""
Dashboard Analytics page: interactive, filterable charts over the active dataset.
"""

from __future__ import annotations

import streamlit as st

from dashboard.charts import (
    generic_heatmap,
    pie_chart,
    scatter_plot,
    trend_line_chart,
)
from utils.config import CONFIG, StateKeys


def render() -> None:
    """Render the Dashboard Analytics page."""
    st.title("Dashboard Analytics")

    df = st.session_state.get(StateKeys.RAW_DF)
    if df is None:
        st.info("Upload a dataset on the **Dataset Explorer** page first.")
        return

    numerical_columns = st.session_state.get(StateKeys.NUMERICAL_COLUMNS, [])
    categorical_columns = st.session_state.get(StateKeys.CATEGORICAL_COLUMNS, [])
    date_column = st.session_state.get(StateKeys.DATE_COLUMN)

    st.subheader("Filters")
    filtered_df = df.copy()
    filter_cols = st.columns(min(3, max(len(categorical_columns), 1)))
    for i, col in enumerate(categorical_columns[:3]):
        options = ["(all)"] + sorted(df[col].dropna().unique().tolist())[: CONFIG.max_categorical_chart_levels]
        choice = filter_cols[i % len(filter_cols)].selectbox(f"Filter by {col}", options, key=f"filter_{col}")
        if choice != "(all)":
            filtered_df = filtered_df[filtered_df[col] == choice]

    st.caption(f"Showing {len(filtered_df):,} of {len(df):,} rows after filtering.")

    tab1, tab2, tab3, tab4 = st.tabs(["Trend", "Scatter", "Pie / Bar", "Heatmap"])

    with tab1:
        if date_column and numerical_columns:
            y_col = st.selectbox("Metric", numerical_columns, key="trend_y")
            color_col = st.selectbox("Split by (optional)", ["(none)"] + categorical_columns, key="trend_color")
            color_col = None if color_col == "(none)" else color_col
            st.plotly_chart(trend_line_chart(filtered_df.sort_values(date_column), date_column, y_col, color_col), use_container_width=True)
        else:
            st.info("A date column and at least one numerical column are needed for trend charts.")

    with tab2:
        if len(numerical_columns) >= 2:
            x_col = st.selectbox("X axis", numerical_columns, key="scatter_x")
            y_col = st.selectbox("Y axis", numerical_columns, index=min(1, len(numerical_columns) - 1), key="scatter_y")
            color_col = st.selectbox("Color by (optional)", ["(none)"] + categorical_columns, key="scatter_color")
            color_col = None if color_col == "(none)" else color_col
            st.plotly_chart(scatter_plot(filtered_df, x_col, y_col, color_col), use_container_width=True)
        else:
            st.info("Need at least two numerical columns for a scatter plot.")

    with tab3:
        if categorical_columns:
            cat_col = st.selectbox("Category column", categorical_columns, key="pie_col")
            st.plotly_chart(pie_chart(filtered_df, cat_col), use_container_width=True)
        else:
            st.info("No categorical columns available.")

    with tab4:
        if len(categorical_columns) >= 2 and numerical_columns:
            row_col = st.selectbox("Rows", categorical_columns, key="heat_row")
            col_col = st.selectbox("Columns", [c for c in categorical_columns if c != row_col], key="heat_col")
            value_col = st.selectbox("Values (aggregated by mean)", numerical_columns, key="heat_val")
            pivot = filtered_df.pivot_table(index=row_col, columns=col_col, values=value_col, aggfunc="mean")
            st.plotly_chart(generic_heatmap(pivot), use_container_width=True)
        else:
            st.info("Need at least two categorical columns and one numerical column for a heatmap.")
