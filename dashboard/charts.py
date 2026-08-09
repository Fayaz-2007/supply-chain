"""
Reusable Plotly chart builders shared by every dashboard page.

Every function takes a plain DataFrame/Series and column names -- nothing is
tied to a specific dataset, so the same chart function powers EDA, Business
Insights, and Dashboard Analytics pages alike.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

TEMPLATE = "plotly_white"
COLOR_SEQUENCE = px.colors.qualitative.Set2


def histogram(df: pd.DataFrame, column: str, bins: int = 30) -> go.Figure:
    """Histogram of a single numerical column."""
    fig = px.histogram(df, x=column, nbins=bins, template=TEMPLATE, color_discrete_sequence=COLOR_SEQUENCE)
    fig.update_layout(title=f"Distribution of {column}", bargap=0.05, height=380)
    return fig


def boxplot(df: pd.DataFrame, column: str, group_by: Optional[str] = None) -> go.Figure:
    """Boxplot of a numerical column, optionally grouped by a categorical column."""
    fig = px.box(df, x=group_by, y=column, template=TEMPLATE, color=group_by, color_discrete_sequence=COLOR_SEQUENCE)
    fig.update_layout(title=f"Boxplot of {column}", height=380)
    return fig


def correlation_heatmap(df: pd.DataFrame, columns: List[str]) -> go.Figure:
    """Correlation heatmap across the given numerical columns."""
    corr = df[columns].corr(numeric_only=True)
    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, template=TEMPLATE
    )
    fig.update_layout(title="Correlation Heatmap", height=500)
    return fig


def missing_value_chart(missing_summary: pd.DataFrame) -> go.Figure:
    """Bar chart of missing-value percentage per column (expects a summary df)."""
    nonzero = missing_summary[missing_summary["missing_count"] > 0]
    if nonzero.empty:
        fig = go.Figure()
        fig.update_layout(title="No Missing Values Detected", height=300, template=TEMPLATE)
        return fig
    fig = px.bar(
        nonzero, x="column", y="missing_percent", template=TEMPLATE, color_discrete_sequence=COLOR_SEQUENCE
    )
    fig.update_layout(title="Missing Values by Column (%)", height=380, xaxis_tickangle=-40)
    return fig


def categorical_bar_chart(df: pd.DataFrame, column: str, top_n: int = 20) -> go.Figure:
    """Bar chart of value counts for a categorical column."""
    counts = df[column].value_counts().head(top_n).reset_index()
    counts.columns = [column, "count"]
    fig = px.bar(counts, x=column, y="count", template=TEMPLATE, color_discrete_sequence=COLOR_SEQUENCE)
    fig.update_layout(title=f"Value Counts: {column}", height=380, xaxis_tickangle=-40)
    return fig


def pie_chart(df: pd.DataFrame, column: str, top_n: int = 10) -> go.Figure:
    """Pie chart of value proportions for a categorical column."""
    counts = df[column].value_counts().head(top_n).reset_index()
    counts.columns = [column, "count"]
    fig = px.pie(counts, names=column, values="count", template=TEMPLATE, color_discrete_sequence=COLOR_SEQUENCE)
    fig.update_layout(title=f"Share by {column}", height=400)
    return fig


def scatter_plot(df: pd.DataFrame, x: str, y: str, color: Optional[str] = None) -> go.Figure:
    """Scatter plot between two numerical columns, optionally colored by a third."""
    fig = px.scatter(df, x=x, y=y, color=color, template=TEMPLATE, color_discrete_sequence=COLOR_SEQUENCE)
    fig.update_layout(title=f"{y} vs {x}", height=420)
    return fig


def trend_line_chart(df: pd.DataFrame, x: str, y: str, color: Optional[str] = None) -> go.Figure:
    """Line chart for trend analysis, typically x=date column, y=numerical column."""
    fig = px.line(df, x=x, y=y, color=color, template=TEMPLATE, color_discrete_sequence=COLOR_SEQUENCE)
    fig.update_layout(title=f"Trend of {y} over {x}", height=420)
    return fig


def feature_importance_chart(feature_names: List[str], importances: List[float], top_n: int = 15) -> go.Figure:
    """Horizontal bar chart of feature importances, sorted descending."""
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=True).tail(top_n)
    fig = px.bar(
        imp_df, x="importance", y="feature", orientation="h", template=TEMPLATE,
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    fig.update_layout(title="Top Feature Importances", height=max(380, top_n * 24))
    return fig


def model_comparison_bar(leaderboard: pd.DataFrame, metric: str) -> go.Figure:
    """Bar chart comparing all trained models on a single metric."""
    fig = px.bar(
        leaderboard, x="Model", y=metric, template=TEMPLATE, color="Model",
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    fig.update_layout(title=f"Model Comparison: {metric}", height=400, showlegend=False)
    return fig


def category_ranking_chart(
    df: pd.DataFrame, category_col: str, value_col: str, agg: str = "mean", top_n: int = 15
) -> go.Figure:
    """Horizontal bar chart ranking an aggregated numeric value by category.

    Generic building block for any "top categories by outcome" business view
    (e.g. average predicted value per category, or the rate of a given
    predicted class per category) -- the caller decides what `value_col` and
    `agg` represent for their dataset.
    """
    grouped = (
        df.groupby(category_col)[value_col]
        .agg(agg)
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )
    fig = px.bar(
        grouped, x=value_col, y=category_col, orientation="h", template=TEMPLATE,
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    fig.update_layout(
        title=f"{category_col} Ranked by {value_col} ({agg})",
        height=max(380, len(grouped) * 32),
        yaxis={"categoryorder": "total ascending"},
    )
    return fig


def generic_heatmap(pivot_df: pd.DataFrame) -> go.Figure:
    """Heatmap from an already-pivoted DataFrame (index/columns/values set by caller)."""
    fig = px.imshow(pivot_df, text_auto=True, color_continuous_scale="Blues", template=TEMPLATE)
    fig.update_layout(title="Heatmap", height=450)
    return fig
