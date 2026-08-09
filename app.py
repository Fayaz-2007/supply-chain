"""
Streamlit entry point for the Universal Business Analytics & ML Studio.

This file only wires together page routing and global page config -- all
actual page logic lives in ``dashboard/pages/*.py`` so each page stays
independently testable and reusable.
"""

from __future__ import annotations

import streamlit as st

from dashboard.components.sidebar import render_sidebar
from dashboard.pages import (
    ai_assistant,
    business_insights,
    dashboard_analytics,
    dataset_explorer,
    eda,
    home,
    model_comparison,
    model_training,
    predictions,
    reports,
)
from utils.config import CONFIG

st.set_page_config(page_title=CONFIG.app_title, page_icon="📊", layout="wide")

_PAGE_ROUTER = {
    "Home": home.render,
    "Dataset Explorer": dataset_explorer.render,
    "EDA": eda.render,
    "Model Training": model_training.render,
    "Model Comparison": model_comparison.render,
    "Predictions": predictions.render,
    "Business Insights": business_insights.render,
    "Dashboard Analytics": dashboard_analytics.render,
    "Reports": reports.render,
    "AI Assistant": ai_assistant.render,
}


def main() -> None:
    """Render the sidebar and dispatch to the selected page."""
    # A page's own custom nav buttons (e.g. Home's "Get Started") can't set the
    # sidebar radio's session_state key directly once that widget has rendered
    # this run -- Streamlit forbids mutating a widget's key after instantiation.
    # They stage the target page here instead; applying it *before* the radio is
    # created keeps the request one rerun away, in the one place that's legal.
    if "pending_nav" in st.session_state:
        st.session_state["nav_page"] = st.session_state.pop("pending_nav")

    selected_page = render_sidebar()
    _PAGE_ROUTER[selected_page]()


if __name__ == "__main__":
    main()
