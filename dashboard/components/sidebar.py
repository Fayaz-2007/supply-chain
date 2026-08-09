"""
Sidebar navigation + global status indicators shown on every page.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

from dashboard.components.kpi import status_badge
from utils.config import CONFIG, StateKeys

PAGES = [
    "Home",
    "Dataset Explorer",
    "EDA",
    "Model Training",
    "Model Comparison",
    "Predictions",
    "Business Insights",
    "Dashboard Analytics",
    "Reports",
    "AI Assistant",
]


def render_sidebar() -> str:
    """Render the sidebar navigation and status panel.

    Returns:
        The name of the currently selected page.
    """
    with st.sidebar:
        st.title(CONFIG.app_title)
        st.caption("Plug in any tabular dataset. Everything below adapts automatically.")

        selected_page = st.radio("Navigate", PAGES, key="nav_page")

        st.divider()
        st.subheader("Status")
        st.markdown(status_badge("Dataset Loaded", st.session_state.get(StateKeys.RAW_DF) is not None))
        model_ready = bool(st.session_state.get(StateKeys.TRAINED_MODELS)) or (
            st.session_state.get(StateKeys.ARTIFACT_BUNDLE) is not None
        )
        st.markdown(status_badge("Model Trained", model_ready))
        st.markdown(status_badge("Predictions Ready", st.session_state.get(StateKeys.PREDICTIONS) is not None))

        dataset_name = st.session_state.get(StateKeys.DATASET_NAME)
        if dataset_name:
            st.caption(f"Active dataset: **{dataset_name}**")

    return selected_page
