"""
Home page: an enterprise-SaaS-style landing page for the platform.

Marketing sections (hero illustration, tech stack, feature cards, workflow,
stats, comparison, CTA banner, footer) render as small HTML/CSS/JS fragments
via ``streamlit.components.v1.html`` -- see ``dashboard/components/landing.py``.
Navigation stays 100% native Streamlit: every call-to-action is a real
``st.button`` (CSS-styled, scoped to this page only) that sets the sidebar's
nav selection and reruns, so page routing never depends on fragile
iframe-to-parent messaging.
"""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

from dashboard.components import landing
from models.train import AUTOML_MODELS
from utils.config import CONFIG, StateKeys
from utils.helpers import list_sample_datasets, load_sample_dataset

# Session-state keys cleared when a fresh dataset is loaded, mirroring
# dataset_explorer.py's own reset list so "View Demo" behaves identically to a
# manual sample-dataset load there.
_DOWNSTREAM_KEYS = (
    StateKeys.TRAINED_MODELS,
    StateKeys.COMPARISON_RESULTS,
    StateKeys.BEST_MODEL_NAME,
    StateKeys.ARTIFACT_BUNDLE,
    StateKeys.PREDICTIONS,
    StateKeys.PREPROCESSING_BUNDLE,
)

# Buttons here are scoped to these two container keys only (via the
# ``.st-key-<key>`` class Streamlit attaches), so this styling never leaks
# onto buttons on any other page.
#
# Buttons are deliberately NOT stretched to fill their column (no
# use_container_width) -- content-sized pill buttons read as a considered,
# tight CTA cluster; stretching them to an arbitrary column width produced
# oddly wide buttons with a large empty trailing gap (verified in-browser).
_BUTTON_STYLE = """
<style>
.st-key-home_hero_actions div[data-testid="stButton"] button,
.st-key-home_banner_actions div[data-testid="stButton"] button{
  border-radius:999px !important;
  padding:0.65rem 1.8rem !important;
  font-weight:700 !important;
  font-size:0.95rem !important;
  border:none !important;
  width:auto !important;
  transition:transform .25s cubic-bezier(.16,.8,.24,1), box-shadow .25s ease !important;
}
.st-key-home_hero_actions button[kind="primary"],
.st-key-home_banner_actions button[kind="primary"]{
  background:linear-gradient(135deg,#2563eb,#10b981) !important;
  color:#fff !important;
  box-shadow:0 10px 24px -8px rgba(37,99,235,.5) !important;
}
.st-key-home_hero_actions button[kind="primary"]:hover,
.st-key-home_banner_actions button[kind="primary"]:hover{
  transform:translateY(-3px) scale(1.02);
  box-shadow:0 16px 32px -8px rgba(16,185,129,.55) !important;
}
.st-key-home_hero_actions button[kind="secondary"],
.st-key-home_banner_actions button[kind="secondary"]{
  background:transparent !important;
  color:#2563eb !important;
  border:1.5px solid #2563eb !important;
}
.st-key-home_hero_actions button[kind="secondary"]:hover,
.st-key-home_banner_actions button[kind="secondary"]:hover{
  background:rgba(37,99,235,.08) !important;
  transform:translateY(-3px);
}
/* Hero CTAs sit left, matching the left-aligned hero heading above them. */
.st-key-home_hero_actions div[data-testid="stHorizontalBlock"]{
  max-width:1180px; margin:0 auto; padding:0 24px; gap:0.75rem !important;
}
.st-key-home_hero_actions div[data-testid="stColumn"]{ width:auto !important; flex:0 0 auto !important; }
/* Banner CTAs sit centered, matching the centered banner heading above them. */
.st-key-home_banner_actions div[data-testid="stHorizontalBlock"]{
  justify-content:center; gap:0.75rem !important;
}
.st-key-home_banner_actions div[data-testid="stColumn"]{ width:auto !important; flex:0 0 auto !important; }
</style>
"""


def _go_to(page_name: str) -> None:
    """Navigate to another page.

    Streamlit forbids writing to a widget's session_state key after that widget
    has already been instantiated in the current run -- and the sidebar's nav
    radio always renders before this page's body does. So this stages the
    target page under a plain (non-widget) key; app.py applies it to the
    radio's key on the next run, before the radio is created.
    """
    st.session_state["pending_nav"] = page_name
    st.rerun()


def _load_demo_dataset() -> bool:
    """Load the first bundled sample dataset, resetting downstream state. Returns success."""
    samples = list_sample_datasets()
    if not samples:
        return False
    df, name = load_sample_dataset(samples[0])
    st.session_state[StateKeys.RAW_DF] = df
    st.session_state[StateKeys.DATASET_NAME] = name
    st.session_state[StateKeys.DATASET_SOURCE] = "sample"
    for key in _DOWNSTREAM_KEYS:
        st.session_state.pop(key, None)
    return True


def _render_hero_actions() -> None:
    with st.container(key="home_hero_actions"):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Get Started", type="primary"):
                _go_to("Dataset Explorer")
        with col2:
            if st.button("View Demo", type="secondary"):
                if _load_demo_dataset():
                    _go_to("Dataset Explorer")
                else:
                    st.warning("No bundled sample dataset available -- upload your own on Dataset Explorer.")


def _render_banner_actions() -> None:
    with st.container(key="home_banner_actions"):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Now", type="primary"):
                _go_to("Dataset Explorer")
        with col2:
            if st.button("Explore Platform", type="secondary"):
                _go_to("Business Insights")


def _build_stats() -> list:
    """Assemble the four headline stats, using live session data where available."""
    raw_df = st.session_state.get(StateKeys.RAW_DF)
    if raw_df is not None:
        records = (len(raw_df), "", "")
    else:
        records = (200_000, "+", "")

    n_models = len(AUTOML_MODELS.get("classification", []))

    return [
        ("Records Processed", records[0], records[1], records[2]),
        ("Machine Learning Models", n_models, "", ""),
        ("Automated Pipeline", 100, "%", ""),
        ("Ready Architecture", "Enterprise", "", ""),
    ]


def render() -> None:
    """Render the Home landing page."""
    st.markdown(_BUTTON_STYLE, unsafe_allow_html=True)

    components.html(
        landing.render_hero(
            title="AI-Powered Supply Chain Decision Intelligence Platform",
            subtitle=(
                "Transform raw supply chain data into intelligent business decisions "
                "using AutoML, Predictive Analytics, and Generative AI."
            ),
        ),
        height=470,
        scrolling=False,
    )
    _render_hero_actions()

    components.html(landing.render_tech_stack(), height=1420, scrolling=False)
    components.html(landing.render_features(), height=1090, scrolling=False)
    components.html(landing.render_how_it_works(), height=410, scrolling=False)
    components.html(landing.render_stats(_build_stats()), height=430, scrolling=False)
    components.html(landing.render_comparison(), height=580, scrolling=False)

    components.html(
        landing.render_cta_banner("Ready to Transform Your Supply Chain Decisions?"),
        height=195,
        scrolling=False,
    )
    _render_banner_actions()

    components.html(landing.render_footer(CONFIG.app_title), height=305, scrolling=False)
