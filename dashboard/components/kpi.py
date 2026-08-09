"""
Reusable KPI/metric card components for Streamlit pages.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import streamlit as st


def render_kpi_row(kpis: List[Tuple[str, str, Optional[str]]]) -> None:
    """Render a row of Streamlit metric cards.

    Args:
        kpis: List of (label, value, delta) tuples. delta may be None.
    """
    columns = st.columns(len(kpis)) if kpis else []
    for col, (label, value, delta) in zip(columns, kpis):
        with col:
            st.metric(label=label, value=value, delta=delta)


def status_badge(label: str, is_ready: bool) -> str:
    """Return a markdown status line with a colored dot for readiness checks.

    Uses raw emoji characters (not ``:shortcode:`` syntax) since Streamlit's
    bundled markdown emoji dictionary doesn't recognize newer shortcodes like
    ``:green_circle:``, which would otherwise render as literal text.
    """
    dot = "\U0001F7E2" if is_ready else "\U0001F534"  # green circle / red circle
    state = "Ready" if is_ready else "Not Ready"
    return f"{dot} **{label}:** {state}"
