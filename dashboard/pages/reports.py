"""
Reports page: export the dataset, leaderboard, and predictions as CSV/Excel/PDF.

Performance note: ``st.download_button``'s ``data=`` argument is evaluated
eagerly the instant the widget is built -- Streamlit has no lazy mode for it.
Measured on the 200K-row demo dataset, Excel serialization alone took ~133s
(CSV ~3s). Rendering the old page (which called this unconditionally for the
dataset, leaderboard, *and* predictions on every single rerun -- i.e. on any
button click anywhere on the page) could therefore block for minutes on a
single click. Every export below is now opt-in (computed only after an
explicit "Prepare Export" click) and cached, so the page itself always
renders instantly and heavy work happens at most once per dataset.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import streamlit as st

from dashboard.components.download import (
    BusinessReportData,
    ChartImage,
    ReportSection,
    build_bar_chart_png,
    build_business_report_pdf,
    build_histogram_png,
)
from services.ai_context import build_full_context, extract_feature_importance
from services.llm_service import get_llm_service
from services.prompt_templates import SYSTEM_PROMPT, build_report_narrative_prompt
from utils.config import StateKeys
from utils.helpers import (
    dataframe_to_csv_bytes,
    dataframe_to_excel_bytes,
    generate_run_id,
    get_active_artifact_bundle,
    get_memory_usage_mb,
)

_NARRATIVE_HEADERS = [
    "Executive Summary",
    "Key Insights",
    "Business Recommendations",
    "Risk Assessment",
    "Suggested Next Steps",
]

# Mirrors the ranking metric compare_models.py uses per problem type -- duplicated
# here only as a tiny naming lookup (not comparison logic), the same convention
# already used in ai_assistant.py/business_insights.py, so the PDF's leaderboard
# chart highlights the same metric the app ranks models on.
_PRIMARY_METRIC_BY_PROBLEM = {"classification": "F1", "regression": "RMSE", "time_series": "RMSE"}

# Session-state keys local to this page (page-scoped UI state, not shared with
# other pages, so kept out of the central StateKeys registry -- same convention
# used by model_training.py's own local keys).
_PDF_BYTES_KEY = "reports_pdf_bytes"
_PDF_FILENAME_KEY = "reports_pdf_file_name"


# --------------------------------------------------------------------------- #
# Cached, opt-in exports
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _cached_csv_bytes(df: pd.DataFrame, _cache_tag: str) -> bytes:
    """Cache key includes ``_cache_tag`` (the export's file stem) purely for
    readability/debugging -- st.cache_data already hashes ``df`` itself, so a
    given dataframe's bytes are computed at most once per Streamlit session."""
    return dataframe_to_csv_bytes(df)


@st.cache_data(show_spinner=False)
def _cached_excel_bytes(df: pd.DataFrame, _cache_tag: str) -> bytes:
    return dataframe_to_excel_bytes(df)


def _render_export_section(title: str, data: Optional[pd.DataFrame], file_stem: str) -> None:
    """One dataset/leaderboard/predictions export block.

    Renders instantly on every visit: nothing here touches the dataframe's
    contents until the user clicks "Prepare Export" for that specific section,
    and the result is cached afterward so it never recomputes.
    """
    st.subheader(title)
    if data is None or data.empty:
        st.info(f"No {title.lower()} available yet.")
        return

    ready_key = f"reports_export_ready_{file_stem}"
    if not st.session_state.get(ready_key):
        st.caption(f"{len(data):,} rows x {data.shape[1]:,} columns.")
        if st.button(f"Prepare {title} Export", key=f"prepare_{file_stem}"):
            st.session_state[ready_key] = True
            st.rerun()
        return

    with st.spinner(f"Preparing {title.lower()} export..."):
        csv_bytes = _cached_csv_bytes(data, file_stem)
        xlsx_bytes = _cached_excel_bytes(data, file_stem)

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            f"Download {title} CSV", data=csv_bytes, file_name=f"{file_stem}.csv",
            mime="text/csv", key=f"dl_csv_{file_stem}",
        )
    with c2:
        st.download_button(
            f"Download {title} Excel", data=xlsx_bytes, file_name=f"{file_stem}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"dl_xlsx_{file_stem}",
        )


def _render_overview(df, leaderboard, predictions_df) -> None:
    """Instant summary metrics -- pure metadata already in memory, zero computation."""
    st.subheader("Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset Rows", f"{len(df):,}" if df is not None else "--")
    c2.metric("Dataset Columns", f"{df.shape[1]:,}" if df is not None else "--")
    c3.metric("Models Compared", f"{len(leaderboard):,}" if leaderboard is not None and not leaderboard.empty else "--")
    c4.metric("Predictions", f"{len(predictions_df):,}" if predictions_df is not None else "--")


# --------------------------------------------------------------------------- #
# AI narrative (unchanged behavior: button-gated, cached in session_state)
# --------------------------------------------------------------------------- #
def _parse_narrative_sections(text: str) -> dict:
    """Split a '## Heading' formatted AI narrative into {heading: body} pairs."""
    sections: dict = {}
    current = None
    buffer: list = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            if current:
                sections[current] = "\n".join(buffer).strip()
            current = stripped[3:].strip()
            buffer = []
        else:
            buffer.append(line)
    if current:
        sections[current] = "\n".join(buffer).strip()
    return sections


def _render_ai_narrative_section() -> list:
    """AI-generated narrative sections (Executive Summary, Key Insights,
    Business Recommendations, Risk Assessment, Suggested Next Steps).

    Purely additive: generates narrative text only and hands it to the PDF
    builder as structured ``ReportSection`` objects -- it never touches how
    CSV/Excel/PDF export itself works. The Groq call only ever runs on an
    explicit button click and the result is cached in session_state, so
    revisiting this page never re-triggers it.

    Returns:
        List of ``ReportSection`` (heading + body), in ``_NARRATIVE_HEADERS`` order.
    """
    st.divider()
    st.subheader("AI-Enhanced Narrative")
    st.caption(
        "Executive summary, key insights, recommendations, and risk assessment -- "
        "generated by AI from the current model and predictions."
    )

    service = get_llm_service()
    if not service.is_available():
        st.caption(f"AI narrative unavailable: {service.status_message()}")
        return []

    artifact_bundle = get_active_artifact_bundle(st.session_state)
    if artifact_bundle is None:
        st.info("Train a model on the **Model Training** page first to unlock the AI narrative.")
        return []

    if st.button("Generate AI Narrative"):
        with st.status("Generating AI narrative...", expanded=True) as status:
            status.write("Gathering model and prediction context...")
            predictions_df = st.session_state.get(StateKeys.PREDICTIONS)
            importance_df = extract_feature_importance(
                artifact_bundle.get("model"), artifact_bundle.get("model_feature_columns") or []
            )
            context = build_full_context(artifact_bundle, predictions_df, importance_df)
            status.write("Generating executive summary...")
            response = service.generate(build_report_narrative_prompt(context), system_instruction=SYSTEM_PROMPT)
            if response.success:
                st.session_state[StateKeys.AI_REPORT_NARRATIVE] = response.text
                status.update(label="AI narrative ready", state="complete")
            else:
                status.update(label="AI narrative failed", state="error")
                st.error(f"⚠️ {response.error}")

    narrative_text = st.session_state.get(StateKeys.AI_REPORT_NARRATIVE)
    if not narrative_text:
        st.info("Click **Generate AI Narrative** to add AI-written sections to this report.")
        return []

    parsed = _parse_narrative_sections(narrative_text)
    report_sections = []
    for header in _NARRATIVE_HEADERS:
        body = parsed.get(header)
        if body:
            with st.expander(header, expanded=True):
                st.markdown(body)
            report_sections.append(ReportSection(header, body))
    return report_sections


# --------------------------------------------------------------------------- #
# PDF summary: a structured executive report (cover page, tables, best-model
# highlight, optional charts). Only generated on explicit click; the finished
# PDF bytes are then cached in session_state so revisiting the page or
# triggering an unrelated rerun never regenerates it.
# --------------------------------------------------------------------------- #
def _build_dataset_overview(df: pd.DataFrame) -> dict:
    """Key-value pairs for the PDF's Dataset Overview table -- pure metadata,
    the same figures already shown on Dataset Explorer, just formatted for print."""
    total_cells = df.shape[0] * df.shape[1]
    missing = int(df.isna().sum().sum())
    missing_pct = (missing / total_cells * 100) if total_cells else 0.0
    return {
        "Rows": f"{df.shape[0]:,}",
        "Columns": f"{df.shape[1]:,}",
        "Missing Cells": f"{missing:,} ({missing_pct:.1f}%)",
        "Duplicate Rows": f"{int(df.duplicated().sum()):,}",
        "Memory (MB)": f"{get_memory_usage_mb(df):.2f}",
    }


def _build_predictions_overview(predictions_df: pd.DataFrame, problem_type: Optional[str]) -> dict:
    """Key-value pairs for the PDF's Predictions Overview table."""
    overview = {"Total Rows Predicted": f"{len(predictions_df):,}"}
    if "Prediction_Confidence" in predictions_df.columns:
        overview["Average Confidence"] = f"{predictions_df['Prediction_Confidence'].mean():.1%}"
    if problem_type == "classification":
        label_col = "Prediction_Label" if "Prediction_Label" in predictions_df.columns else "Prediction"
        overview["Most Common Prediction"] = str(predictions_df[label_col].value_counts().idxmax())
    elif "Prediction" in predictions_df.columns:
        overview["Average Predicted Value"] = f"{predictions_df['Prediction'].mean():,.2f}"
    return overview


def _build_leaderboard_chart(leaderboard: pd.DataFrame, problem_type: Optional[str]) -> Optional[ChartImage]:
    """Bonus: a bar chart of the primary ranking metric across all compared models."""
    metric = _PRIMARY_METRIC_BY_PROBLEM.get(problem_type)
    if not metric or metric not in leaderboard.columns or "Model" not in leaderboard.columns:
        return None
    ranks = leaderboard["Rank"].tolist() if "Rank" in leaderboard.columns else []
    highlight_pos = ranks.index(1) if 1 in ranks else 0
    png = build_bar_chart_png(
        leaderboard["Model"].astype(str).tolist(),
        leaderboard[metric].astype(float).tolist(),
        title=f"Model Comparison: {metric}",
        y_label=metric,
        highlight_index=highlight_pos,
    )
    return ChartImage("leaderboard", png)


def _build_predictions_chart(predictions_df: pd.DataFrame, problem_type: Optional[str]) -> Optional[ChartImage]:
    """Bonus: prediction distribution -- a class-count bar chart for classification,
    a histogram of predicted values for regression."""
    if problem_type == "classification":
        label_col = "Prediction_Label" if "Prediction_Label" in predictions_df.columns else "Prediction"
        if label_col not in predictions_df.columns:
            return None
        counts = predictions_df[label_col].astype(str).value_counts().head(10)
        if counts.empty:
            return None
        png = build_bar_chart_png(
            counts.index.tolist(), counts.values.tolist(),
            title="Prediction Distribution", y_label="Count",
        )
        return ChartImage("predictions", png)

    if "Prediction" in predictions_df.columns:
        values = predictions_df["Prediction"].dropna().tolist()
        if not values:
            return None
        png = build_histogram_png(values, title="Prediction Distribution", x_label="Predicted Value")
        return ChartImage("predictions", png)

    return None


def _render_pdf_section(ai_narrative_sections: list, df, leaderboard, predictions_df) -> None:
    st.divider()
    st.subheader("Summary PDF Report")
    st.caption(
        "A structured executive report -- cover page, dataset overview, model "
        "leaderboard table, best-model highlight, and the AI narrative above."
    )

    if st.button("Generate PDF Summary"):
        with st.spinner("Building PDF report..."):
            problem_type = st.session_state.get(StateKeys.PROBLEM_TYPE)
            has_leaderboard = leaderboard is not None and not leaderboard.empty
            has_predictions = predictions_df is not None and not predictions_df.empty

            best_model = None
            charts = []
            if has_leaderboard and "Rank" in leaderboard.columns:
                best_rows = leaderboard.loc[leaderboard["Rank"] == 1]
                if not best_rows.empty:
                    best_model = best_rows.iloc[0].to_dict()
                chart = _build_leaderboard_chart(leaderboard, problem_type)
                if chart:
                    charts.append(chart)

            if has_predictions:
                chart = _build_predictions_chart(predictions_df, problem_type)
                if chart:
                    charts.append(chart)

            report_data = BusinessReportData(
                report_title="Business Analytics Summary Report",
                dataset_name=st.session_state.get(StateKeys.DATASET_NAME),
                generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                dataset_overview=_build_dataset_overview(df) if df is not None else None,
                leaderboard=leaderboard if has_leaderboard else None,
                best_model=best_model,
                narrative_sections=ai_narrative_sections,
                predictions_overview=_build_predictions_overview(predictions_df, problem_type) if has_predictions else None,
                charts=charts,
            )

            try:
                pdf_bytes = build_business_report_pdf(report_data)
            except Exception as exc:  # never let a malformed section break the whole page
                st.error(f"Could not build the PDF report: {exc}")
            else:
                st.session_state[_PDF_BYTES_KEY] = pdf_bytes
                st.session_state[_PDF_FILENAME_KEY] = f"report_{generate_run_id()}.pdf"

    cached_pdf_bytes = st.session_state.get(_PDF_BYTES_KEY)
    if cached_pdf_bytes:
        st.download_button(
            "Download PDF",
            data=cached_pdf_bytes,
            file_name=st.session_state.get(_PDF_FILENAME_KEY, f"report_{generate_run_id()}.pdf"),
            mime="application/pdf",
        )
    else:
        st.caption("Click **Generate PDF Summary** to build a downloadable report.")


def render() -> None:
    """Render the Reports page."""
    st.title("Reports")
    st.caption("Export the current dataset, model leaderboard, and predictions in your preferred format.")

    df = st.session_state.get(StateKeys.RAW_DF)
    leaderboard = st.session_state.get(StateKeys.COMPARISON_RESULTS)
    predictions_df = st.session_state.get(StateKeys.PREDICTIONS)

    # 1. Overview/KPIs first -- instant, so the page always paints immediately.
    _render_overview(df, leaderboard, predictions_df)

    # 2. Exports -- opt-in per section (see _render_export_section docstring).
    st.divider()
    _render_export_section("Dataset", df, "dataset")
    st.divider()
    _render_export_section("Model Leaderboard", leaderboard, "model_leaderboard")
    st.divider()
    _render_export_section("Predictions", predictions_df, "predictions")

    # 3. AI narrative -- Groq only ever called on explicit click, cached after.
    ai_narrative_sections = _render_ai_narrative_section()

    # 4. PDF -- only built on explicit click, cached after.
    _render_pdf_section(ai_narrative_sections, df, leaderboard, predictions_df)
