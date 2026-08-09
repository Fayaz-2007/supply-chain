"""
Predictions page: the Prediction Engine.

Loads the saved model artifact bundle -- from the current session, or straight
from disk if the app was restarted, so no retraining is ever required -- lets
the user upload new data, validates it's structurally compatible with the
trained model, reuses the exact preprocessing pipeline learned at training
time, and generates predictions with a confidence score where available.
"""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from dashboard.charts import categorical_bar_chart, histogram, pie_chart
from dashboard.components.download import render_csv_download, render_excel_download
from models.predict import ModelPredictor, validate_new_data
from services.ai_context import build_row_context, extract_feature_importance
from services.llm_service import get_llm_service
from services.prompt_templates import SYSTEM_PROMPT, build_prediction_explanation_prompt
from utils.config import StateKeys
from utils.helpers import get_active_artifact_bundle, read_uploaded_file
from utils.logger import get_logger

logger = get_logger(__name__)


def _render_model_banner(artifact_bundle: dict) -> None:
    """Confirmation banner describing which model is currently active."""
    model_name = artifact_bundle.get("model_name") or "trained model"
    st.success(
        f"✅ Model loaded successfully -- **{model_name}** predicting "
        f"**{artifact_bundle['target_column']}** ({artifact_bundle['problem_type']})"
    )


def _render_compatibility_check(validation: dict) -> bool:
    """Show validation results for the uploaded file. Returns True if prediction can proceed."""
    if not validation["is_compatible"]:
        missing = validation["missing_columns"]
        st.error(
            "This file doesn't look compatible with the trained model -- "
            f"{len(missing)} of the required feature columns are missing "
            f"({', '.join(missing[:10])}{'...' if len(missing) > 10 else ''}). "
            "Please upload data with the same structure used for training."
        )
        return False

    if validation["missing_columns"]:
        missing = validation["missing_columns"]
        st.warning(
            f"⚠️ {len(missing)} expected column(s) are missing and will be treated as missing "
            f"values (auto-imputed by the saved preprocessing pipeline): {', '.join(missing)}"
        )
    if validation["extra_columns"]:
        extra = validation["extra_columns"]
        st.caption(
            f"Note: {len(extra)} extra column(s) in your file will be ignored by the model: "
            f"{', '.join(extra[:10])}{'...' if len(extra) > 10 else ''}"
        )
    return True


def _render_prediction_summary(result_df: pd.DataFrame, problem_type: str, elapsed_seconds: float) -> None:
    """Business-friendly summary of a completed prediction run."""
    st.subheader("Prediction Summary")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Rows Predicted", f"{len(result_df):,}")
    c2.metric("Prediction Time (s)", f"{elapsed_seconds:.2f}")
    if "Prediction_Confidence" in result_df.columns:
        c3.metric("Average Confidence", f"{result_df['Prediction_Confidence'].mean():.1%}")
    else:
        c3.metric("Average Confidence", "N/A")

    st.markdown("**Prediction Distribution**")
    if problem_type == "classification":
        label_col = "Prediction_Label" if "Prediction_Label" in result_df.columns else "Prediction"
        d1, d2 = st.columns(2)
        with d1:
            st.plotly_chart(categorical_bar_chart(result_df, label_col), use_container_width=True, key="pred_dist_bar")
        with d2:
            st.plotly_chart(pie_chart(result_df, label_col), use_container_width=True, key="pred_dist_pie")
    else:
        st.plotly_chart(histogram(result_df, "Prediction"), use_container_width=True, key="pred_dist_hist")


def _render_explain_section(
    result_df: pd.DataFrame, display_df: pd.DataFrame, artifact_bundle: dict
) -> None:
    """Module 5: let the user pick a row and have the AI explain that single prediction.

    Purely additive to the Predictions page -- reads already-generated results,
    never re-runs or alters the prediction itself.
    """
    service = get_llm_service()
    st.divider()
    st.subheader("Explain a Prediction")

    if not service.is_available():
        st.caption(f"AI explanations unavailable: {service.status_message()}")
        return
    if display_df.empty:
        st.caption("No rows to explain (current filter has no matches).")
        return

    row_options = display_df.index.tolist()
    label_col = "Prediction_Label" if "Prediction_Label" in display_df.columns else "Prediction"

    def _format_row(idx: int) -> str:
        return f"Row {idx} -- predicted {display_df.loc[idx, label_col]}"

    selected_idx = st.selectbox("Select a row to explain", row_options, format_func=_format_row)

    if st.button("Explain This Prediction with AI"):
        with st.spinner("Analyzing prediction results..."):
            importance_df = extract_feature_importance(
                artifact_bundle.get("model"), artifact_bundle.get("model_feature_columns") or []
            )
            row = result_df.loc[selected_idx]
            confidence = row.get("Prediction_Confidence")
            row_context = build_row_context(
                row=row,
                prediction_value=row.get("Prediction"),
                prediction_label=row.get("Prediction_Label"),
                confidence=confidence,
                importance_df=importance_df,
                target_column=artifact_bundle.get("target_column", ""),
                categorical_columns=artifact_bundle.get("categorical_columns") or [],
            )
            response = service.generate(
                build_prediction_explanation_prompt(row_context), system_instruction=SYSTEM_PROMPT
            )
        if response.success:
            st.session_state[StateKeys.AI_PREDICTION_EXPLANATION] = (selected_idx, response.text)
        else:
            st.error(f"⚠️ {response.error}")

    cached = st.session_state.get(StateKeys.AI_PREDICTION_EXPLANATION)
    if cached and cached[0] == selected_idx:
        st.markdown(cached[1])


def render() -> None:
    """Render the Predictions page."""
    st.title("Predictions")
    st.caption("Generate predictions on new data using the currently saved model -- no retraining required.")

    artifact_bundle = get_active_artifact_bundle(st.session_state)
    if artifact_bundle is None:
        st.info("No trained model found yet. Train a model on the **Model Training** page first.")
        return

    _render_model_banner(artifact_bundle)

    new_file = st.file_uploader("Upload new data for prediction (CSV or Excel)", type=["csv", "xlsx", "xls"])
    if new_file is None:
        return

    try:
        new_df = read_uploaded_file(new_file)
    except (ValueError, pd.errors.ParserError) as exc:
        st.error(f"Could not read the uploaded file: {exc}")
        return

    if new_df.empty:
        st.error("The uploaded file has no rows.")
        return

    st.write(f"Loaded {new_df.shape[0]:,} rows for prediction.")

    validation = validate_new_data(new_df, artifact_bundle)
    can_predict = _render_compatibility_check(validation)

    if can_predict and st.button("Generate Predictions", type="primary"):
        with st.status("Running prediction pipeline...", expanded=True) as status:
            status.write("✅ Model loaded successfully")
            try:
                start = time.perf_counter()
                predictor = ModelPredictor(artifact_bundle)
                result_df = predictor.predict(new_df)
                elapsed = time.perf_counter() - start
            except Exception as exc:  # surface any pipeline failure as a friendly message, not a crash
                logger.exception("Prediction failed")
                status.update(label="Prediction failed", state="error")
                st.error(f"Prediction failed: {exc}")
            else:
                status.write("✅ Preprocessing applied")
                status.write("✅ Prediction completed")
                st.session_state[StateKeys.PREDICTIONS] = result_df
                st.session_state[StateKeys.LAST_PREDICTION_SECONDS] = round(elapsed, 3)
                status.write("✅ Results ready")
                status.update(label="Prediction complete", state="complete")

    result_df = st.session_state.get(StateKeys.PREDICTIONS)
    if result_df is not None:
        elapsed_seconds = st.session_state.get(StateKeys.LAST_PREDICTION_SECONDS, 0.0)
        _render_prediction_summary(result_df, artifact_bundle["problem_type"], elapsed_seconds)

        st.divider()
        st.subheader("Prediction Results")

        display_df = result_df
        if artifact_bundle["problem_type"] == "classification":
            label_col = "Prediction_Label" if "Prediction_Label" in result_df.columns else "Prediction"
            class_options = ["(all)"] + sorted(result_df[label_col].astype(str).unique().tolist())
            chosen_class = st.selectbox("Filter by predicted class", class_options)
            if chosen_class != "(all)":
                display_df = result_df[result_df[label_col].astype(str) == chosen_class]

        st.dataframe(display_df, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            render_csv_download(result_df, "predictions.csv")
        with c2:
            render_excel_download(result_df, "predictions.xlsx")

        _render_explain_section(result_df, display_df, artifact_bundle)
