"""
Dataset Explorer page: upload a dataset (or use a bundled sample) and auto-profile it.

This is the single entry point where a new dataset is loaded into
session_state. Every other page reads from the state populated here.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components.kpi import status_badge
from preprocessing.clean import get_missing_value_summary
from utils.config import CONFIG, PROBLEM_TYPE_LABELS, StateKeys
from utils.helpers import (
    build_preprocessing_preview,
    detect_column_types,
    detect_target_column,
    get_memory_usage_mb,
    infer_problem_type,
    list_sample_datasets,
    load_sample_dataset,
    read_uploaded_file,
    validate_csv_upload,
)


def _reset_downstream_state() -> None:
    """Clear everything downstream of a freshly loaded dataset."""
    for key in (
        StateKeys.TRAINED_MODELS,
        StateKeys.COMPARISON_RESULTS,
        StateKeys.BEST_MODEL_NAME,
        StateKeys.ARTIFACT_BUNDLE,
        StateKeys.PREDICTIONS,
        StateKeys.PREPROCESSING_BUNDLE,
    ):
        st.session_state.pop(key, None)


def _load_dataset(df: pd.DataFrame, name: str, source: str) -> None:
    """Store a newly loaded dataset in session_state and reset downstream state."""
    st.session_state[StateKeys.RAW_DF] = df
    st.session_state[StateKeys.DATASET_NAME] = name
    st.session_state[StateKeys.DATASET_SOURCE] = source
    _reset_downstream_state()


def _render_dataset_source() -> None:
    """Render the upload widget and the bundled-sample fallback/picker.

    Uploading a dataset always takes priority. If nothing has ever been
    loaded in this session, the first bundled sample dataset is loaded
    automatically so the app is never empty.
    """
    uploaded_file = st.file_uploader("Upload dataset (CSV only)", type=["csv"], key="dataset_explorer_uploader")

    if uploaded_file is not None:
        error = validate_csv_upload(uploaded_file)
        if error:
            st.error(error)
        else:
            try:
                df_uploaded = read_uploaded_file(uploaded_file)
            except (ValueError, pd.errors.ParserError) as exc:
                st.error(f"Could not read the uploaded file: {exc}")
            else:
                if df_uploaded.empty:
                    st.error("The uploaded CSV has no rows.")
                else:
                    _load_dataset(df_uploaded, uploaded_file.name, source="uploaded")

    sample_paths = list_sample_datasets()
    if sample_paths:
        with st.expander("Or use a bundled sample dataset"):
            sample_names = [p.name for p in sample_paths]
            chosen_name = st.selectbox("Sample dataset", sample_names, key="sample_dataset_choice")
            if st.button("Load sample dataset"):
                chosen_path = next(p for p in sample_paths if p.name == chosen_name)
                df_sample, name = load_sample_dataset(chosen_path)
                _load_dataset(df_sample, name, source="sample")

    if StateKeys.RAW_DF not in st.session_state and sample_paths:
        df_sample, name = load_sample_dataset(sample_paths[0])
        _load_dataset(df_sample, name, source="sample")


def render() -> None:
    """Render the Dataset Explorer page."""
    st.title("Dataset Explorer")
    st.caption(
        "Upload a CSV dataset, or fall back to a bundled sample. Columns, types, "
        "and the target are detected automatically."
    )

    _render_dataset_source()

    df = st.session_state.get(StateKeys.RAW_DF)
    if df is None:
        st.info("No dataset loaded yet. Upload a CSV file above to get started.")
        return

    source = st.session_state.get(StateKeys.DATASET_SOURCE, "sample")
    source_label = "uploaded" if source == "uploaded" else "bundled sample"
    st.success(
        f"Using **{st.session_state.get(StateKeys.DATASET_NAME, 'dataset')}** ({source_label}) -- "
        f"{df.shape[0]:,} rows x {df.shape[1]:,} columns"
    )

    # ------------------------------------------------------------------ #
    # Target / date column configuration
    # ------------------------------------------------------------------ #
    st.subheader("Target & Date Configuration")
    suggested_target = detect_target_column(df)
    col1, col2 = st.columns(2)
    with col1:
        target_column = st.selectbox(
            "Target column",
            options=list(df.columns),
            index=list(df.columns).index(suggested_target) if suggested_target in df.columns else 0,
        )
    column_types = detect_column_types(df, target_column=target_column)
    with col2:
        date_options = ["(none)"] + column_types["datetime"]
        date_column = st.selectbox("Date column (optional, for time series)", options=date_options)
        date_column = None if date_column == "(none)" else date_column

    suggested_problem = infer_problem_type(df, target_column)
    problem_type = st.selectbox(
        "Problem type",
        options=list(PROBLEM_TYPE_LABELS.keys()),
        format_func=lambda k: PROBLEM_TYPE_LABELS[k],
        index=list(PROBLEM_TYPE_LABELS.keys()).index(suggested_problem),
    )

    st.session_state[StateKeys.TARGET_COLUMN] = target_column
    st.session_state[StateKeys.DATE_COLUMN] = date_column
    st.session_state[StateKeys.PROBLEM_TYPE] = problem_type
    st.session_state[StateKeys.CATEGORICAL_COLUMNS] = column_types["categorical"]
    st.session_state[StateKeys.NUMERICAL_COLUMNS] = column_types["numerical"]
    st.session_state[StateKeys.FEATURE_COLUMNS] = column_types["categorical"] + column_types["numerical"]

    # ------------------------------------------------------------------ #
    # Profile summary
    # ------------------------------------------------------------------ #
    st.divider()
    st.subheader("Dataset Profile")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Rows", f"{df.shape[0]:,}")
    m2.metric("Columns", f"{df.shape[1]:,}")
    m3.metric("Missing Cells", f"{int(df.isna().sum().sum()):,}")
    m4.metric("Duplicate Rows", f"{int(df.duplicated().sum()):,}")
    m5.metric("Memory (MB)", f"{get_memory_usage_mb(df):.2f}")

    with st.expander(f"Column names ({df.shape[1]})"):
        st.write(", ".join(str(c) for c in df.columns))

    tab1, tab2, tab3 = st.tabs(["Feature Summary", "Data Types & Missing Values", "Preview"])

    with tab1:
        st.markdown(
            f"**Detected {len(column_types['numerical'])} numerical**, "
            f"**{len(column_types['categorical'])} categorical**, "
            f"**{len(column_types['boolean'])} boolean**, "
            f"**{len(column_types['datetime'])} datetime**, and "
            f"**{len(column_types['id_like'])} identifier-like** columns."
        )
        summary_df = pd.DataFrame(
            {
                "column": df.columns,
                "role": [
                    "target" if c == target_column
                    else "date" if c == date_column
                    else "id_like" if c in column_types["id_like"]
                    else "boolean" if c in column_types["boolean"]
                    else "categorical" if c in column_types["categorical"]
                    else "numerical" if c in column_types["numerical"]
                    else "other"
                    for c in df.columns
                ],
                "dtype": df.dtypes.astype(str).values,
                "unique_values": [df[c].nunique(dropna=True) for c in df.columns],
            }
        )
        st.dataframe(summary_df, use_container_width=True)

    with tab2:
        missing_summary = get_missing_value_summary(df)
        st.dataframe(missing_summary, use_container_width=True)

    with tab3:
        st.dataframe(df.head(CONFIG.max_preview_rows), use_container_width=True)

    # ------------------------------------------------------------------ #
    # Smart preprocessing preview + readiness checklist
    # ------------------------------------------------------------------ #
    st.divider()
    st.subheader("Smart Preprocessing")
    st.caption(
        "A preview of the automatic cleaning pipeline that will run before training -- "
        "no configuration required."
    )

    preview, preview_error = None, None
    try:
        preview = build_preprocessing_preview(df, target_column, date_column=date_column)
    except Exception as exc:  # defensive: a preview failure should never break the page
        preview_error = str(exc)

    checklist = [
        ("Dataset Loaded", df is not None),
        ("Dataset Analysed", preview is not None),
        ("Missing Values Processed", preview is not None),
        ("Features Encoded", preview is not None),
        ("Task Detected", preview is not None),
        ("Ready for Training", preview is not None and bool(target_column)),
    ]
    checklist_cols = st.columns(3)
    for i, (label, ready) in enumerate(checklist):
        with checklist_cols[i % 3]:
            st.markdown(status_badge(label, ready))

    if preview_error:
        st.error(f"Preprocessing preview failed: {preview_error}")
    elif preview:
        p1, p2, p3 = st.columns(3)
        p1.metric("Missing Values", f"{preview['missing_before']:,} -> {preview['missing_after']:,}")
        p2.metric("Duplicate Rows", f"{preview['duplicate_rows']:,}")
        p3.metric("Task Detected", PROBLEM_TYPE_LABELS[preview["problem_type"]])

        detail_col1, detail_col2 = st.columns(2)
        with detail_col1:
            st.markdown("**Boolean columns converted to 0/1**")
            st.write(", ".join(preview["boolean_columns"]) or "None detected")
            st.markdown("**Identifier columns removed from training**")
            st.write(", ".join(preview["id_columns_removed"]) or "None detected")
        with detail_col2:
            st.markdown("**Date columns engineered into numeric features**")
            st.write(", ".join(preview["datetime_columns_engineered"]) or "None detected")
            st.markdown("**Categorical columns auto-encoded**")
            st.write(", ".join(preview["categorical_columns_encoded"]) or "None detected")

        st.caption(
            f"Feature columns: {preview['feature_count_before']} raw -> "
            f"{preview['feature_count_after']} after smart preprocessing (target excluded)."
        )
