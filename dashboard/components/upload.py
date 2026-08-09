"""
Reusable dataset-upload component.

Renders a file uploader, reads CSV/Excel into a DataFrame, and stores it in
Streamlit session_state under a caller-specified key -- used on the Home /
Dataset Explorer pages for the training set and on the Predictions page for
new inference data.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

from utils.config import CONFIG
from utils.helpers import read_uploaded_file
from utils.logger import get_logger

logger = get_logger(__name__)


def render_dataset_uploader(
    label: str = "Upload your dataset (CSV or Excel)",
    key: str = "dataset_uploader",
) -> Optional[pd.DataFrame]:
    """Render a file uploader widget and return the parsed DataFrame, if any.

    Args:
        label: Widget label shown to the user.
        key: Unique Streamlit widget key (avoid collisions across pages).

    Returns:
        Parsed DataFrame, or None if nothing has been uploaded yet.
    """
    uploaded_file = st.file_uploader(label, type=CONFIG.supported_upload_types, key=key)
    if uploaded_file is None:
        return None

    try:
        df = read_uploaded_file(uploaded_file)
        logger.info("Loaded uploaded file '%s' with shape %s", uploaded_file.name, df.shape)
        return df
    except (ValueError, pd.errors.ParserError) as exc:
        st.error(f"Could not read the uploaded file: {exc}")
        logger.exception("Failed to parse uploaded file")
        return None
