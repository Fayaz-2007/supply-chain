"""
Generic, dataset-agnostic helper functions.

These utilities perform automatic column-type detection, target/date column
inference, problem-type inference, and small IO helpers. Nothing here refers
to any specific dataset or column name -- everything is derived at runtime
from whatever DataFrame is plugged in.
"""

from __future__ import annotations

import io
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from utils.config import (
    ID_LIKE_UNIQUE_RATIO,
    LOW_CARDINALITY_THRESHOLD,
    MODELS_OUTPUT_DIR,
    RAW_DATA_DIR,
    StateKeys,
)
from utils.logger import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Column-type detection
# --------------------------------------------------------------------------- #
def _is_string_like_dtype(series: pd.Series) -> bool:
    """True for classic ``object`` columns and pandas' dedicated string dtype.

    Pandas 2.x/3.x can infer a non-``object`` string dtype for text columns
    (e.g. when ``pd.options.future.infer_string`` is enabled), so a strict
    ``dtype == object`` check silently misses those columns.
    """
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)


def detect_datetime_columns(df: pd.DataFrame) -> List[str]:
    """Detect columns that are already datetime dtype or parse cleanly as dates.

    Args:
        df: Input dataframe.

    Returns:
        List of column names considered datetime-like.
    """
    datetime_cols: List[str] = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)
            continue
        if _is_string_like_dtype(df[col]):
            sample = df[col].dropna().head(50)
            if sample.empty:
                continue
            try:
                parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
                if parsed.notna().mean() > 0.8:
                    datetime_cols.append(col)
            except (ValueError, TypeError):
                continue
    return datetime_cols


def detect_boolean_columns(df: pd.DataFrame) -> List[str]:
    """Detect columns that are boolean (native bool dtype, or an object column
    containing only True/False *objects* -- e.g. a bool column that picked up
    NaNs and got upcast to object dtype).

    Deliberately does not treat plain 0/1 integer columns as boolean: Python
    considers ``0 == False`` and ``1 == True``, so a naive value-set check
    would misclassify ordinary binary numeric columns (including many
    classification targets) as boolean.
    """
    boolean_cols: List[str] = []
    for col in df.columns:
        if pd.api.types.is_bool_dtype(df[col]):
            boolean_cols.append(col)
            continue
        non_null_values = df[col].dropna().unique().tolist()
        if non_null_values and all(isinstance(v, (bool, np.bool_)) for v in non_null_values):
            boolean_cols.append(col)
    return boolean_cols


def detect_id_like_columns(df: pd.DataFrame) -> List[str]:
    """Detect columns that look like unique identifiers (near-unique per row).

    These are excluded from modeling features by default since they carry no
    generalizable signal.
    """
    id_cols: List[str] = []
    n_rows = len(df)
    if n_rows == 0:
        return id_cols
    for col in df.columns:
        unique_ratio = df[col].nunique(dropna=True) / n_rows
        looks_like_id = "id" in col.lower().replace("_", "").replace(" ", "")
        if unique_ratio >= ID_LIKE_UNIQUE_RATIO and (looks_like_id or _is_string_like_dtype(df[col])):
            id_cols.append(col)
    return id_cols


def detect_column_types(
    df: pd.DataFrame, target_column: Optional[str] = None
) -> Dict[str, List[str]]:
    """Classify every column into categorical / numerical / datetime / id-like buckets.

    Args:
        df: Input dataframe.
        target_column: Optional target column to exclude from feature buckets.

    Returns:
        Dict with keys: 'categorical', 'numerical', 'datetime', 'id_like', 'boolean'.
    """
    datetime_cols = detect_datetime_columns(df)
    id_cols = detect_id_like_columns(df)
    boolean_cols = detect_boolean_columns(df)

    excluded_cols = set(datetime_cols) | set(id_cols) | set(boolean_cols)
    candidate_cols = [c for c in df.columns if c not in excluded_cols]
    if target_column and target_column in candidate_cols:
        candidate_cols.remove(target_column)

    categorical_cols: List[str] = []
    numerical_cols: List[str] = []

    for col in candidate_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Low-cardinality numeric columns (e.g. encoded flags/ratings) behave
            # like categoricals for EDA/encoding purposes.
            if df[col].nunique(dropna=True) <= LOW_CARDINALITY_THRESHOLD and df[col].dtype != float:
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
        else:
            categorical_cols.append(col)

    return {
        "categorical": categorical_cols,
        "numerical": numerical_cols,
        "datetime": datetime_cols,
        "id_like": id_cols,
        "boolean": boolean_cols,
    }


def detect_target_column(df: pd.DataFrame) -> Optional[str]:
    """Best-effort heuristic guess of the target column.

    Looks for common business-target keywords first, then falls back to the
    last column of the dataframe. This is only a *suggestion* -- the UI always
    lets the user override it.

    Args:
        df: Input dataframe.

    Returns:
        Suggested target column name, or None if the dataframe is empty.
    """
    if df.empty:
        return None

    keywords = [
        "target", "label", "class", "outcome", "result", "churn", "default",
        "fraud", "risk", "demand", "sales", "revenue", "price", "profit",
        "quantity", "amount", "status", "y",
    ]
    lower_cols = {col: col.lower() for col in df.columns}
    for keyword in keywords:
        for col, lower in lower_cols.items():
            if lower == keyword or lower.endswith(f"_{keyword}") or lower.startswith(f"{keyword}_"):
                return col

    return df.columns[-1]


def infer_problem_type(df: pd.DataFrame, target_column: str) -> str:
    """Infer whether the task is classification or regression from the target.

    Args:
        df: Input dataframe.
        target_column: Name of the target column.

    Returns:
        'classification' or 'regression'.
    """
    series = df[target_column].dropna()
    if series.empty:
        return "classification"

    if not pd.api.types.is_numeric_dtype(series):
        return "classification"

    unique_ratio = series.nunique() / len(series)
    n_unique = series.nunique()

    # Small number of distinct numeric values (e.g. 0/1, 1-5 ratings) => classification.
    if n_unique <= 20 and unique_ratio < 0.05:
        return "classification"

    return "regression"


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def read_uploaded_file(uploaded_file: Any) -> pd.DataFrame:
    """Read a Streamlit ``UploadedFile`` (or file path) into a DataFrame.

    Supports CSV and Excel transparently based on file extension.

    Args:
        uploaded_file: Streamlit UploadedFile object or a path-like string.

    Returns:
        Parsed DataFrame.
    """
    name = getattr(uploaded_file, "name", str(uploaded_file))
    suffix = Path(name).suffix.lower()

    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(uploaded_file)
    return pd.read_csv(uploaded_file)


def validate_csv_upload(uploaded_file: Any) -> Optional[str]:
    """Validate that an uploaded file is a usable, non-empty CSV.

    Args:
        uploaded_file: Streamlit UploadedFile object.

    Returns:
        An error message if validation fails, or None if the file looks valid.
    """
    name = getattr(uploaded_file, "name", "")
    if not name.lower().endswith(".csv"):
        return "Only CSV files are supported. Please upload a file with a .csv extension."
    size = getattr(uploaded_file, "size", None)
    if size == 0:
        return "The uploaded file is empty."
    return None


def list_sample_datasets() -> List[Path]:
    """List the bundled sample CSV datasets shipped under ``data/raw``.

    These power the automatic no-upload fallback so the app always has
    something to display, even before a real dataset is uploaded.
    """
    return sorted(RAW_DATA_DIR.glob("*.csv"))


def load_sample_dataset(path: Optional[Path] = None) -> Tuple[pd.DataFrame, str]:
    """Load a bundled sample dataset for the automatic no-upload fallback.

    Args:
        path: Specific sample dataset to load. Defaults to the first bundled
            sample found under ``data/raw`` (alphabetical) if omitted.

    Returns:
        Tuple of (dataframe, display_name).
    """
    if path is None:
        candidates = list_sample_datasets()
        if not candidates:
            raise FileNotFoundError(f"No sample datasets found in {RAW_DATA_DIR}")
        path = candidates[0]
    df = pd.read_csv(path)
    logger.info("Loaded sample dataset '%s' with shape %s", path.name, df.shape)
    return df, path.name


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a DataFrame to CSV bytes for Streamlit download buttons."""
    return df.to_csv(index=False).encode("utf-8")


def dataframe_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Sheet1") -> bytes:
    """Serialize a DataFrame to XLSX bytes for Streamlit download buttons."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return buffer.getvalue()


def save_artifact(obj: Any, path: Path) -> Path:
    """Persist any Python object (model, pipeline, dict of artifacts) via joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    logger.info("Saved artifact to %s", path)
    return path


def load_artifact(path: Path) -> Any:
    """Load a joblib-persisted artifact."""
    logger.info("Loading artifact from %s", path)
    return joblib.load(path)


def get_active_artifact_bundle(
    session_state: Any, model_path: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    """Return the currently trained model's artifact bundle, without retraining.

    Checks the live session first (fast path while the app is running); if no
    model was trained in this session -- e.g. after an app restart -- falls back
    to the persisted joblib artifact on disk, so the Predictions and Executive
    Dashboard pages work immediately using whatever was last trained.

    Args:
        session_state: Streamlit's ``st.session_state`` (or any dict-like object
            supporting ``.get`` and item assignment).
        model_path: Override for the saved artifact location. Defaults to
            ``outputs/models/best_model.joblib``.

    Returns:
        The artifact bundle dict, or None if no trained model is available
        in the session or on disk.
    """
    bundle = session_state.get(StateKeys.ARTIFACT_BUNDLE)
    if bundle is not None:
        return bundle

    path = model_path or (MODELS_OUTPUT_DIR / "best_model.joblib")
    if not path.exists():
        return None

    bundle = load_artifact(path)
    session_state[StateKeys.ARTIFACT_BUNDLE] = bundle
    return bundle


def generate_run_id() -> str:
    """Generate a short unique identifier for a training run/report."""
    return uuid.uuid4().hex[:8]


def get_memory_usage_mb(df: pd.DataFrame) -> float:
    """Return the approximate memory footprint of a DataFrame in megabytes."""
    return round(df.memory_usage(deep=True).sum() / (1024 ** 2), 3)


def train_test_ratio_valid(test_size: float) -> bool:
    """Validate that a train/test split ratio is sane."""
    return 0.05 <= test_size <= 0.5


def safe_column_list(df: pd.DataFrame, columns: List[str]) -> List[str]:
    """Filter a list of column names down to those actually present in df."""
    return [c for c in columns if c in df.columns]


def run_preprocessing_pipeline(
    df: pd.DataFrame,
    target_column: str,
    numerical_columns: List[str],
    categorical_columns: List[str],
    problem_type: str,
    date_column: Optional[str] = None,
    test_size: float = 0.2,
    exclude_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run the full split -> clean -> encode -> scale pipeline end to end.

    Centralizes the exact sequence of preprocessing steps so both the
    Streamlit dashboard and the CLI (main.py) produce identical results from
    the same raw dataframe, target, and column-type lists.

    The train/test split happens *before* any transformer is fitted. Fitting on
    the full frame would let held-out rows influence imputation medians, scaler
    means, and the one-hot vocabulary, which inflates test scores. Every
    transformer here is therefore fitted on the training split only and merely
    applied to the test split.

    Args:
        df: Raw (or already target-augmented) dataframe.
        target_column: Name of the target column.
        numerical_columns: Detected numerical feature columns.
        categorical_columns: Detected categorical feature columns.
        problem_type: 'classification', 'regression', or 'time_series'.
        date_column: Detected date column, used only for time-series splitting.
        test_size: Fraction of rows held out for testing.
        exclude_columns: Columns to drop before modeling -- typically the
            suspected target leaks reported by ``preprocessing.leakage``.

    Returns:
        Dict with keys: X_train, X_test, y_train, y_test, cleaner, encoder,
        scaler, target_encoder, model_feature_columns, dropped_columns,
        rows_dropped_missing_target.
    """
    # Imported locally to avoid a circular import (preprocessing -> utils -> preprocessing).
    from preprocessing.clean import DataCleaner
    from preprocessing.encode import FeatureEncoder, encode_target
    from preprocessing.scale import FeatureScaler
    from preprocessing.split import split_data

    working = df.copy()

    # Rows with no label teach nothing and, for a categorical target, would be
    # label-encoded into a spurious extra class -- drop them up front.
    n_before = len(working)
    working = working[working[target_column].notna()]
    rows_dropped_missing_target = n_before - len(working)
    if rows_dropped_missing_target:
        logger.info("Dropped %d row(s) with a missing target", rows_dropped_missing_target)

    # Drop caller-nominated columns (suspected leaks) before anything else sees them.
    dropped_columns = [c for c in (exclude_columns or []) if c in working.columns and c != target_column]
    if dropped_columns:
        working = working.drop(columns=dropped_columns)
        logger.info("Excluded %d column(s) from modeling: %s", len(dropped_columns), ", ".join(dropped_columns))

    # De-duplicate before splitting. Row hygiene is not target-dependent, and doing it
    # here (rather than inside the cleaner's transform) keeps X and y row-aligned once
    # the split has happened.
    working = working.drop_duplicates()

    # Smart preprocessing needs to know about booleans/dates/identifiers too, not just
    # the numerical/categorical lists the caller passed in -- auto-detect them here so
    # every caller (dashboard, CLI) gets the same treatment for free.
    column_types = detect_column_types(working, target_column=target_column)
    boolean_columns = [c for c in column_types["boolean"] if c != target_column]
    datetime_columns = [
        c for c in column_types["datetime"] if c != target_column and c != date_column
    ]
    id_columns = [c for c in column_types["id_like"] if c != target_column]
    numerical_columns = [c for c in numerical_columns if c in working.columns]
    categorical_columns = [c for c in categorical_columns if c in working.columns]

    # Label-encoding the target is a pure value->code mapping (it uses no feature
    # information), so fitting it on the full column leaks nothing and avoids
    # crashing on classes that appear only in the test split.
    y_encoded, target_encoder = encode_target(working[target_column])
    working[target_column] = y_encoded

    # --- split FIRST, then fit every transformer on the training split only --- #
    X_train_raw, X_test_raw, y_train, y_test = split_data(
        working, target_column, problem_type=problem_type, test_size=test_size, date_column=date_column
    )

    # Duplicate-dropping is disabled here: it already happened above, and a
    # transform that removes rows would desynchronise X from y.
    cleaner = DataCleaner(drop_duplicates=False)
    cleaner.fit(
        X_train_raw,
        numerical_columns,
        categorical_columns,
        boolean_columns=boolean_columns,
        datetime_columns=datetime_columns,
        id_columns=id_columns,
    )
    X_train_clean = cleaner.transform(X_train_raw)
    X_test_clean = cleaner.transform(X_test_raw)

    encoder = FeatureEncoder()
    encode_cols = [c for c in categorical_columns if c in X_train_clean.columns]
    encoder.fit(X_train_clean, encode_cols)
    X_train_encoded = encoder.transform(X_train_clean)
    X_test_encoded = encoder.transform(X_test_clean)

    scaler = FeatureScaler(method="standard")
    scale_cols = [
        c for c in numerical_columns + cleaner.get_engineered_numeric_columns()
        if c in X_train_encoded.columns
    ]
    scaler.fit(X_train_encoded, scale_cols)
    X_train = scaler.transform(X_train_encoded)
    X_test = scaler.transform(X_test_encoded)

    # Guarantee identical column sets/order across splits before the model sees them.
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "cleaner": cleaner,
        "encoder": encoder,
        "scaler": scaler,
        "target_encoder": target_encoder,
        "model_feature_columns": list(X_train.columns),
        "dropped_columns": dropped_columns,
        "rows_dropped_missing_target": rows_dropped_missing_target,
    }


def build_preprocessing_preview(
    df: pd.DataFrame, target_column: str, date_column: Optional[str] = None
) -> Dict[str, Any]:
    """Run a lightweight clean + encode preview over the whole dataset for UI feedback.

    This powers the Dataset Explorer's readiness checklist and analysis tables. It
    mirrors the smart-cleaning rules used by ``run_preprocessing_pipeline`` (missing
    value imputation, boolean conversion, date feature engineering, identifier
    removal, categorical encoding) but skips target-encoding/scaling/splitting since
    those only matter once a train/test split exists. The actual training run still
    fits its own cleaner/encoder/scaler on the train split via
    ``run_preprocessing_pipeline``.

    Args:
        df: Raw dataframe.
        target_column: Selected target column (excluded from feature detection).
        date_column: Optional user-selected time-series date column, left untouched.

    Returns:
        Dict describing the detected column types and what the preview pipeline did.
    """
    from preprocessing.clean import DataCleaner
    from preprocessing.encode import FeatureEncoder

    column_types = detect_column_types(df, target_column=target_column)
    boolean_columns = [c for c in column_types["boolean"] if c != target_column]
    datetime_columns = [
        c for c in column_types["datetime"] if c != target_column and c != date_column
    ]
    id_columns = [c for c in column_types["id_like"] if c != target_column]
    categorical_columns = column_types["categorical"]
    numerical_columns = column_types["numerical"]

    cleaner = DataCleaner()
    cleaned = cleaner.fit_transform(
        df,
        numerical_columns,
        categorical_columns,
        boolean_columns=boolean_columns,
        datetime_columns=datetime_columns,
        id_columns=id_columns,
    )

    encoder = FeatureEncoder()
    encode_cols = [c for c in categorical_columns if c in cleaned.columns]
    encoded = encoder.fit_transform(cleaned, encode_cols)

    return {
        "column_types": column_types,
        "missing_before": int(df.isna().sum().sum()),
        "missing_after": int(cleaned.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "boolean_columns": boolean_columns,
        "datetime_columns_engineered": datetime_columns,
        "engineered_datetime_features": cleaner.get_engineered_numeric_columns(),
        "id_columns_removed": id_columns,
        "categorical_columns_encoded": encode_cols,
        "problem_type": infer_problem_type(df, target_column),
        "feature_count_before": df.shape[1],
        "feature_count_after": encoded.shape[1],
    }
