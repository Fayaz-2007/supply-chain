"""
Target-leakage detection.

A feature "leaks" when it encodes information that would not actually be
available at prediction time -- typically because it is a downstream
*consequence* of the target rather than a cause of it. Such columns produce
near-perfect validation scores that collapse on real data.

This module is deliberately dataset-agnostic: it never looks at column names,
only at how much of the target a single column can explain on its own. That
makes it safe to run against any tabular dataset plugged into the project.

Detection strategy (per candidate column, target excluded):

* Classification -- the best accuracy achievable using *only* that column.
  Categoricals are scored by mapping each level (including a distinct level
  for missing values) to its majority class; numerics by the best single
  threshold split. A column scoring at/above the threshold effectively
  reproduces the target by itself.
* Regression -- the strongest absolute Pearson/Spearman correlation with the
  target, which catches columns that are linear or monotonic restatements of it.

Missing values are treated as their own category on purpose: a very common
leak is a column that is populated only when the positive class occurred, so
its *null pattern* alone gives the answer away.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)

# A column at/above this score is reported as a suspected leak.
DEFAULT_LEAKAGE_THRESHOLD: float = 0.98

# Detection is a diagnostic, not a fit -- sampling keeps it responsive on very
# large datasets without meaningfully changing the verdict.
_MAX_DETECTION_ROWS: int = 50_000

# Columns with more distinct values than this are scored as numerics rather than
# by per-level majority vote (a near-unique column would otherwise trivially
# "explain" the target by memorising one row per level).
_MAX_CATEGORICAL_LEVELS: int = 50


def _categorical_solo_accuracy(feature: pd.Series, target: pd.Series) -> float:
    """Best accuracy from mapping each level of `feature` to its majority class.

    Missing values are folded into their own explicit level so that
    "populated only for the positive class" leaks are caught.
    """
    table = pd.crosstab(feature.fillna("__MISSING__").astype(str), target)
    if table.empty:
        return 0.0
    return float(table.max(axis=1).sum() / len(target))


def _numeric_solo_accuracy(feature: pd.Series, target: pd.Series) -> float:
    """Best accuracy achievable from a single threshold split on `feature`.

    Sorts once and sweeps every candidate cut point in O(n log n), taking the
    better of the two class orientations.
    """
    values = pd.to_numeric(feature, errors="coerce")
    values = values.fillna(values.median())
    if values.nunique() <= 1:
        return 0.0

    positive_label = target.value_counts().index[0]
    order = np.argsort(values.to_numpy(), kind="mergesort")
    is_positive = (target.to_numpy()[order] == positive_label).astype(np.int64)

    n = len(is_positive)
    cumulative = np.cumsum(is_positive)
    total_positive = cumulative[-1]
    index = np.arange(1, n + 1)

    # Predict positive left-of-cut (and the mirrored orientation); take the best.
    left_positive = (cumulative + (n - index) - (total_positive - cumulative)) / n
    left_negative = ((index - cumulative) + (total_positive - cumulative)) / n
    return float(max(left_positive.max(), left_negative.max()))


def _regression_solo_score(feature: pd.Series, target: pd.Series) -> float:
    """Strongest absolute linear/monotonic correlation between feature and target."""
    values = pd.to_numeric(feature, errors="coerce")
    if values.nunique(dropna=True) <= 1:
        return 0.0
    valid = values.notna() & target.notna()
    if valid.sum() < 3:
        return 0.0

    x, y = values[valid], target[valid]
    scores = []
    for method in ("pearson", "spearman"):
        try:
            corr = x.corr(y, method=method)
        except (ValueError, TypeError):
            continue
        if pd.notna(corr):
            scores.append(abs(float(corr)))
    return max(scores) if scores else 0.0


def _detect_datetime_pair_leaks(
    df: pd.DataFrame,
    target: pd.Series,
    threshold: float,
    is_classification: bool,
    skip: set,
) -> List[dict]:
    """Flag pairs of date columns whose *difference* reproduces the target.

    Single-column scanning cannot catch this: a planned date and an actual date
    are each individually harmless, but subtracting them yields the realised
    delay -- which is often exactly what the target encodes. Because date
    columns are later feature-engineered into numeric offsets, a model can
    recover that difference on its own, so both columns must be flagged.
    """
    date_columns = []
    for column in df.columns:
        if column in skip:
            continue
        series = df[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            date_columns.append((column, series))
        elif not pd.api.types.is_numeric_dtype(series):
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
            if parsed.notna().mean() > 0.8:
                date_columns.append((column, parsed))

    findings: List[dict] = []
    for i in range(len(date_columns)):
        for j in range(i + 1, len(date_columns)):
            left_name, left = date_columns[i]
            right_name, right = date_columns[j]
            delta = (right - left).dt.days
            if delta.nunique(dropna=True) <= 1:
                continue

            score = (
                _numeric_solo_accuracy(delta, target)
                if is_classification
                else _regression_solo_score(delta, target)
            )
            if score >= threshold:
                reason = (
                    f"difference with '{right_name}' reproduces the target "
                    f"({score:.1%}) once dates are engineered into numeric features"
                )
                findings.append(
                    {"column": left_name, "score": round(score, 4), "is_suspected_leak": True,
                     "reason": reason}
                )
                findings.append(
                    {"column": right_name, "score": round(score, 4), "is_suspected_leak": True,
                     "reason": reason.replace(f"'{right_name}'", f"'{left_name}'")}
                )
    return findings


def detect_leaky_columns(
    df: pd.DataFrame,
    target_column: str,
    problem_type: str,
    threshold: float = DEFAULT_LEAKAGE_THRESHOLD,
    ignore_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Score every column by how much of the target it explains on its own.

    Args:
        df: Raw dataframe (before cleaning/encoding -- missing-value patterns matter).
        target_column: Name of the target column.
        problem_type: 'classification', 'regression', or 'time_series'.
        threshold: Score at/above which a column is flagged as a suspected leak.
        ignore_columns: Columns to skip entirely (e.g. an identifier column that is
            already excluded from modeling anyway).

    Returns:
        DataFrame with one row per scored column, sorted strongest-first, columns:
        ``column``, ``score``, ``is_suspected_leak``, ``reason``. Empty if the
        target is unusable (missing, constant, or all-null).
    """
    empty = pd.DataFrame(columns=["column", "score", "is_suspected_leak", "reason"])
    if target_column not in df.columns:
        return empty

    working = df[df[target_column].notna()]
    if working.empty or working[target_column].nunique() <= 1:
        return empty

    if len(working) > _MAX_DETECTION_ROWS:
        working = working.sample(_MAX_DETECTION_ROWS, random_state=42)

    target = working[target_column]
    is_classification = problem_type == "classification"
    if is_classification:
        target = target.astype(str)
        baseline = float(target.value_counts(normalize=True).iloc[0])
    else:
        target = pd.to_numeric(target, errors="coerce")
        baseline = 0.0

    skip = set(ignore_columns or []) | {target_column}
    rows = []

    for column in working.columns:
        if column in skip:
            continue

        series = working[column]
        n_unique = series.nunique(dropna=True)
        if n_unique <= 1:
            continue

        is_numeric = pd.api.types.is_numeric_dtype(series)
        treat_as_categorical = not is_numeric or n_unique <= _MAX_CATEGORICAL_LEVELS

        if is_classification:
            if treat_as_categorical and n_unique <= _MAX_CATEGORICAL_LEVELS:
                score = _categorical_solo_accuracy(series, target)
            elif is_numeric:
                score = _numeric_solo_accuracy(series, target)
            else:
                # High-cardinality text: only its null pattern could plausibly leak.
                score = _categorical_solo_accuracy(series.isna(), target)
        else:
            if is_numeric:
                score = _regression_solo_score(series, target)
            else:
                continue

        rows.append(
            {
                "column": column,
                "score": round(score, 4),
                "is_suspected_leak": bool(score >= threshold),
                "reason": _explain(series, target, score, baseline, is_classification),
            }
        )

    rows.extend(_detect_datetime_pair_leaks(working, target, threshold, is_classification, skip))

    if not rows:
        return empty

    # A column may be scored both on its own and as half of a leaking date pair --
    # keep whichever verdict is strongest so the pair finding is never masked.
    result = (
        pd.DataFrame(rows)
        .sort_values("score", ascending=False)
        .drop_duplicates(subset="column", keep="first")
        .reset_index(drop=True)
    )
    n_flagged = int(result["is_suspected_leak"].sum())
    if n_flagged:
        logger.warning(
            "Leakage scan flagged %d suspected leaky column(s): %s",
            n_flagged,
            ", ".join(result.loc[result["is_suspected_leak"], "column"].tolist()),
        )
    return result


def _explain(
    series: pd.Series, target: pd.Series, score: float, baseline: float, is_classification: bool
) -> str:
    """Build a short, human-readable justification for a column's score."""
    if not is_classification:
        return f"|correlation| with target = {score:.3f}"

    # A null pattern that lines up with the target is the classic "only recorded
    # when the event happened" leak -- worth calling out explicitly.
    if series.isna().any():
        null_score = _categorical_solo_accuracy(series.isna(), target)
        if null_score >= 0.98:
            return f"missing-value pattern alone predicts the target ({null_score:.1%})"

    lift = (score - baseline) / (1 - baseline) if baseline < 1 else 0.0
    return f"predicts target alone with {score:.1%} accuracy (baseline {baseline:.1%}, lift {lift:.1%})"


def summarize_leakage(leakage_df: pd.DataFrame) -> List[str]:
    """Return the column names flagged as suspected leaks, strongest first."""
    if leakage_df.empty or "is_suspected_leak" not in leakage_df.columns:
        return []
    return leakage_df.loc[leakage_df["is_suspected_leak"], "column"].tolist()
