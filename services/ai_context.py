"""
AI context builder.

Every AI-powered feature in this app (executive summary, recommendations,
chat, prediction explanations, report narratives) must reason only from facts
the ML pipeline actually produced -- never invented numbers. This module is
the single place that reads session data (artifact bundle, predictions,
feature importance, KPIs) and turns it into a small, plain-data dict safe to
drop into a prompt. Every AI prompt builder consumes this same shape, so the
"don't fabricate numbers" rule has exactly one source of truth to enforce.

Nothing here calls an LLM provider or touches the ML pipeline -- it only
*reads* already-computed results. Provider-agnostic by design: this module
has no idea whether Groq, Gemini, or anything else is answering.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

_PRIMARY_METRIC_BY_PROBLEM = {"classification": "F1", "regression": "RMSE", "time_series": "RMSE"}


def extract_feature_importance(model: Any, feature_names: List[str]) -> pd.DataFrame:
    """Extract feature importances (tree models) or absolute coefficients (linear models).

    Shared by every AI feature that needs "what drove this model" -- the
    Executive Dashboard has its own identical helper it computes independently
    (out of scope to touch this phase), so this is intentionally a small,
    self-contained duplicate rather than a shared import across that boundary.
    """
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(np.ravel(model.coef_))
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    n = min(len(feature_names), len(values))
    return pd.DataFrame({"feature": feature_names[:n], "importance": values[:n]}).sort_values(
        "importance", ascending=False
    )


def _to_native(value: Any) -> Any:
    """Coerce numpy/pandas scalar types to plain Python so they format cleanly in prompts."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return round(float(value), 4)
    if isinstance(value, (pd.Timestamp,)):
        return str(value)
    if isinstance(value, dict):
        return {k: _to_native(v) for k, v in value.items()}
    return value


def build_model_context(artifact_bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize the trained model's identity and performance."""
    problem_type = artifact_bundle.get("problem_type", "classification")
    metrics = _to_native(artifact_bundle.get("metrics") or {})
    primary_metric = _PRIMARY_METRIC_BY_PROBLEM.get(problem_type)

    leaderboard = artifact_bundle.get("leaderboard")
    leaderboard_summary: List[Dict[str, Any]] = []
    if leaderboard is not None and hasattr(leaderboard, "empty") and not leaderboard.empty:
        cols = [c for c in leaderboard.columns if c not in ("Rank", "Training Time (s)")]
        leaderboard_summary = [
            {c: _to_native(row[c]) for c in cols} for _, row in leaderboard.head(6).iterrows()
        ]

    return {
        "model_name": artifact_bundle.get("model_name") or "unknown",
        "problem_type": problem_type,
        "target_column": artifact_bundle.get("target_column") or "unknown",
        "dataset_name": artifact_bundle.get("dataset_name") or "unknown",
        "trained_at": artifact_bundle.get("trained_at") or "unknown",
        "number_of_features": len(artifact_bundle.get("model_feature_columns") or []),
        "metrics": metrics,
        "primary_metric_name": primary_metric,
        "primary_metric_value": metrics.get(primary_metric) if primary_metric else None,
        "leaderboard": leaderboard_summary,
    }


def build_prediction_context(
    predictions_df: Optional[pd.DataFrame], problem_type: str
) -> Optional[Dict[str, Any]]:
    """Summarize the most recent prediction batch (counts/rates only, never raw rows)."""
    if predictions_df is None or predictions_df.empty:
        return None

    context: Dict[str, Any] = {"total_rows": int(len(predictions_df))}

    if "Prediction_Confidence" in predictions_df.columns:
        context["average_confidence"] = _to_native(predictions_df["Prediction_Confidence"].mean())
        context["low_confidence_rate"] = _to_native((predictions_df["Prediction_Confidence"] < 0.6).mean())

    if problem_type == "classification":
        label_col = "Prediction_Label" if "Prediction_Label" in predictions_df.columns else "Prediction"
        counts = predictions_df[label_col].value_counts()
        context["distribution"] = {
            str(label): {"count": int(count), "percent": round(count / len(predictions_df) * 100, 2)}
            for label, count in counts.items()
        }
    else:
        desc = predictions_df["Prediction"].describe()
        context["distribution"] = {k: _to_native(v) for k, v in desc.to_dict().items()}

    return context


def build_feature_importance_context(importance_df: pd.DataFrame, top_n: int = 10) -> List[Dict[str, Any]]:
    """Top-N feature importances as plain dicts, ordered strongest-first."""
    if importance_df is None or importance_df.empty:
        return []
    top = importance_df.head(top_n)
    return [
        {"feature": str(row["feature"]), "importance": _to_native(row["importance"])}
        for _, row in top.iterrows()
    ]


def build_kpi_context(kpis: List[tuple]) -> Dict[str, str]:
    """Flatten the (label, value, delta) KPI tuples already computed by a page into a dict."""
    return {label: value for label, value, *_ in kpis}


def build_full_context(
    artifact_bundle: Dict[str, Any],
    predictions_df: Optional[pd.DataFrame] = None,
    importance_df: Optional[pd.DataFrame] = None,
    kpis: Optional[List[tuple]] = None,
) -> Dict[str, Any]:
    """Assemble the complete context dict shared by every AI feature (summary, chat, reports, ...)."""
    problem_type = artifact_bundle.get("problem_type", "classification")
    return {
        "model": build_model_context(artifact_bundle),
        "predictions": build_prediction_context(predictions_df, problem_type),
        "feature_importance": build_feature_importance_context(importance_df) if importance_df is not None else [],
        "kpis": build_kpi_context(kpis) if kpis else {},
    }


def build_row_context(
    row: pd.Series,
    prediction_value: Any,
    prediction_label: Optional[str],
    confidence: Optional[float],
    importance_df: Optional[pd.DataFrame],
    target_column: str,
    categorical_columns: Optional[List[str]] = None,
    top_n: int = 8,
) -> Dict[str, Any]:
    """Context for explaining a single prediction: its top-importance feature values + outcome.

    Feature-importance names come from the *post-encoding* feature matrix, so a
    one-hot-encoded categorical shows up as e.g. ``port_congestion_level_High``
    rather than the raw column. That dummy name won't match anything in the raw
    prediction row and, even if it did, "= 1" means nothing to a business user.
    When a name doesn't match directly, this recovers the real original column
    (from ``categorical_columns``) and reports its actual raw value instead.
    """
    categorical_columns = categorical_columns or []
    top_features: List[Dict[str, Any]] = []
    seen_originals: set = set()

    if importance_df is not None and not importance_df.empty:
        for _, imp_row in importance_df.iterrows():
            if len(top_features) >= top_n:
                break
            feature = str(imp_row["feature"])

            if feature in row.index and feature not in seen_originals:
                top_features.append({"feature": feature, "value": _to_native(row[feature])})
                seen_originals.add(feature)
                continue

            matched_col = next(
                (c for c in categorical_columns if feature.startswith(f"{c}_") and c not in seen_originals),
                None,
            )
            if matched_col and matched_col in row.index:
                top_features.append({"feature": matched_col, "value": _to_native(row[matched_col])})
                seen_originals.add(matched_col)

    return {
        "target_column": target_column,
        "prediction_value": _to_native(prediction_value),
        "prediction_label": prediction_label,
        "confidence": _to_native(confidence) if confidence is not None else None,
        "top_features": top_features,
    }


def format_context_block(context: Dict[str, Any], title: str = "CONTEXT") -> str:
    """Render a context dict as a clean, readable text block for embedding in a prompt.

    Shared by every prompt builder in prompt_templates.py so formatting logic
    (and the "these are the ONLY real numbers" framing) lives in exactly one place.
    """
    lines = [f"=== {title} (all figures below are real outputs from the ML pipeline) ==="]

    model = context.get("model")
    if model:
        lines.append(
            f"Model: {model['model_name']} | Problem type: {model['problem_type']} | "
            f"Target column: {model['target_column']} | Dataset: {model['dataset_name']} | "
            f"Trained at: {model['trained_at']} | Features used: {model['number_of_features']}"
        )
        if model.get("metrics"):
            metric_str = ", ".join(f"{k}={v}" for k, v in model["metrics"].items())
            lines.append(f"Best model metrics: {metric_str}")
        if model.get("leaderboard"):
            lines.append("Model comparison leaderboard:")
            for row in model["leaderboard"]:
                lines.append("  - " + ", ".join(f"{k}={v}" for k, v in row.items()))

    predictions = context.get("predictions")
    if predictions:
        lines.append(f"Predictions: {predictions['total_rows']} rows scored.")
        if "average_confidence" in predictions:
            lines.append(
                f"Average confidence: {predictions['average_confidence']:.1%} "
                f"(low-confidence rate: {predictions.get('low_confidence_rate', 0):.1%})"
            )
        dist = predictions.get("distribution", {})
        if dist:
            lines.append("Prediction distribution: " + str(dist))

    features = context.get("feature_importance")
    if features:
        lines.append("Top contributing features (by model importance): " + ", ".join(
            f"{f['feature']} ({f['importance']})" for f in features
        ))

    kpis = context.get("kpis")
    if kpis:
        lines.append("Dashboard KPIs: " + ", ".join(f"{k}={v}" for k, v in kpis.items()))

    return "\n".join(lines)
