"""
Reusable, provider-agnostic LLM prompt templates.

Every AI feature in this app (executive summary, recommendations, prediction
explanations, report narratives, chat) builds its prompt here rather than
inlining prompt text on a page -- so wording stays consistent, the
"never fabricate numbers" constraint is stated exactly once (in
``SYSTEM_PROMPT``), and the context block powering every prompt is formatted
by exactly one function (``ai_context.format_context_block``).

Each ``build_*_prompt`` function returns the user-turn text only; callers pass
``SYSTEM_PROMPT`` separately as the ``system_instruction`` argument to
``services/llm_service.py``'s ``get_llm_service()``, so it is never duplicated
inside prompt bodies. Nothing here is specific to any one LLM provider.
"""

from __future__ import annotations

from typing import Any, Dict

from services.ai_context import format_context_block

# --------------------------------------------------------------------------- #
# System prompt -- the platform's AI behavior contract, sent once per request
# as the model's system_instruction, never repeated inside prompt bodies.
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """\
You are the AI business analyst embedded in a generic, dataset-agnostic \
machine learning platform. The platform's own AutoML pipeline handles all \
data preprocessing, model training, model comparison, and prediction -- your \
role is strictly to EXPLAIN and INTERPRET the outputs that pipeline already \
produced, for a business audience that may not have a technical background.

Hard rules, no exceptions:
1. You NEVER run predictions, train models, tune hyperparameters, or claim to \
   have done so. All of that already happened before you were called.
2. You NEVER invent, estimate, guess, or round numbers that were not given to \
   you in the CONTEXT block. If a number isn't in the CONTEXT, say the data \
   isn't available rather than making one up.
3. You NEVER assume a specific business domain (e.g. do not assume "shipment" \
   or "supplier" unless those words literally appear in the CONTEXT) -- the \
   dataset could be from any industry.
4. Keep responses concise, structured, and written for a business executive: \
   short paragraphs or bullet points, minimal ML jargon, no code.
5. If the CONTEXT is sparse (e.g. no predictions generated yet), say so \
   plainly and suggest what the user should do next in the app.
"""


def build_executive_summary_prompt(context: Dict[str, Any]) -> str:
    """Prompt for Module 2: AI Executive Summary."""
    return (
        f"{format_context_block(context)}\n\n"
        "Write a concise executive summary of the above results for a business "
        "stakeholder. Cover, in short paragraphs or bullets:\n"
        "- Overall dataset / model observations\n"
        "- Key business findings from the predictions\n"
        "- Any operational risks suggested by the confidence levels or distribution\n"
        "- Notable patterns worth highlighting\n"
        "- One clear executive recommendation to close\n"
        "Use only the figures given above. Keep it under 200 words."
    )


def build_recommendation_prompt(context: Dict[str, Any]) -> str:
    """Prompt for Module 3: Business Recommendation Engine."""
    return (
        f"{format_context_block(context)}\n\n"
        "Based only on the results above, recommend 3-5 concrete business actions. "
        "Recommendations must stay generic enough to apply to any business dataset "
        "(this platform is not limited to one industry) -- for example, phrase them "
        "around themes like resource allocation, risk mitigation, process optimization, "
        "or planning, adapted to whatever the target column and top features actually are. "
        "Format as a short bulleted list, each bullet starting with a bold action title "
        "followed by a one-sentence justification grounded in the CONTEXT figures."
    )


def build_prediction_explanation_prompt(row_context: Dict[str, Any]) -> str:
    """Prompt for Module 5: Explain Prediction (a single row)."""
    lines = [
        "=== SINGLE PREDICTION TO EXPLAIN ===",
        f"Target being predicted: {row_context['target_column']}",
        f"Predicted value: {row_context.get('prediction_label') or row_context['prediction_value']}",
    ]
    if row_context.get("confidence") is not None:
        lines.append(f"Model confidence: {row_context['confidence']:.1%}")
    if row_context.get("top_features"):
        lines.append("This record's values for the model's most important features:")
        for f in row_context["top_features"]:
            lines.append(f"  - {f['feature']} = {f['value']}")

    return (
        "\n".join(lines) + "\n\n"
        "Explain this single prediction for a business user in 4 short parts:\n"
        "1. Why the model likely produced this result (reference the feature values above)\n"
        "2. Which contributing features mattered most\n"
        "3. What this means in business terms\n"
        "4. One recommended next step for this specific record\n"
        "Be concise -- a few sentences per part. Use only the values given above."
    )


def build_report_narrative_prompt(context: Dict[str, Any]) -> str:
    """Prompt for Module 6: AI Report Enhancement -- five labeled narrative sections."""
    return (
        f"{format_context_block(context)}\n\n"
        "Write a business report narrative based only on the results above, with "
        "exactly these five sections, each starting on its own line with the exact "
        "heading shown (including the ## marker), in this order:\n\n"
        "## Executive Summary\n"
        "## Key Insights\n"
        "## Business Recommendations\n"
        "## Risk Assessment\n"
        "## Suggested Next Steps\n\n"
        "Each section should be 2-4 sentences or a short bullet list. Do not add any "
        "other sections or commentary outside these five headings."
    )


def build_chat_context_message(context: Dict[str, Any]) -> str:
    """The first turn injected into a chat session: grounds all follow-up answers.

    Sent once as an early user-role turn (not resent on every message) so the
    assistant can answer free-form follow-up questions about the dataset,
    model, predictions, and recommendations without re-quoting the context
    block on every turn.
    """
    return (
        f"{format_context_block(context)}\n\n"
        "This is the context for our conversation. I'll ask questions about this "
        "dataset, the model, its predictions, and business implications. Answer only "
        "from the context above -- if I ask something the context doesn't cover, say so. "
        "Acknowledge briefly that you have this context, then wait for my first question."
    )
