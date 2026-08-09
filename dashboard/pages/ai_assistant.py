"""
AI Assistant page: the hub for every AI-powered feature -- free-form chat, an
on-demand executive summary (Module 2), and business recommendations
(Module 3).

Answers/summaries are grounded entirely in structured facts pulled from
session state (``services.ai_context``) -- the LLM never sees raw data beyond
what's summarized there, and is explicitly instructed (``SYSTEM_PROMPT``)
never to invent numbers, run predictions, or train models itself.

Talks to whichever LLM provider is active (see ``services/llm_service.py``)
through the provider-agnostic ``get_llm_service()`` facade -- this page never
imports or knows about a specific backend (currently Groq).

Kept as its own page (rather than appended to the Executive Dashboard) so the
already-verified Executive Dashboard page needs no changes this phase.
"""

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from services.ai_context import build_full_context, extract_feature_importance
from services.llm_service import get_llm_service
from services.prompt_templates import (
    SYSTEM_PROMPT,
    build_chat_context_message,
    build_executive_summary_prompt,
    build_recommendation_prompt,
)
from utils.config import StateKeys
from utils.helpers import get_active_artifact_bundle

_EXAMPLE_QUESTIONS = [
    "Why was this model selected?",
    "Summarize this dataset.",
    "Explain these predictions.",
    "What actions should I take?",
    "What are the major business risks?",
]


def _cache_key(artifact_bundle: dict, predictions_df) -> str:
    """Cheap staleness key: regenerate cached AI content only when the underlying
    model or prediction batch actually changes, not on every rerun."""
    n_rows = len(predictions_df) if predictions_df is not None else 0
    return f"{artifact_bundle.get('dataset_name')}|{artifact_bundle.get('model_name')}|{n_rows}"


def _invalidate_stale_cache(cache_key: str) -> None:
    if st.session_state.get(StateKeys.AI_INSIGHTS_CACHE_KEY) != cache_key:
        st.session_state.pop(StateKeys.AI_EXECUTIVE_SUMMARY, None)
        st.session_state.pop(StateKeys.AI_RECOMMENDATIONS, None)
        st.session_state[StateKeys.AI_INSIGHTS_CACHE_KEY] = cache_key


def _send_message(user_text: str, context: Dict[str, Any], history: List[dict]) -> None:
    """Append a user message to history, prime context on the first turn, get a reply."""
    if not history:
        # Ground the whole conversation with one hidden context turn + a hidden
        # acknowledgement, so every later reply can reference it without resending
        # the full context block (and without showing this bookkeeping to the user).
        history.append({"role": "user", "text": build_chat_context_message(context), "hidden": True})
        history.append({"role": "assistant", "text": "Understood -- I have the context ready.", "hidden": True})

    history.append({"role": "user", "text": user_text})

    with st.spinner("Analyzing your question..."):
        response = get_llm_service().chat(
            [{"role": t["role"], "text": t["text"]} for t in history],
            system_instruction=SYSTEM_PROMPT,
        )

    if response.success:
        history.append({"role": "assistant", "text": response.text})
    else:
        history.append({"role": "assistant", "text": f"⚠️ {response.error}"})

    st.rerun()


def _render_chat_tab(context: Dict[str, Any]) -> None:
    history: List[dict] = st.session_state.setdefault(StateKeys.AI_CHAT_HISTORY, [])

    _, clear_col = st.columns([5, 1])
    with clear_col:
        if st.button("Clear Chat", use_container_width=True, disabled=not history):
            st.session_state[StateKeys.AI_CHAT_HISTORY] = []
            st.rerun()

    if not history:
        st.markdown("**Try asking:**")
        cols = st.columns(2)
        for i, question in enumerate(_EXAMPLE_QUESTIONS):
            if cols[i % 2].button(question, key=f"example_q_{i}", use_container_width=True):
                _send_message(question, context, history)

    for turn in history:
        if turn.get("hidden"):
            continue
        with st.chat_message("user" if turn["role"] == "user" else "assistant"):
            st.markdown(turn["text"])

    user_input = st.chat_input("Ask about your dataset, model, or predictions...")
    if user_input:
        _send_message(user_input, context, history)


def _render_executive_summary_tab(context: Dict[str, Any]) -> None:
    st.caption("A concise, business-oriented summary of the current model and predictions (AI-generated).")

    if st.button("Generate Executive Summary", type="primary"):
        with st.spinner("Generating executive summary..."):
            response = get_llm_service().generate(
                build_executive_summary_prompt(context), system_instruction=SYSTEM_PROMPT
            )
        if response.success:
            st.session_state[StateKeys.AI_EXECUTIVE_SUMMARY] = response.text
        else:
            st.error(f"⚠️ {response.error}")

    summary = st.session_state.get(StateKeys.AI_EXECUTIVE_SUMMARY)
    if summary:
        st.markdown(summary)
    else:
        st.info("Click **Generate Executive Summary** to have the AI interpret the current results.")


def _render_recommendations_tab(context: Dict[str, Any]) -> None:
    st.caption("Business actions suggested from the model's predictions and feature importance (AI-generated).")

    if st.button("Generate Recommendations", type="primary"):
        with st.spinner("Preparing recommendations..."):
            response = get_llm_service().generate(
                build_recommendation_prompt(context), system_instruction=SYSTEM_PROMPT
            )
        if response.success:
            st.session_state[StateKeys.AI_RECOMMENDATIONS] = response.text
        else:
            st.error(f"⚠️ {response.error}")

    recommendations = st.session_state.get(StateKeys.AI_RECOMMENDATIONS)
    if recommendations:
        st.markdown(recommendations)
    else:
        st.info("Click **Generate Recommendations** to have the AI suggest next actions.")


def render() -> None:
    """Render the AI Assistant page."""
    st.title("AI Assistant")
    st.caption("Chat, executive summaries, and business recommendations -- AI-powered.")

    service = get_llm_service()
    if not service.is_available():
        st.warning(f"⚠️ {service.status_message()}")
        st.info("The rest of the platform works fully without this -- the AI Assistant is optional.")
        return

    artifact_bundle = get_active_artifact_bundle(st.session_state)
    if artifact_bundle is None:
        st.info("Train a model on the **Model Training** page first -- the assistant needs results to discuss.")
        return

    predictions_df = st.session_state.get(StateKeys.PREDICTIONS)
    importance_df = extract_feature_importance(
        artifact_bundle.get("model"), artifact_bundle.get("model_feature_columns") or []
    )
    context = build_full_context(artifact_bundle, predictions_df, importance_df)
    _invalidate_stale_cache(_cache_key(artifact_bundle, predictions_df))

    n_predictions = len(predictions_df) if predictions_df is not None else 0
    st.caption(
        f"Discussing **{artifact_bundle.get('dataset_name') or 'the active dataset'}** -- "
        f"model **{artifact_bundle.get('model_name') or 'unknown'}**, "
        f"{n_predictions:,} prediction(s) available."
    )

    tab_chat, tab_summary, tab_recs = st.tabs(["Chat", "Executive Summary", "Recommendations"])
    with tab_chat:
        _render_chat_tab(context)
    with tab_summary:
        _render_executive_summary_tab(context)
    with tab_recs:
        _render_recommendations_tab(context)
