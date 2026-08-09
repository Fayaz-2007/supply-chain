"""
Provider-agnostic LLM service facade.

Every AI-touching page in this app calls ``get_llm_service()`` and never
imports a specific provider -- so which LLM backend is active is a single
line below, not something scattered across the UI. Swapping providers (or
adding a new one) never requires touching a page.

Currently active: Groq (``llama-3.3-70b-versatile``), chosen for low-latency
inference. Gemini remains available and fully implemented in
``services/providers/gemini_provider.py`` -- flipping back is one line here.
"""

from __future__ import annotations

import streamlit as st

from services.providers.base import AIResponse, LLMProvider

__all__ = ["AIResponse", "LLMProvider", "get_llm_service"]

# --------------------------------------------------------------------------- #
# The single place that decides which provider is active. To add a new
# provider: implement LLMProvider in services/providers/<name>_provider.py,
# import it below, and add an entry to _PROVIDERS.
# --------------------------------------------------------------------------- #
_ACTIVE_PROVIDER = "groq"


def _build_provider(name: str) -> LLMProvider:
    if name == "groq":
        from services.providers.groq_provider import GroqProvider

        return GroqProvider()
    if name == "gemini":
        from services.providers.gemini_provider import GeminiProvider

        return GeminiProvider()
    raise ValueError(f"Unknown LLM provider: {name!r}")


@st.cache_resource(show_spinner=False)
def get_llm_service() -> LLMProvider:
    """Return the shared, active LLMProvider instance, constructed once per server process."""
    return _build_provider(_ACTIVE_PROVIDER)
