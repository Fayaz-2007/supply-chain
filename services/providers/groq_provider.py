"""
Groq LLM provider -- the platform's default/active LLM backend.

Uses the official Groq Python SDK against its OpenAI-compatible chat
completions API. Chosen for low-latency inference (important during a live
hackathon demo) and reliability. Swappable for any other ``LLMProvider``
implementation via ``services/llm_service.py`` without touching any page.
"""

from __future__ import annotations

import os
from typing import List, Optional

import groq
from dotenv import load_dotenv

from services.providers.base import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_TEMPERATURE,
    AIResponse,
    LLMProvider,
)
from utils.logger import get_logger

logger = get_logger(__name__)

# Single place to change the active Groq model.
MODEL_NAME = "llama-3.3-70b-versatile"

_env_loaded = False


def _ensure_env_loaded() -> None:
    """Load ``.env`` into the process environment exactly once."""
    global _env_loaded
    if not _env_loaded:
        load_dotenv()
        _env_loaded = True


def _describe_error(exc: Exception) -> str:
    """Translate a raw Groq SDK exception into a specific, user-friendly message.

    Checked most-specific-first since Groq's exception hierarchy has several
    leaf types (AuthenticationError, RateLimitError, ...) all rooted at the
    same ``APIStatusError``/``GroqError`` base classes.
    """
    if isinstance(exc, groq.AuthenticationError):
        return "Invalid Groq API key. Please check GROQ_API_KEY in your `.env` file."
    if isinstance(exc, groq.RateLimitError):
        return "Groq API rate limit reached. Please wait a moment and try again."
    if isinstance(exc, (groq.APIConnectionError, groq.APITimeoutError)):
        return f"Could not reach Groq -- check your internet connection. ({exc})"
    if isinstance(exc, groq.InternalServerError):
        return "Groq is temporarily unavailable on their side. Please try again shortly."
    if isinstance(exc, groq.APIStatusError):
        message = getattr(exc, "message", str(exc))
        return f"Groq API error ({exc.status_code}): {message}"
    return f"Unexpected error calling Groq: {exc}"


class GroqProvider(LLMProvider):
    """Lazily-initialized, reusable Groq client with graceful degradation."""

    def __init__(self) -> None:
        _ensure_env_loaded()
        self._api_key = (os.environ.get("GROQ_API_KEY") or "").strip()
        self._client: Optional[groq.Groq] = None
        self._init_error: Optional[str] = None

        if not self._api_key:
            self._init_error = (
                "Groq API key not configured. Add GROQ_API_KEY to a `.env` file "
                "in the project root to enable AI features (see `.env.example`)."
            )
            return

        try:
            self._client = groq.Groq(api_key=self._api_key)
        except Exception as exc:  # defensive: malformed key format, SDK/env issues, etc.
            logger.exception("Failed to initialize Groq client")
            self._init_error = f"Could not initialize the Groq client: {exc}"

    def is_available(self) -> bool:
        return self._client is not None

    def status_message(self) -> Optional[str]:
        return self._init_error

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ) -> AIResponse:
        messages = self._build_messages(system_instruction, [{"role": "user", "text": prompt}])
        return self._complete(messages, temperature, max_output_tokens)

    def chat(
        self,
        history: List[dict],
        system_instruction: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ) -> AIResponse:
        messages = self._build_messages(system_instruction, history)
        return self._complete(messages, temperature, max_output_tokens)

    @staticmethod
    def _build_messages(system_instruction: Optional[str], turns: List[dict]) -> list:
        """Translate the app's canonical {"role","text"} turns into Groq/OpenAI-shaped messages."""
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.extend({"role": turn["role"], "content": turn["text"]} for turn in turns)
        return messages

    def _complete(self, messages: list, temperature: float, max_output_tokens: int) -> AIResponse:
        if not self.is_available():
            return AIResponse(success=False, error=self._init_error or "Groq is not available.")

        try:
            response = self._client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_output_tokens,
            )
        except groq.GroqError as exc:
            logger.warning("Groq error: %s", exc)
            return AIResponse(success=False, error=_describe_error(exc))
        except Exception as exc:  # network failures, timeouts, DNS errors, etc.
            logger.exception("Unexpected error calling Groq")
            return AIResponse(success=False, error=f"Could not reach Groq -- check your internet connection. ({exc})")

        text = response.choices[0].message.content if response.choices else None
        if not text:
            return AIResponse(success=False, error="Groq returned an empty response. Please try again.")
        return AIResponse(success=True, text=text.strip())
