"""
Gemini LLM provider -- kept available as a secondary/future provider.

Ported from the original flat ``services/gemini_service.py`` into the
provider-agnostic architecture. Not the active default (Groq is, for lower
latency during live demos), but fully functional: switching back is a
one-line change in ``services/llm_service.py``, and adding any other
provider follows this same pattern.
"""

from __future__ import annotations

import os
from typing import List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from services.providers.base import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_TEMPERATURE,
    AIResponse,
    LLMProvider,
)
from utils.logger import get_logger

logger = get_logger(__name__)

MODEL_NAME = "gemini-2.5-pro"

_env_loaded = False


def _ensure_env_loaded() -> None:
    """Load ``.env`` into the process environment exactly once."""
    global _env_loaded
    if not _env_loaded:
        load_dotenv()
        _env_loaded = True


def _describe_client_error(exc: "errors.ClientError") -> str:
    """Translate a raw ClientError into a specific, user-friendly message.

    Classification is based on the actual (status, code) values Google's API
    returns -- e.g. an invalid key comes back as HTTP 400/INVALID_ARGUMENT
    with "API key not valid" in the message, not 401, so message content is
    checked too rather than relying on status code alone.
    """
    code = getattr(exc, "code", None)
    status = (getattr(exc, "status", "") or "").upper()
    message = getattr(exc, "message", "") or str(exc)

    if code == 429 or status == "RESOURCE_EXHAUSTED":
        return "Gemini API quota exceeded. Please wait a bit and try again, or check your plan's rate limits."
    if status in ("UNAUTHENTICATED", "PERMISSION_DENIED") or "api key" in message.lower():
        return "Invalid Gemini API key. Please check GEMINI_API_KEY in your `.env` file."
    if code == 400:
        return f"Gemini rejected the request: {message}"
    return f"Gemini API error ({code or status or 'unknown'}): {message}"


class GeminiProvider(LLMProvider):
    """Lazily-initialized, reusable Gemini client with graceful degradation."""

    def __init__(self) -> None:
        _ensure_env_loaded()
        self._api_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
        self._client: Optional[genai.Client] = None
        self._init_error: Optional[str] = None

        if not self._api_key:
            self._init_error = (
                "Gemini API key not configured. Add GEMINI_API_KEY to a `.env` file "
                "in the project root to enable AI features (see `.env.example`)."
            )
            return

        try:
            self._client = genai.Client(api_key=self._api_key)
        except Exception as exc:  # defensive: malformed key format, SDK/env issues, etc.
            logger.exception("Failed to initialize Gemini client")
            self._init_error = f"Could not initialize the Gemini client: {exc}"

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
        contents = [{"role": "user", "parts": [{"text": prompt}]}]
        return self._generate(contents, system_instruction, temperature, max_output_tokens)

    def chat(
        self,
        history: List[dict],
        system_instruction: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ) -> AIResponse:
        # The app's canonical role is "assistant"; Gemini's own SDK expects "model".
        contents = [
            {"role": "model" if turn["role"] == "assistant" else turn["role"], "parts": [{"text": turn["text"]}]}
            for turn in history
        ]
        return self._generate(contents, system_instruction, temperature, max_output_tokens)

    def _generate(
        self,
        contents: list,
        system_instruction: Optional[str],
        temperature: float,
        max_output_tokens: int,
    ) -> AIResponse:
        if not self.is_available():
            return AIResponse(success=False, error=self._init_error or "Gemini is not available.")

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            # gemini-2.5's "thinking" tokens count against max_output_tokens. Left on,
            # thinking regularly consumed nearly the whole budget and cut the visible
            # answer off mid-sentence (finish_reason still reported STOP, not
            # MAX_TOKENS -- confirmed via usage_metadata.thoughts_token_count). These
            # are direct, single-pass summarization/explanation prompts over data
            # already computed by the ML pipeline, not multi-step reasoning tasks, so
            # disabling it is the right trade: fast, complete, cheaper.
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        try:
            response = self._client.models.generate_content(model=MODEL_NAME, contents=contents, config=config)
        except errors.ClientError as exc:
            logger.warning("Gemini client error: %s", exc)
            return AIResponse(success=False, error=_describe_client_error(exc))
        except errors.ServerError as exc:
            logger.warning("Gemini server error: %s", exc)
            return AIResponse(
                success=False,
                error="Gemini is temporarily unavailable on Google's side. Please try again shortly.",
            )
        except errors.APIError as exc:
            logger.warning("Gemini API error: %s", exc)
            code = getattr(exc, "code", "?")
            message = getattr(exc, "message", str(exc))
            return AIResponse(success=False, error=f"Gemini API error ({code}): {message}")
        except Exception as exc:  # network failures, timeouts, DNS errors, etc.
            logger.exception("Unexpected error calling Gemini")
            return AIResponse(success=False, error=f"Could not reach Gemini -- check your internet connection. ({exc})")

        text = getattr(response, "text", None)
        if not text:
            return AIResponse(success=False, error="Gemini returned an empty response. Please try again.")
        return AIResponse(success=True, text=text.strip())
