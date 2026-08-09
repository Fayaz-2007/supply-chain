"""
Provider-agnostic LLM contract.

Every LLM backend (Groq, Gemini, ...) implements this same interface, so
``services/llm_service.py`` -- and every page that calls it -- never needs to
know which provider is active or how its SDK works. Adding a new provider
means writing one class here and registering it in ``llm_service.py``;
nothing in the UI changes.

Conversation turns use one canonical shape everywhere in this app:
``{"role": "user"|"assistant", "text": str}``. Each provider translates that
into whatever shape its own SDK expects (e.g. Gemini's "model" role and
"parts" list, or Groq/OpenAI's "content" key).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

DEFAULT_TEMPERATURE = 0.4
DEFAULT_MAX_OUTPUT_TOKENS = 1024


@dataclass
class AIResponse:
    """Normalized result of any LLM call -- never raises; always check ``.success``."""

    success: bool
    text: str = ""
    error: Optional[str] = None

    def __bool__(self) -> bool:
        return self.success


class LLMProvider(ABC):
    """Common interface every LLM provider must implement.

    Construction must never raise: if a provider can't be used (missing key,
    bad SDK config, ...), ``is_available()`` returns False and
    ``status_message()`` explains why, so callers show a friendly notice
    instead of crashing.
    """

    @abstractmethod
    def is_available(self) -> bool:
        """Whether this provider can be used right now (key present, client constructed)."""
        raise NotImplementedError

    @abstractmethod
    def status_message(self) -> Optional[str]:
        """Human-friendly reason this provider is unavailable, or None if it's fine."""
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ) -> AIResponse:
        """Single-turn generation. Never raises -- failures come back as AIResponse(success=False)."""
        raise NotImplementedError

    @abstractmethod
    def chat(
        self,
        history: List[dict],
        system_instruction: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ) -> AIResponse:
        """Multi-turn generation from a full conversation history.

        Args:
            history: List of {"role": "user"|"assistant", "text": str} turns,
                oldest first, ending with the newest user turn.
        """
        raise NotImplementedError
