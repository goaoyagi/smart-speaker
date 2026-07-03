#!/usr/bin/env python3
"""
ConversationHistory - Sliding window memory for multi-turn dialogue.

Maintains the last N question/answer pairs so that context-dependent
follow-up questions (e.g. "それはどういうこと？", "もう一回言って") work correctly.

Phase B extension: LLM-based summarisation collapses older turns into a
compact summary, reducing prompt length while preserving semantic context.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .brain import Brain

logger = logging.getLogger(__name__)

REPEAT_COMMANDS = {"もう一回言って", "もう一度言って", "繰り返して", "もう一回", "もう一度"}

_SUMMARIZE_PROMPT_TEMPLATE = """\
以下の会話履歴を、現在の質問への回答に役立つよう、日本語で簡潔に要約してください。
要約は3〜5文以内にまとめてください。

会話履歴：
{history}

要約："""


class ConversationHistory:
    """Sliding window memory using collections.deque.

    Appending a new turn is O(1); when maxlen is reached the oldest turn is
    automatically discarded (popleft), keeping memory bounded.
    """

    def __init__(self, max_turns: int = 5):
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        self._history: deque = deque(maxlen=max_turns)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, question: str, answer: str) -> None:
        """Append a completed turn to the history."""
        self._history.append({"question": question, "answer": answer})

    def clear(self) -> None:
        """Remove all stored turns."""
        self._history.clear()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def is_empty(self) -> bool:
        return len(self._history) == 0

    def last_answer(self) -> str | None:
        """Return the most recent answer, or None if history is empty."""
        if self._history:
            return self._history[-1]["answer"]
        return None

    def get_history(self) -> list[dict]:
        """Return a list of {question, answer} dicts (oldest first)."""
        return list(self._history)

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    def format_for_prompt(self) -> str:
        """Serialise history as a plain-text block for prompt injection.

        Returns an empty string when there is no history so that callers
        can cheaply test ``if history_text`` before embedding.
        """
        if not self._history:
            return ""
        lines = []
        for turn in self._history:
            lines.append(f"ユーザー: {turn['question']}")
            lines.append(f"アシスタント: {turn['answer']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # LLM-based summarisation (Phase B)
    # ------------------------------------------------------------------

    def summarize_with_llm(self, brain: "Brain") -> str:
        """Ask the LLM to condense the current history into a short summary.

        The summary is returned as a plain string.  Callers may pass it back
        through ``format_for_prompt`` by replacing stored turns, or embed it
        directly in a prompt.  On failure (empty LLM response, GenerationError)
        the raw ``format_for_prompt()`` output is returned as a safe fallback
        so the pipeline never loses context entirely.

        Args:
            brain: A ``Brain`` instance used to call the LLM.

        Returns:
            A compact Japanese summary of the stored conversation turns.
        """
        history_text = self.format_for_prompt()
        if not history_text:
            return ""

        prompt = _SUMMARIZE_PROMPT_TEMPLATE.format(history=history_text)
        try:
            summary = brain.generate_response(prompt)
        except Exception as exc:  # pragma: no cover — network / model errors
            logger.warning("LLM summarisation failed, falling back to raw history: %s", exc)
            return history_text

        if not summary or not summary.strip():
            logger.warning("LLM returned empty summary, falling back to raw history.")
            return history_text

        return summary.strip()

    def replace_with_summary(self, summary: str) -> None:
        """Replace all stored turns with a single synthetic summary turn.

        This compacts an arbitrarily long history into one deque entry, which
        keeps subsequent prompts short while preserving the gist of context.

        Args:
            summary: The condensed text produced by ``summarize_with_llm``.
        """
        self._history.clear()
        self._history.append({"question": "[要約]", "answer": summary})

    # ------------------------------------------------------------------
    # Special command detection
    # ------------------------------------------------------------------

    @staticmethod
    def is_repeat_command(text: str) -> bool:
        """Return True when the user asked to repeat the previous answer."""
        return text.strip() in REPEAT_COMMANDS
