#!/usr/bin/env python3
"""
ConversationSummarizer module - LLM-based history condensation

Uses the existing Ollama Brain to compress past conversation turns into a
single concise Japanese summary string, reducing prompt length while
preserving conversational context (Condense Question pattern).
"""

import logging

from .conversation_history import ConversationHistory

logger = logging.getLogger(__name__)

# Summarization prompt template (Japanese output to match the assistant's language)
_SUMMARIZE_PROMPT = """\
以下の会話履歴を、1〜3文の簡潔な日本語で要約してください。
要点のみを残し、アルファベットは使わないでください。

会話履歴：
{history}

要約："""


class ConversationSummarizer:
    """Condense a :class:`ConversationHistory` into a short summary using the LLM.

    Args:
        brain: A :class:`Brain` instance whose ``generate_response`` method is
            used to call Ollama.  Injected to allow easy mocking in tests.
    """

    def __init__(self, brain):
        self._brain = brain

    def summarize(self, history: ConversationHistory) -> str:
        """Return a condensed Japanese summary of *history*.

        Falls back gracefully:
        - Returns ``""`` if *history* is empty.
        - Returns the raw formatted history if the LLM call fails, so the
          pipeline always has *some* context rather than none.

        Args:
            history: The conversation history to summarize.

        Returns:
            A summary string suitable for embedding directly into a prompt, or
            an empty string when there is nothing to summarize.
        """
        if history.is_empty():
            return ""

        raw = history.format_for_prompt()
        prompt = _SUMMARIZE_PROMPT.format(history=raw)

        try:
            summary = self._brain.generate_response(prompt)
            logger.debug("History summary: %s", summary)
            return summary
        except Exception as e:
            logger.warning(
                "LLM summarization failed, falling back to raw history: %s", e
            )
            return raw
