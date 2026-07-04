#!/usr/bin/env python3
"""
HistorySummarizer module - LLM-based conversation history summarization

Provides an optional enhancement over the plain sliding-window template
formatting in ConversationHistory.  When a Brain instance is available,
older turns can be condensed into a compact summary string so that the
prompt fed to the main LLM stays short while still carrying the key
context from previous exchanges.
"""

import logging

from .conversation_history import ConversationHistory

logger = logging.getLogger(__name__)

_SUMMARIZE_PROMPT_TEMPLATE = (
    "以下の会話履歴を、重要な情報を保ちながら3文以内で簡潔に要約してください。\n\n"
    "{history_text}\n\n"
    "要約："
)


class HistorySummarizer:
    """Condense a :class:`ConversationHistory` into a single summary string.

    Uses the provided *brain* (an :class:`~src.brain.Brain` instance) to
    generate a natural-language summary of all stored turns.  This is the
    *Phase B* enhancement: instead of blindly appending raw turns to the
    prompt, the LLM is asked to distil them into a short, information-dense
    paragraph, reducing token usage while preserving conversational context.

    Args:
        brain: A :class:`~src.brain.Brain` instance used to call the LLM.
    """

    def __init__(self, brain):
        self._brain = brain

    def summarize(self, history: ConversationHistory) -> str:
        """Return an LLM-generated summary of *history*.

        Returns an empty string when *history* is empty (no LLM call is
        made in that case).

        Args:
            history: The conversation history to summarize.

        Returns:
            A compact summary string, or ``""`` if history is empty.
        """
        if history.is_empty():
            return ""

        history_text = history.format_for_prompt()
        prompt = _SUMMARIZE_PROMPT_TEMPLATE.format(history_text=history_text)

        logger.debug("Summarizing %d turn(s) with LLM.", len(history))
        summary = self._brain.generate_response(prompt)
        logger.debug("History summary: %s", summary)
        return summary
