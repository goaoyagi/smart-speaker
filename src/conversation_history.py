#!/usr/bin/env python3
"""
Conversation history module - Multi-turn dialogue memory.

Keeps the last N (query, answer) pairs so the assistant can:
- Re-speak the previous answer on a "repeat" command ("もう一回言って").
- Answer follow-up questions using prior context ("それはどういうこと？").

Uses a Sliding Window Memory (``collections.deque`` with ``maxlen``) so old
turns are discarded automatically in O(1). ``as_condensed_context`` renders the
retained turns into a short, prompt-friendly summary (Condense Question) with
each answer clipped to keep the prompt length bounded.
"""

import logging
from collections import deque

from .config import CONVERSATION_MAX_TURNS, CONVERSATION_ANSWER_CLIP

logger = logging.getLogger(__name__)

# Keywords that trigger re-speaking the previous answer.
_REPEAT_KEYWORDS = (
    "もう一回",
    "もう一度",
    "もういちど",
    "繰り返して",
    "くりかえして",
    "リピート",
)

# Keywords that end the continuous conversation loop.
_EXIT_KEYWORDS = (
    "終了",
    "しゅうりょう",
    "さようなら",
    "さよなら",
    "バイバイ",
    "ばいばい",
    "おやすみ",
    "会話をやめて",
    "会話を終わって",
)


class ConversationHistory:
    def __init__(self, max_turns=CONVERSATION_MAX_TURNS, answer_clip=CONVERSATION_ANSWER_CLIP,
                 summarizer=None):
        self._turns = deque(maxlen=max_turns)
        self._answer_clip = answer_clip
        # Optional callable(str) -> str for LLM-based condensation. When None,
        # the raw template context is used. Failures fall back to the template.
        self._summarizer = summarizer

    def add(self, query, answer):
        """Record a completed (query, answer) turn."""
        if not query or not answer:
            return
        self._turns.append((query.strip(), answer.strip()))

    def last_answer(self):
        """Return the most recent answer, or None if there is no history."""
        if not self._turns:
            return None
        return self._turns[-1][1]

    def is_empty(self):
        return len(self._turns) == 0

    def clear(self):
        self._turns.clear()

    def is_repeat_command(self, text):
        """Return True if the utterance asks to repeat the previous answer."""
        if not text:
            return False
        normalized = "".join(text.split())
        return any(keyword in normalized for keyword in _REPEAT_KEYWORDS)

    def is_exit_command(self, text):
        """Return True if the utterance asks to end the conversation loop."""
        if not text:
            return False
        normalized = "".join(text.split())
        return any(keyword in normalized for keyword in _EXIT_KEYWORDS)

    def _clip(self, answer):
        if self._answer_clip > 0 and len(answer) > self._answer_clip:
            return answer[: self._answer_clip] + "…"
        return answer

    def _template_context(self):
        lines = []
        for query, answer in self._turns:
            lines.append(f"ユーザーは「{query}」と質問し、「{self._clip(answer)}」と回答された。")
        return "\n".join(lines)

    def as_condensed_context(self):
        """Render retained turns into a short summary string for the prompt.

        Returns an empty string when there is no history, so callers can embed
        the result unconditionally. When a summarizer is configured it is used
        to compress the history; if it fails, we fall back to the raw template.
        """
        if not self._turns:
            return ""

        template = self._template_context()
        if self._summarizer is None:
            return template

        try:
            summary = self._summarizer(template)
        except Exception as e:
            logger.warning("History summarization failed, using raw context: %s", e)
            return template

        return summary.strip() if summary and summary.strip() else template
