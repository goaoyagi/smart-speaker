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

from collections import deque

from .config import CONVERSATION_MAX_TURNS, CONVERSATION_ANSWER_CLIP

# Keywords that trigger re-speaking the previous answer.
_REPEAT_KEYWORDS = (
    "もう一回",
    "もう一度",
    "もういちど",
    "繰り返して",
    "くりかえして",
    "リピート",
)


class ConversationHistory:
    def __init__(self, max_turns=CONVERSATION_MAX_TURNS, answer_clip=CONVERSATION_ANSWER_CLIP):
        self._turns = deque(maxlen=max_turns)
        self._answer_clip = answer_clip

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

    def _clip(self, answer):
        if self._answer_clip > 0 and len(answer) > self._answer_clip:
            return answer[: self._answer_clip] + "…"
        return answer

    def as_condensed_context(self):
        """Render retained turns into a short summary string for the prompt.

        Returns an empty string when there is no history, so callers can embed
        the result unconditionally.
        """
        if not self._turns:
            return ""
        lines = []
        for query, answer in self._turns:
            lines.append(f"ユーザーは「{query}」と質問し、「{self._clip(answer)}」と回答された。")
        return "\n".join(lines)
