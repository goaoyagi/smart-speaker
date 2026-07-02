#!/usr/bin/env python3
"""
Conversation history module - Sliding Window Memory for multi-turn dialogue
"""

from collections import deque

DEFAULT_MAX_TURNS = 5

# Keywords that indicate a user wants the previous answer repeated
REPEAT_KEYWORDS = [
    "もう一回言って",
    "もう一度言って",
    "もう一回",
    "もう一度",
    "繰り返して",
    "繰り返し",
]


class ConversationHistory:
    """Manages conversation history using a sliding window (deque).

    Stores up to ``max_turns`` question-answer pairs.  Older turns are
    automatically discarded when the window is full (O(1) append/popleft).
    """

    def __init__(self, max_turns: int = DEFAULT_MAX_TURNS) -> None:
        self._history: deque = deque(maxlen=max_turns)

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def add_turn(self, user_input: str, ai_response: str) -> None:
        """Append a completed question-answer pair to the history."""
        self._history.append({"user": user_input, "assistant": ai_response})

    def clear(self) -> None:
        """Remove all stored turns."""
        self._history.clear()

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_history(self) -> list:
        """Return a copy of the current history as a plain list."""
        return list(self._history)

    def get_last_response(self) -> str | None:
        """Return the most recent AI response, or None if history is empty."""
        if self._history:
            return self._history[-1]["assistant"]
        return None

    def format_history_summary(self) -> str:
        """Condense the history into a text block for prompt injection.

        Returns an empty string when there is no history so that callers can
        use a simple truthiness check before embedding it into a prompt.
        """
        if not self._history:
            return ""

        parts = []
        for turn in self._history:
            parts.append(f"ユーザー：{turn['user']}\nアシスタント：{turn['assistant']}")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Special-command detection
    # ------------------------------------------------------------------

    def is_repeat_request(self, text: str) -> bool:
        """Return True when *text* contains a phrase asking to repeat the last answer."""
        return any(keyword in text for keyword in REPEAT_KEYWORDS)
