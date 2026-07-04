#!/usr/bin/env python3
"""
ConversationHistory module - Sliding-window memory for multi-turn dialogue

Stores the last N (user, assistant) turn pairs and provides a formatted
string representation suitable for inclusion in a prompt template.
"""

from collections import deque


DEFAULT_MAX_TURNS = 5


class ConversationHistory:
    """Manage a bounded history of (user_query, assistant_response) pairs.

    Uses :class:`collections.deque` with *maxlen* so that old turns are
    automatically discarded in O(1) when the window is full.
    """

    def __init__(self, max_turns: int = DEFAULT_MAX_TURNS):
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        self._turns: deque = deque(maxlen=max_turns)

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def add_turn(self, user_query: str, assistant_response: str) -> None:
        """Append a completed turn to the history."""
        self._turns.append((user_query, assistant_response))

    def clear(self) -> None:
        """Remove all stored turns."""
        self._turns.clear()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def is_empty(self) -> bool:
        """Return *True* when no turns have been recorded yet."""
        return len(self._turns) == 0

    def get_turns(self) -> list:
        """Return a list of (user_query, assistant_response) tuples."""
        return list(self._turns)

    def __len__(self) -> int:
        return len(self._turns)

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_for_prompt(self) -> str:
        """Return history formatted as a multi-line string for prompt injection.

        Each turn is rendered as::

            ユーザー: <question>
            アシスタント: <answer>

        Returns an empty string when there is no history.
        """
        if self.is_empty():
            return ""

        lines = []
        for user_query, assistant_response in self._turns:
            lines.append(f"ユーザー: {user_query}")
            lines.append(f"アシスタント: {assistant_response}")

        return "\n".join(lines)
