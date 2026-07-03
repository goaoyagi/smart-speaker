#!/usr/bin/env python3
"""
ConversationHistory - Sliding window memory for multi-turn dialogue.

Maintains the last N question/answer pairs so that context-dependent
follow-up questions (e.g. "それはどういうこと？", "もう一回言って") work correctly.
"""

from collections import deque


REPEAT_COMMANDS = {"もう一回言って", "もう一度言って", "繰り返して", "もう一回", "もう一度"}


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
    # Special command detection
    # ------------------------------------------------------------------

    @staticmethod
    def is_repeat_command(text: str) -> bool:
        """Return True when the user asked to repeat the previous answer."""
        return text.strip() in REPEAT_COMMANDS
