#!/usr/bin/env python3
"""
Tests for ConversationHistory module
"""

import pytest
from src.conversation_history import ConversationHistory, DEFAULT_MAX_TURNS


class TestConversationHistoryInit:
    def test_default_initialization(self):
        """Default instance should be empty with DEFAULT_MAX_TURNS capacity."""
        history = ConversationHistory()
        assert len(history) == 0
        assert history.is_empty()

    def test_custom_max_turns(self):
        """Custom max_turns should be respected."""
        history = ConversationHistory(max_turns=3)
        assert history.is_empty()

    def test_invalid_max_turns_raises(self):
        """max_turns < 1 should raise ValueError."""
        with pytest.raises(ValueError):
            ConversationHistory(max_turns=0)
        with pytest.raises(ValueError):
            ConversationHistory(max_turns=-1)


class TestAddTurn:
    def test_add_single_turn(self):
        """Adding one turn increments length to 1."""
        history = ConversationHistory()
        history.add_turn("質問1", "回答1")
        assert len(history) == 1
        assert not history.is_empty()

    def test_add_multiple_turns(self):
        """Adding multiple turns within capacity should all be retained."""
        history = ConversationHistory(max_turns=3)
        history.add_turn("質問1", "回答1")
        history.add_turn("質問2", "回答2")
        history.add_turn("質問3", "回答3")
        assert len(history) == 3

    def test_sliding_window_evicts_oldest(self):
        """Exceeding max_turns should evict the oldest turn."""
        history = ConversationHistory(max_turns=2)
        history.add_turn("古い質問", "古い回答")
        history.add_turn("中間の質問", "中間の回答")
        history.add_turn("新しい質問", "新しい回答")

        assert len(history) == 2
        formatted = history.format_for_prompt()
        assert "古い質問" not in formatted
        assert "新しい質問" in formatted
        assert "中間の質問" in formatted


class TestClear:
    def test_clear_removes_all_turns(self):
        """clear() should reset history to empty."""
        history = ConversationHistory()
        history.add_turn("質問", "回答")
        history.clear()
        assert history.is_empty()
        assert len(history) == 0


class TestFormatForPrompt:
    def test_empty_history_returns_empty_string(self):
        """Empty history should produce an empty string."""
        history = ConversationHistory()
        assert history.format_for_prompt() == ""

    def test_single_turn_format(self):
        """Single turn should include user and assistant labels."""
        history = ConversationHistory()
        history.add_turn("今日の天気は？", "晴れです。")
        formatted = history.format_for_prompt()

        assert "ユーザー: 今日の天気は？" in formatted
        assert "アシスタント: 晴れです。" in formatted

    def test_multiple_turns_order(self):
        """Turns should appear in chronological order."""
        history = ConversationHistory()
        history.add_turn("最初の質問", "最初の回答")
        history.add_turn("二番目の質問", "二番目の回答")
        formatted = history.format_for_prompt()

        pos_first = formatted.index("最初の質問")
        pos_second = formatted.index("二番目の質問")
        assert pos_first < pos_second

    def test_format_contains_all_turns(self):
        """All turns within the window should appear in the formatted output."""
        history = ConversationHistory(max_turns=3)
        for i in range(1, 4):
            history.add_turn(f"質問{i}", f"回答{i}")
        formatted = history.format_for_prompt()

        for i in range(1, 4):
            assert f"質問{i}" in formatted
            assert f"回答{i}" in formatted
