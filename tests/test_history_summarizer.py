#!/usr/bin/env python3
"""
Tests for HistorySummarizer module (Phase B)
"""

import pytest
from unittest.mock import MagicMock

from src.history_summarizer import HistorySummarizer
from src.conversation_history import ConversationHistory


@pytest.fixture
def mock_brain():
    brain = MagicMock()
    brain.generate_response.return_value = "ユーザーは天気と気温について質問した。"
    return brain


@pytest.fixture
def summarizer(mock_brain):
    return HistorySummarizer(mock_brain)


class TestHistorySummarizerInit:
    def test_stores_brain(self, mock_brain):
        """HistorySummarizer should store the provided brain."""
        s = HistorySummarizer(mock_brain)
        assert s._brain is mock_brain


class TestSummarize:
    def test_empty_history_returns_empty_string(self, summarizer, mock_brain):
        """Empty history should return '' without calling the LLM."""
        history = ConversationHistory()
        result = summarizer.summarize(history)
        assert result == ""
        mock_brain.generate_response.assert_not_called()

    def test_non_empty_history_calls_llm(self, summarizer, mock_brain):
        """Non-empty history should trigger a brain.generate_response call."""
        history = ConversationHistory()
        history.add_turn("今日の天気は？", "晴れです。")
        summarizer.summarize(history)
        mock_brain.generate_response.assert_called_once()

    def test_returns_llm_output(self, summarizer, mock_brain):
        """summarize() should return whatever the LLM responds with."""
        mock_brain.generate_response.return_value = "要約テキスト"
        history = ConversationHistory()
        history.add_turn("質問", "回答")
        result = summarizer.summarize(history)
        assert result == "要約テキスト"

    def test_prompt_contains_history_text(self, summarizer, mock_brain):
        """The prompt passed to the LLM should include the formatted history."""
        history = ConversationHistory()
        history.add_turn("テスト質問", "テスト回答")
        summarizer.summarize(history)
        call_args = mock_brain.generate_response.call_args
        prompt = call_args[0][0]
        assert "テスト質問" in prompt
        assert "テスト回答" in prompt

    def test_multiple_turns_all_included_in_prompt(self, summarizer, mock_brain):
        """All stored turns should appear in the summarization prompt."""
        history = ConversationHistory()
        history.add_turn("質問1", "回答1")
        history.add_turn("質問2", "回答2")
        history.add_turn("質問3", "回答3")
        summarizer.summarize(history)
        prompt = mock_brain.generate_response.call_args[0][0]
        for i in range(1, 4):
            assert f"質問{i}" in prompt
            assert f"回答{i}" in prompt
