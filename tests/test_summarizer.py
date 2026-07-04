#!/usr/bin/env python3
"""
Tests for ConversationSummarizer module
"""

import pytest
from unittest.mock import Mock, MagicMock
from src.summarizer import ConversationSummarizer
from src.conversation_history import ConversationHistory
from src.exceptions import GenerationError


@pytest.fixture
def mock_brain():
    """Return a mock Brain with configurable generate_response."""
    brain = Mock()
    brain.generate_response.return_value = "ユーザーは天気について質問した。"
    return brain


@pytest.fixture
def summarizer(mock_brain):
    """Return a ConversationSummarizer backed by mock_brain."""
    return ConversationSummarizer(mock_brain)


class TestSummarizeEmpty:
    def test_returns_empty_string_for_empty_history(self, summarizer, mock_brain):
        """Empty history should return '' without calling the LLM."""
        history = ConversationHistory()
        result = summarizer.summarize(history)
        assert result == ""
        mock_brain.generate_response.assert_not_called()


class TestSummarizeSuccess:
    def test_calls_brain_with_history_content(self, summarizer, mock_brain):
        """Summarizer should pass the formatted history to the LLM."""
        history = ConversationHistory()
        history.add_turn("今日の天気は？", "晴れです。")

        summarizer.summarize(history)

        prompt_used = mock_brain.generate_response.call_args[0][0]
        assert "今日の天気は？" in prompt_used
        assert "晴れです。" in prompt_used

    def test_returns_llm_response(self, summarizer, mock_brain):
        """Summarizer should return the LLM's response string."""
        history = ConversationHistory()
        history.add_turn("質問", "回答")

        result = summarizer.summarize(history)

        assert result == "ユーザーは天気について質問した。"

    def test_multiple_turns_all_included_in_prompt(self, summarizer, mock_brain):
        """All turns should appear in the prompt sent to the LLM."""
        history = ConversationHistory()
        history.add_turn("質問1", "回答1")
        history.add_turn("質問2", "回答2")

        summarizer.summarize(history)

        prompt_used = mock_brain.generate_response.call_args[0][0]
        assert "質問1" in prompt_used
        assert "回答1" in prompt_used
        assert "質問2" in prompt_used
        assert "回答2" in prompt_used


class TestSummarizeFallback:
    def test_falls_back_to_raw_history_on_llm_error(self, mock_brain):
        """When the LLM raises, summarizer should fall back to raw history text."""
        mock_brain.generate_response.side_effect = GenerationError("timeout")
        summarizer = ConversationSummarizer(mock_brain)

        history = ConversationHistory()
        history.add_turn("質問A", "回答A")

        result = summarizer.summarize(history)

        assert "質問A" in result
        assert "回答A" in result

    def test_fallback_does_not_raise(self, mock_brain):
        """Fallback path must not propagate the LLM exception."""
        mock_brain.generate_response.side_effect = Exception("network error")
        summarizer = ConversationSummarizer(mock_brain)

        history = ConversationHistory()
        history.add_turn("q", "a")

        result = summarizer.summarize(history)
        assert isinstance(result, str)
