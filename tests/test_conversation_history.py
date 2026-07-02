#!/usr/bin/env python3
"""
Tests for conversation history module
"""

import pytest
from src.conversation_history import ConversationHistory, DEFAULT_MAX_TURNS, REPEAT_KEYWORDS


@pytest.fixture
def history():
    """Create a ConversationHistory instance with default settings."""
    return ConversationHistory()


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def test_initial_history_is_empty(history):
    assert history.get_history() == []


def test_initial_last_response_is_none(history):
    assert history.get_last_response() is None


def test_initial_format_summary_is_empty_string(history):
    assert history.format_history_summary() == ""


# ---------------------------------------------------------------------------
# add_turn / get_history
# ---------------------------------------------------------------------------

def test_add_single_turn(history):
    history.add_turn("質問1", "回答1")
    turns = history.get_history()
    assert len(turns) == 1
    assert turns[0] == {"user": "質問1", "assistant": "回答1"}


def test_add_multiple_turns(history):
    history.add_turn("質問1", "回答1")
    history.add_turn("質問2", "回答2")
    turns = history.get_history()
    assert len(turns) == 2
    assert turns[1]["user"] == "質問2"


def test_get_history_returns_copy(history):
    history.add_turn("質問", "回答")
    copy1 = history.get_history()
    copy2 = history.get_history()
    # Mutating the returned list must not affect the internal deque
    copy1.append({"user": "x", "assistant": "y"})
    assert history.get_history() == copy2


# ---------------------------------------------------------------------------
# Sliding window (maxlen eviction)
# ---------------------------------------------------------------------------

def test_sliding_window_evicts_oldest_turn():
    h = ConversationHistory(max_turns=3)
    for i in range(4):
        h.add_turn(f"質問{i}", f"回答{i}")
    turns = h.get_history()
    assert len(turns) == 3
    # Oldest turn (index 0) should be gone
    assert turns[0]["user"] == "質問1"


def test_default_max_turns():
    h = ConversationHistory()
    for i in range(DEFAULT_MAX_TURNS + 2):
        h.add_turn(f"Q{i}", f"A{i}")
    assert len(h.get_history()) == DEFAULT_MAX_TURNS


# ---------------------------------------------------------------------------
# get_last_response
# ---------------------------------------------------------------------------

def test_get_last_response_after_turns(history):
    history.add_turn("質問1", "回答1")
    history.add_turn("質問2", "回答2")
    assert history.get_last_response() == "回答2"


# ---------------------------------------------------------------------------
# format_history_summary
# ---------------------------------------------------------------------------

def test_format_summary_single_turn(history):
    history.add_turn("質問", "回答")
    summary = history.format_history_summary()
    assert "ユーザー：質問" in summary
    assert "アシスタント：回答" in summary


def test_format_summary_multiple_turns(history):
    history.add_turn("質問1", "回答1")
    history.add_turn("質問2", "回答2")
    summary = history.format_history_summary()
    assert "質問1" in summary
    assert "回答1" in summary
    assert "質問2" in summary
    assert "回答2" in summary


def test_format_summary_preserves_order(history):
    history.add_turn("最初の質問", "最初の回答")
    history.add_turn("次の質問", "次の回答")
    summary = history.format_history_summary()
    assert summary.index("最初の質問") < summary.index("次の質問")


# ---------------------------------------------------------------------------
# is_repeat_request
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("phrase", REPEAT_KEYWORDS)
def test_is_repeat_request_recognises_all_keywords(history, phrase):
    assert history.is_repeat_request(phrase) is True


def test_is_repeat_request_embedded_in_sentence(history):
    assert history.is_repeat_request("すみません、もう一回言ってください") is True


def test_is_repeat_request_unrelated_text_returns_false(history):
    assert history.is_repeat_request("今日の天気はどうですか") is False


def test_is_repeat_request_empty_string_returns_false(history):
    assert history.is_repeat_request("") is False


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------

def test_clear_removes_all_turns(history):
    history.add_turn("質問", "回答")
    history.clear()
    assert history.get_history() == []
    assert history.get_last_response() is None
    assert history.format_history_summary() == ""
