#!/usr/bin/env python3
"""
Tests for ConversationHistory module
"""

import pytest
from src.conversation_history import ConversationHistory, REPEAT_COMMANDS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def history():
    return ConversationHistory(max_turns=3)


@pytest.fixture
def history_with_turns(history):
    history.add("今日の天気は？", "今日は晴れです。")
    history.add("明日は？", "明日は雨の予報です。")
    return history


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def test_init_defaults():
    h = ConversationHistory()
    assert h.is_empty()


def test_init_custom_max_turns():
    h = ConversationHistory(max_turns=2)
    h.add("q1", "a1")
    h.add("q2", "a2")
    h.add("q3", "a3")
    assert len(h.get_history()) == 2
    assert h.get_history()[0]["question"] == "q2"


def test_init_invalid_max_turns():
    with pytest.raises(ValueError):
        ConversationHistory(max_turns=0)


# ---------------------------------------------------------------------------
# add / get_history
# ---------------------------------------------------------------------------

def test_add_single_turn(history):
    history.add("質問", "回答")
    turns = history.get_history()
    assert len(turns) == 1
    assert turns[0] == {"question": "質問", "answer": "回答"}


def test_add_respects_max_turns(history):
    for i in range(5):
        history.add(f"q{i}", f"a{i}")
    turns = history.get_history()
    assert len(turns) == 3
    assert turns[0]["question"] == "q2"
    assert turns[-1]["question"] == "q4"


def test_get_history_returns_copy(history):
    history.add("q", "a")
    result = history.get_history()
    result.append({"question": "extra", "answer": "extra"})
    assert len(history.get_history()) == 1


# ---------------------------------------------------------------------------
# is_empty / clear
# ---------------------------------------------------------------------------

def test_is_empty_initial(history):
    assert history.is_empty()


def test_is_empty_after_add(history):
    history.add("q", "a")
    assert not history.is_empty()


def test_clear(history_with_turns):
    history_with_turns.clear()
    assert history_with_turns.is_empty()


# ---------------------------------------------------------------------------
# last_answer
# ---------------------------------------------------------------------------

def test_last_answer_empty(history):
    assert history.last_answer() is None


def test_last_answer_returns_most_recent(history_with_turns):
    assert history_with_turns.last_answer() == "明日は雨の予報です。"


def test_last_answer_updates_on_new_add(history_with_turns):
    history_with_turns.add("来週は？", "来週は曇りです。")
    assert history_with_turns.last_answer() == "来週は曇りです。"


# ---------------------------------------------------------------------------
# format_for_prompt
# ---------------------------------------------------------------------------

def test_format_for_prompt_empty(history):
    assert history.format_for_prompt() == ""


def test_format_for_prompt_contains_turns(history_with_turns):
    text = history_with_turns.format_for_prompt()
    assert "今日の天気は？" in text
    assert "今日は晴れです。" in text
    assert "明日は？" in text
    assert "明日は雨の予報です。" in text


def test_format_for_prompt_speaker_labels(history):
    history.add("質問", "回答")
    text = history.format_for_prompt()
    assert "ユーザー:" in text or "ユーザー：" in text or "ユーザー: " in text
    assert "アシスタント" in text


def test_format_for_prompt_ordering(history):
    history.add("first", "answer1")
    history.add("second", "answer2")
    text = history.format_for_prompt()
    assert text.index("first") < text.index("second")


# ---------------------------------------------------------------------------
# is_repeat_command
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cmd", list(REPEAT_COMMANDS))
def test_is_repeat_command_known(cmd):
    assert ConversationHistory.is_repeat_command(cmd) is True


def test_is_repeat_command_with_whitespace():
    assert ConversationHistory.is_repeat_command("  もう一回言って  ") is True


def test_is_repeat_command_ordinary_question():
    assert ConversationHistory.is_repeat_command("今日の天気は？") is False


def test_is_repeat_command_empty():
    assert ConversationHistory.is_repeat_command("") is False
