#!/usr/bin/env python3
"""
Tests for conversation_history module
"""

import pytest
from src.conversation_history import ConversationHistory


@pytest.fixture
def history():
    return ConversationHistory(max_turns=3, answer_clip=200)


def test_starts_empty(history):
    assert history.is_empty()
    assert history.last_answer() is None
    assert history.as_condensed_context() == ""
    assert history.as_messages() == []


def test_add_and_last_answer(history):
    history.add("東京の天気は？", "晴れです。")
    assert not history.is_empty()
    assert history.last_answer() == "晴れです。"


def test_add_ignores_empty_query_or_answer(history):
    history.add("", "回答")
    history.add("質問", "")
    assert history.is_empty()


def test_sliding_window_discards_oldest(history):
    history.add("q1", "a1")
    history.add("q2", "a2")
    history.add("q3", "a3")
    history.add("q4", "a4")  # exceeds maxlen=3, drops q1/a1

    context = history.as_condensed_context()
    assert "q1" not in context
    assert "q4" in context
    assert history.last_answer() == "a4"


def test_clear(history):
    history.add("q", "a")
    history.clear()
    assert history.is_empty()


@pytest.mark.parametrize("text", [
    "もう一回言って",
    "もう一度お願い",
    "もういちど",
    "繰り返して",
    "リピートして",
    "  もう 一 回  ",  # whitespace is normalized away
])
def test_is_repeat_command_true(history, text):
    assert history.is_repeat_command(text) is True


@pytest.mark.parametrize("text", [
    "今日の天気は？",
    "",
    "東京について教えて",
])
def test_is_repeat_command_false(history, text):
    assert history.is_repeat_command(text) is False


def test_condensed_context_format(history):
    history.add("東京の天気は？", "晴れです。")
    context = history.as_condensed_context()
    assert "東京の天気は？" in context
    assert "晴れです。" in context


def test_condensed_context_clips_long_answer():
    history = ConversationHistory(max_turns=3, answer_clip=10)
    history.add("q", "あ" * 50)
    context = history.as_condensed_context()
    assert "…" in context
    # clipped answer (10 chars) + ellipsis, far shorter than the original 50
    assert len(context) < 50


def test_as_messages_role_order_and_clip():
    history = ConversationHistory(max_turns=3, answer_clip=10)
    history.add("東京の天気は？", "晴れです。")
    history.add("詳しくは？", "あ" * 50)

    messages = history.as_messages()

    assert [m["role"] for m in messages] == [
        "user", "assistant", "user", "assistant",
    ]
    assert messages[0]["content"] == "東京の天気は？"
    assert messages[1]["content"] == "晴れです。"
    assert messages[2]["content"] == "詳しくは？"
    assert messages[3]["content"] == "あ" * 10 + "…"
