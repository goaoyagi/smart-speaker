#!/usr/bin/env python3
"""
Tests for composer module
"""

import pytest
import src.composer as composer_module
from src.composer import Composer


HISTORY = "ユーザーは「東京の天気は？」と質問し、「晴れです。」と回答された。"


@pytest.fixture
def composer():
    """Create Composer instance"""
    return Composer()


def test_composer_initialization(composer):
    """Test that Composer initializes correctly"""
    assert composer is not None


def test_compose_prompt_with_results(composer, mock_search_results):
    """Test prompt composition with search results"""
    query = "今日の天気"
    prompt = composer.compose_prompt(query, mock_search_results)
    
    assert "以下の検索結果のうち、質問に関係するものを『絶対に事実』として扱い" in prompt
    assert "回答は3〜5文で、結論を先に述べてから補足を加える形で、具体的に説明しなさい。" in prompt
    assert "「それ」「さっきの」などの指示語は、これまでの会話を参照して解釈しなさい。" in prompt
    assert "検索結果が質問と無関係な場合は、検索結果を使わず、これまでの会話とあなたの知識で答えなさい。" in prompt
    assert "検索結果を使って答える場合は、検索結果に書かれていないことを推測で補ってはいけません。" in prompt
    assert query in prompt
    assert "テスト結果1" in prompt
    assert "テストコンテンツ" in prompt


def test_compose_prompt_without_results(composer):
    """Test prompt composition without search results"""
    query = "今日の天気"
    prompt = composer.compose_prompt(query, [])
    
    assert "日本語のみで、アルファベット（英語の単語や文）を含めずに答えなさい。" in prompt
    assert "回答は3〜5文で、結論を先に述べてから補足を加える形で、具体的に説明しなさい。" in prompt
    assert "「それ」「さっきの」などの指示語は、これまでの会話を参照して解釈しなさい。" in prompt
    assert query in prompt
    assert "検索結果" not in prompt


def test_compose_prompt_with_history(composer, mock_search_results):
    """History context is included when provided"""
    prompt = composer.compose_prompt("では大阪は？", mock_search_results, HISTORY)

    assert "これまでの会話：" in prompt
    assert "晴れです。" in prompt
    assert "では大阪は？" in prompt


def test_compose_prompt_empty_history_is_backward_compatible(composer, mock_search_results):
    """Empty history_context yields the same prompt as the two-arg form"""
    with_default = composer.compose_prompt("質問", mock_search_results)
    with_empty = composer.compose_prompt("質問", mock_search_results, "")

    assert with_default == with_empty
    assert "これまでの会話：" not in with_empty


def test_compose_prompt_keeps_japanese_only_instruction(composer, mock_search_results):
    """Japanese-only / no-alphabet wording stays in both prompt branches."""
    with_results = composer.compose_prompt("今日の天気", mock_search_results)
    without_results = composer.compose_prompt("今日の天気", [])

    for prompt in (with_results, without_results):
        assert "日本語のみ" in prompt
        assert "アルファベット" in prompt
        assert "3〜5文" in prompt
        assert "結論を先に" in prompt
        assert "それ" in prompt
        assert "さっきの" in prompt
        assert prompt.rstrip().endswith("回答：")
        assert prompt.rfind("質問：今日の天気") < prompt.rfind("回答：")


def test_compose_prompt_treats_only_related_results_as_facts(composer, mock_search_results):
    """Related hits are facts; unrelated hits must not be used or padded."""
    prompt = composer.compose_prompt("今日の天気", mock_search_results)

    assert "質問に関係するもの" in prompt
    assert "絶対に事実" in prompt
    assert "無関係" in prompt
    assert "推測で補って" in prompt


def test_compose_prompt_structure_history_and_search_combinations(composer, mock_search_results):
    """History × search-result presence keeps the prompt structure intact."""
    history = "ユーザーは「東京の天気は？」と質問し、「晴れです。」と回答された。"
    with_search_history = composer.compose_prompt("では大阪は？", mock_search_results, history)
    with_search_no_history = composer.compose_prompt("では大阪は？", mock_search_results, "")
    no_search_history = composer.compose_prompt("では大阪は？", [], history)
    no_search_no_history = composer.compose_prompt("では大阪は？", [], "")

    for prompt in (
        with_search_history,
        with_search_no_history,
        no_search_history,
        no_search_no_history,
    ):
        assert "では大阪は？" in prompt
        assert "日本語のみ" in prompt
        assert "アルファベット" in prompt
        assert "3〜5文" in prompt
        assert prompt.rstrip().endswith("回答：")

    assert "これまでの会話：" in with_search_history
    assert "これまでの会話：" in no_search_history
    assert "これまでの会話：" not in with_search_no_history
    assert "これまでの会話：" not in no_search_no_history
    assert "検索結果" in with_search_history
    assert "検索結果" in with_search_no_history
    assert "検索結果" not in no_search_history
    assert "検索結果" not in no_search_no_history


def test_compose_messages_separates_system_history_and_user(composer, mock_search_results):
    """Chat messages use the system prompt and role-tagged history."""
    history_messages = [
        {"role": "user", "content": "前の質問"},
        {"role": "assistant", "content": "前の回答"},
    ]

    messages = composer.compose_messages(
        "現在の質問", mock_search_results, history_messages
    )

    assert messages[0] == {
        "role": "system",
        "content": composer_module.OLLAMA_SYSTEM_PROMPT,
    }
    assert messages[1:3] == history_messages
    assert messages[-1] == {
        "role": "user",
        "content": (
            "検索結果：\n"
            "- テスト結果1: これはテストコンテンツです。\n"
            "- テスト結果2: 別のテストコンテンツ。\n\n"
            "質問：現在の質問"
        ),
    }
    assert "回答はすべて日本語のみで行い" in messages[0]["content"]
    assert "アルファベット（英語の単語や文）を含めてはいけません" in messages[0]["content"]


def test_compose_messages_without_results_keeps_question_in_user_message(composer):
    """A search failure still produces a system and user message."""
    messages = composer.compose_messages("質問", [], [])

    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[-1]["content"] == "質問：質問"
    assert "質問：" in messages[-1]["content"]


def test_compose_messages_drops_search_results_by_whole_result(
    composer, mock_search_results, monkeypatch
):
    """Search context is trimmed by dropping trailing result units."""
    monkeypatch.setattr(composer_module, "CONTEXT_CHAR_BUDGET", 35)

    messages = composer.compose_messages("質問", mock_search_results, [])

    user_content = messages[-1]["content"]
    assert "テスト結果1" in user_content
    assert "テスト結果2" not in user_content
    assert user_content.endswith("質問：質問")


def test_compose_messages_drops_old_history_but_keeps_recent_and_user(
    composer, monkeypatch
):
    """Budget overflow drops old history while preserving system and user."""
    monkeypatch.setattr(composer_module, "OLLAMA_SYSTEM_PROMPT", "システム")
    monkeypatch.setattr(composer_module, "OLLAMA_NUM_CTX", 25)
    monkeypatch.setattr(composer_module, "OLLAMA_NUM_PREDICT", 0)
    monkeypatch.setattr(composer_module, "CHAR_TO_TOKEN_RATIO", 1.0)
    history_messages = [
        {"role": "user", "content": "古い質問"},
        {"role": "assistant", "content": "古い回答"},
        {"role": "user", "content": "新しい質問"},
        {"role": "assistant", "content": "新しい回答"},
    ]

    messages = composer.compose_messages("最後の質問", [], history_messages)

    assert messages[0] == {"role": "system", "content": "システム"}
    assert {"role": "user", "content": "古い質問"} not in messages
    assert {"role": "assistant", "content": "古い回答"} not in messages
    assert messages[1:3] == history_messages[2:]
    assert messages[-1] == {"role": "user", "content": "質問：最後の質問"}


def test_compose_messages_keeps_required_messages_when_user_exceeds_budget(
    composer, monkeypatch
):
    """An oversized final user message degrades without raising."""
    monkeypatch.setattr(composer_module, "OLLAMA_SYSTEM_PROMPT", "システム")
    monkeypatch.setattr(composer_module, "OLLAMA_NUM_CTX", 1)
    monkeypatch.setattr(composer_module, "OLLAMA_NUM_PREDICT", 0)

    messages = composer.compose_messages("長い質問", [], [])

    assert messages == [
        {"role": "system", "content": "システム"},
        {"role": "user", "content": "質問：長い質問"},
    ]
