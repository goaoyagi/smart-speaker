#!/usr/bin/env python3
"""
Tests for composer module
"""

import pytest
from src.composer import Composer


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
    
    assert "検索結果" in prompt
    assert query in prompt
    assert "テスト結果1" in prompt
    assert "テストコンテンツ" in prompt


def test_compose_prompt_without_results(composer):
    """Test prompt composition without search results"""
    query = "今日の天気"
    prompt = composer.compose_prompt(query, [])
    
    assert query in prompt
    assert "検索結果" not in prompt


def test_compose_prompt_with_history(composer, mock_search_results):
    """History context is included when provided"""
    history = "ユーザーは「東京の天気は？」と質問し、「晴れです。」と回答された。"
    prompt = composer.compose_prompt("では大阪は？", mock_search_results, history)

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


def test_compose_messages_puts_instructions_in_system(composer, mock_search_results):
    """Phase 2 instructions live on the system message, not the last user turn."""
    messages = composer.compose_messages("今日の天気", mock_search_results, [])

    assert messages[0]["role"] == "system"
    system = messages[0]["content"]
    assert "日本語のみ" in system
    assert "アルファベット" in system
    assert "3〜5文" in system
    assert "結論を先に" in system
    assert "それ" in system
    assert "さっきの" in system
    assert "無関係" in system
    assert "推測で補って" in system

    last = messages[-1]
    assert last["role"] == "user"
    assert last["content"].startswith("検索結果：")
    assert last["content"].endswith("質問：今日の天気")


def test_compose_messages_without_results_omits_search_block(composer):
    messages = composer.compose_messages("今日の天気", [], [])
    assert messages[-1]["content"] == "質問：今日の天気"
    assert "検索結果" not in messages[-1]["content"]


def test_compose_messages_inserts_history_between_system_and_question(composer):
    history = [
        {"role": "user", "content": "東京の天気は？"},
        {"role": "assistant", "content": "晴れです。"},
    ]
    messages = composer.compose_messages("では大阪は？", [], history)

    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[1]["content"] == "東京の天気は？"
    assert messages[2]["content"] == "晴れです。"
    assert messages[-1]["content"] == "質問：では大阪は？"


def test_compose_messages_drops_overflow_search_results_from_the_back(composer):
    """Search hits that exceed CONTEXT_CHAR_BUDGET are dropped from the back."""
    results = [
        {"title": "A", "content": "あ" * 900, "url": "http://a"},
        {"title": "B", "content": "い" * 900, "url": "http://b"},
        {"title": "C", "content": "う" * 900, "url": "http://c"},
    ]
    messages = composer.compose_messages("質問", results, [])
    user = messages[-1]["content"]

    assert "- A:" in user
    assert "- B:" in user
    assert "- C:" not in user
    assert "質問：質問" in user

