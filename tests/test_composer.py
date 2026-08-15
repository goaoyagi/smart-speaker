#!/usr/bin/env python3
"""
Tests for composer module
"""

import pytest
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

    assert "これまでの会話" in prompt
    assert "晴れです。" in prompt
    assert "では大阪は？" in prompt


def test_compose_prompt_empty_history_is_backward_compatible(composer, mock_search_results):
    """Empty history_context yields the same prompt as the two-arg form"""
    with_default = composer.compose_prompt("質問", mock_search_results)
    with_empty = composer.compose_prompt("質問", mock_search_results, "")

    assert with_default == with_empty
    assert "これまでの会話：\n" not in with_empty


@pytest.mark.parametrize("with_results", [False, True])
@pytest.mark.parametrize("with_history", [False, True])
def test_compose_prompt_preserves_structure_for_all_input_combinations(
    composer, mock_search_results, with_results, with_history
):
    """Prompt sections stay ordered for history/results combinations."""
    query = "では大阪は？"
    history = HISTORY if with_history else ""
    results = mock_search_results if with_results else []

    prompt = composer.compose_prompt(query, results, history)

    assert prompt.endswith("回答：")
    assert prompt.index("質問：" + query) < prompt.index("回答：")
    assert ("これまでの会話：\n" in prompt) is with_history
    assert ("検索結果：" in prompt) is with_results
    if with_results:
        assert prompt.index("検索結果：") < prompt.index("質問：" + query)
        assert "回答にはアルファベット（英語の単語や文）を含めず" in prompt
    else:
        assert "日本語のみで、アルファベット（英語の単語や文）を含めずに答えなさい。" in prompt
    if with_history:
        assert prompt.index("これまでの会話：\n") < prompt.index("質問：" + query)


@pytest.mark.parametrize("search_results", [[], [{"title": "結果", "content": "内容"}]])
def test_compose_prompt_keeps_japanese_only_constraints(
    composer, search_results
):
    """Japanese-only and alphabet prohibition instructions remain present."""
    prompt = composer.compose_prompt("質問", search_results)

    assert "日本語のみ" in prompt
    assert "アルファベット（英語の単語や文）を含め" in prompt
