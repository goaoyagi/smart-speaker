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


def test_compose_prompt_with_history(composer):
    """Test that conversation history is embedded in the prompt"""
    history_text = "ユーザー: 前の質問\nアシスタント: 前の回答"
    prompt = composer.compose_prompt("今の質問", [], history_text=history_text)

    assert "前の質問" in prompt
    assert "前の回答" in prompt
    assert "今の質問" in prompt


def test_compose_prompt_history_appears_before_query(composer):
    """History context must appear before the current question"""
    history_text = "ユーザー: 過去\nアシスタント: 回答済み"
    prompt = composer.compose_prompt("現在", [], history_text=history_text)

    assert prompt.index("過去") < prompt.index("現在")


def test_compose_prompt_empty_history_omitted(composer):
    """An empty history string should not inject the history header"""
    prompt = composer.compose_prompt("質問", [], history_text="")

    assert "会話履歴" not in prompt


def test_compose_prompt_with_history_and_results(composer, mock_search_results):
    """History + search results should both appear in the prompt"""
    history_text = "ユーザー: 昨日の天気\nアシスタント: 昨日は晴れ"
    prompt = composer.compose_prompt("今日の天気", mock_search_results, history_text=history_text)

    assert "昨日の天気" in prompt
    assert "検索結果" in prompt
    assert "今日の天気" in prompt
