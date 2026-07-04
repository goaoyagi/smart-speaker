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


def test_compose_prompt_no_history_context_by_default(composer):
    """Omitting history_context should produce the same output as passing empty string."""
    query = "今日の天気"
    prompt_default = composer.compose_prompt(query, [])
    prompt_empty = composer.compose_prompt(query, [], "")
    assert prompt_default == prompt_empty


def test_compose_prompt_includes_history_context(composer):
    """Non-empty history_context should appear in the prompt."""
    context = "ユーザーは昨日の天気について質問した。"
    prompt = composer.compose_prompt("今日は？", [], context)

    assert "会話の要約" in prompt
    assert context in prompt


def test_compose_prompt_history_context_precedes_question(composer):
    """History context block should appear before the current question."""
    context = "過去の要約テキスト"
    prompt = composer.compose_prompt("今の質問", [], context)

    pos_context = prompt.index(context)
    pos_question = prompt.index("今の質問")
    assert pos_context < pos_question


def test_compose_prompt_with_results_and_history_context(composer, mock_search_results):
    """History context and search results should both appear in the prompt."""
    context = "ユーザーはスポーツについて話していた。"
    prompt = composer.compose_prompt("今日の天気", mock_search_results, context)

    assert "会話の要約" in prompt
    assert context in prompt
    assert "検索結果" in prompt
    assert "今日の天気" in prompt
