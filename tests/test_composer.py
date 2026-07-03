#!/usr/bin/env python3
"""
Tests for composer module
"""

import pytest
from src.composer import Composer
from src.conversation_history import ConversationHistory


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


def test_compose_prompt_with_empty_history(composer):
    """Empty history should produce same output as no history argument."""
    query = "今日の天気"
    history = ConversationHistory()
    prompt_no_history = composer.compose_prompt(query, [])
    prompt_empty_history = composer.compose_prompt(query, [], history)
    assert prompt_no_history == prompt_empty_history


def test_compose_prompt_includes_history_block(composer):
    """Non-empty history should appear in the prompt."""
    history = ConversationHistory()
    history.add_turn("昨日の天気は？", "曇りでした。")

    prompt = composer.compose_prompt("今日は？", [], history)

    assert "会話履歴" in prompt
    assert "昨日の天気は？" in prompt
    assert "曇りでした。" in prompt


def test_compose_prompt_history_precedes_question(composer):
    """History block should appear before the current question."""
    history = ConversationHistory()
    history.add_turn("前の質問", "前の回答")

    prompt = composer.compose_prompt("今の質問", [], history)

    pos_history = prompt.index("前の質問")
    pos_question = prompt.index("今の質問")
    assert pos_history < pos_question


def test_compose_prompt_with_results_and_history(composer, mock_search_results):
    """History should be present alongside search results."""
    history = ConversationHistory()
    history.add_turn("以前の質問", "以前の回答")

    prompt = composer.compose_prompt("今日の天気", mock_search_results, history)

    assert "会話履歴" in prompt
    assert "以前の質問" in prompt
    assert "検索結果" in prompt
    assert "今日の天気" in prompt
