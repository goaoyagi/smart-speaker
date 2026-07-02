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


def test_compose_prompt_with_history_summary_and_results(composer, mock_search_results):
    """History summary is embedded into the prompt when provided alongside search results"""
    query = "それについて詳しく教えて"
    summary = "ユーザー：前の質問\nアシスタント：前の回答"
    prompt = composer.compose_prompt(query, mock_search_results, history_summary=summary)

    assert "会話履歴" in prompt
    assert "前の質問" in prompt
    assert "前の回答" in prompt
    assert "検索結果" in prompt
    assert query in prompt


def test_compose_prompt_with_history_summary_no_results(composer):
    """History summary is embedded even without search results"""
    query = "それはどういうこと？"
    summary = "ユーザー：初めての質問\nアシスタント：初めての回答"
    prompt = composer.compose_prompt(query, [], history_summary=summary)

    assert "会話履歴" in prompt
    assert "初めての質問" in prompt
    assert query in prompt
    assert "検索結果" not in prompt


def test_compose_prompt_empty_history_summary_omitted(composer):
    """An empty history_summary must not add a history block to the prompt"""
    query = "今日の天気"
    prompt = composer.compose_prompt(query, [], history_summary="")

    assert "会話履歴" not in prompt
    assert query in prompt

