#!/usr/bin/env python3
"""
Tests for brain module
"""

import pytest
import requests
from unittest.mock import Mock, patch
from src.brain import Brain
from src.exceptions import GenerationError


@pytest.fixture
def brain():
    """Create Brain instance"""
    return Brain()


def test_brain_initialization(brain):
    """Test that Brain initializes correctly"""
    assert brain.ollama_api_url == "http://localhost:11434/api/generate"
    assert brain.ollama_model == "qwen2.5:3b"


def test_generate_response_success(brain):
    """Test successful AI response generation"""
    mock_response = Mock()
    mock_response.json.return_value = {'response': 'テスト回答'}
    mock_response.raise_for_status = Mock()

    with patch('src.http_client.requests.post', return_value=mock_response):
        result = brain.generate_response("テストプロンプト")

    assert result == 'テスト回答'


def test_generate_response_connection_error(brain):
    """Test AI response generation raises GenerationError on connection failure"""
    with patch('src.http_client.requests.post',
               side_effect=requests.exceptions.ConnectionError("Connection refused")):
        with pytest.raises(GenerationError, match="Cannot connect to Ollama"):
            brain.generate_response("テストプロンプト")


def test_generate_response_empty(brain):
    """Test AI response generation raises GenerationError on empty response"""
    mock_response = Mock()
    mock_response.json.return_value = {'response': ''}
    mock_response.raise_for_status = Mock()

    with patch('src.http_client.requests.post', return_value=mock_response):
        with pytest.raises(GenerationError, match="empty response"):
            brain.generate_response("テストプロンプト")


def test_generate_response_empty_prompt(brain):
    """Test AI response generation with empty prompt"""
    result = brain.generate_response("")
    assert "質問が空です" in result


def test_generate_response_none_prompt(brain):
    """Test AI response generation with None prompt"""
    result = brain.generate_response(None)
    assert "質問が空です" in result


def test_summarize_history_success(brain):
    """Test successful history summarization"""
    mock_response = Mock()
    mock_response.json.return_value = {'response': '要約されたテキスト'}
    mock_response.raise_for_status = Mock()

    with patch('src.http_client.requests.post', return_value=mock_response):
        result = brain.summarize_history("ユーザーは天気を質問し晴れと回答された。")

    assert result == '要約されたテキスト'


def test_summarize_history_empty_input(brain):
    """Empty history returns empty string without calling Ollama"""
    with patch('src.http_client.requests.post') as mock_post:
        result = brain.summarize_history("")
    assert result == ""
    mock_post.assert_not_called()


def test_summarize_history_empty_response(brain):
    """Empty summary from Ollama raises GenerationError"""
    mock_response = Mock()
    mock_response.json.return_value = {'response': ''}
    mock_response.raise_for_status = Mock()

    with patch('src.http_client.requests.post', return_value=mock_response):
        with pytest.raises(GenerationError, match="empty summary"):
            brain.summarize_history("何らかの履歴")
