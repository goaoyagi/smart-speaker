#!/usr/bin/env python3
"""
Tests for brain module
"""

import pytest
import requests
import importlib
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


def test_generate_response_sends_generation_parameters(brain):
    """Test that Ollama receives the configured generation parameters."""
    mock_response = Mock()
    mock_response.json.return_value = {'response': 'テスト回答'}
    mock_response.raise_for_status = Mock()

    with patch('src.http_client.requests.post', return_value=mock_response) as post:
        brain.generate_response("テストプロンプト")

    payload = post.call_args.kwargs["json"]
    assert payload["system"] == (
        "あなたは日本語専用の音声アシスタントです。"
        "回答はすべて日本語のみで行い、アルファベット（英語の単語や文）を含めてはいけません。"
        "必要であればカタカナや日本語表現に翻訳して出力してください。"
    )
    assert payload["keep_alive"] == "30m"
    assert payload["options"] == {
        "num_ctx": 8192,
        "num_predict": 512,
        "temperature": 0.3,
        "repeat_penalty": 1.1,
    }


def test_generate_response_uses_environment_generation_parameters(monkeypatch):
    """Test that generation parameters can be overridden by environment variables."""
    values = {
        "OLLAMA_NUM_CTX": "4096",
        "OLLAMA_NUM_PREDICT": "256",
        "OLLAMA_TEMPERATURE": "0.7",
        "OLLAMA_REPEAT_PENALTY": "1.25",
        "OLLAMA_KEEP_ALIVE": "5m",
        "OLLAMA_SYSTEM_PROMPT": "日本語だけで答えてください。",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)

    import src.brain as brain_module
    import src.config as config_module
    importlib.reload(config_module)
    importlib.reload(brain_module)

    try:
        mock_response = Mock()
        mock_response.json.return_value = {'response': 'テスト回答'}
        mock_response.raise_for_status = Mock()

        with patch('src.http_client.requests.post', return_value=mock_response) as post:
            brain_module.Brain().generate_response("テストプロンプト")

        payload = post.call_args.kwargs["json"]
        assert payload["system"] == values["OLLAMA_SYSTEM_PROMPT"]
        assert payload["keep_alive"] == values["OLLAMA_KEEP_ALIVE"]
        assert payload["options"] == {
            "num_ctx": 4096,
            "num_predict": 256,
            "temperature": 0.7,
            "repeat_penalty": 1.25,
        }
    finally:
        for key in values:
            monkeypatch.delenv(key)
        importlib.reload(config_module)
        importlib.reload(brain_module)


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
