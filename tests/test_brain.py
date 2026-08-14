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


def test_generate_response_sends_options_system_and_keep_alive(brain, mocker):
    """Ollama /api/generate receives options, system, and keep_alive."""
    mock_response = Mock()
    mock_response.json.return_value = {'response': 'テスト回答'}
    mock_response.raise_for_status = Mock()
    mock_post = mocker.patch('src.http_client.requests.post', return_value=mock_response)

    brain.generate_response("テストプロンプト")

    json_body = mock_post.call_args.kwargs['json']
    assert json_body['model'] == 'qwen2.5:3b'
    assert json_body['prompt'] == 'テストプロンプト'
    assert json_body['stream'] is False
    assert json_body['keep_alive'] == '30m'
    assert json_body['system'] == (
        "あなたは日本語専用の音声アシスタントです。"
        "回答はすべて日本語のみで行い、アルファベット（英語の単語や文）を含めてはいけません。"
        "必要であればカタカナや日本語表現に翻訳して出力してください。"
    )
    assert json_body['options'] == {
        'num_ctx': 8192,
        'num_predict': 512,
        'temperature': 0.3,
        'repeat_penalty': 1.1,
    }


def test_generate_response_honors_env_overrides(monkeypatch, mocker):
    """Generation parameters can be overridden via environment variables."""
    monkeypatch.setenv("OLLAMA_NUM_CTX", "4096")
    monkeypatch.setenv("OLLAMA_NUM_PREDICT", "256")
    monkeypatch.setenv("OLLAMA_TEMPERATURE", "0.7")
    monkeypatch.setenv("OLLAMA_REPEAT_PENALTY", "1.3")
    monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "10m")
    monkeypatch.setenv("OLLAMA_SYSTEM_PROMPT", "カスタム指示")

    import importlib
    import src.brain as brain_mod
    import src.config as config_mod
    importlib.reload(config_mod)
    importlib.reload(brain_mod)

    mock_response = Mock()
    mock_response.json.return_value = {'response': 'テスト回答'}
    mock_response.raise_for_status = Mock()
    mock_post = mocker.patch('src.http_client.requests.post', return_value=mock_response)

    try:
        brain_mod.Brain().generate_response("テストプロンプト")
        json_body = mock_post.call_args.kwargs['json']
        assert json_body['system'] == "カスタム指示"
        assert json_body['keep_alive'] == "10m"
        assert json_body['options'] == {
            'num_ctx': 4096,
            'num_predict': 256,
            'temperature': 0.7,
            'repeat_penalty': 1.3,
        }
    finally:
        monkeypatch.delenv("OLLAMA_NUM_CTX", raising=False)
        monkeypatch.delenv("OLLAMA_NUM_PREDICT", raising=False)
        monkeypatch.delenv("OLLAMA_TEMPERATURE", raising=False)
        monkeypatch.delenv("OLLAMA_REPEAT_PENALTY", raising=False)
        monkeypatch.delenv("OLLAMA_KEEP_ALIVE", raising=False)
        monkeypatch.delenv("OLLAMA_SYSTEM_PROMPT", raising=False)
        importlib.reload(config_mod)
        importlib.reload(brain_mod)
