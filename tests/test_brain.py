#!/usr/bin/env python3
"""
Tests for brain module
"""

import pytest
import requests
import importlib
from unittest.mock import Mock, patch
from src.brain import Brain, trim_messages_to_token_budget
from src.config import OLLAMA_SYSTEM_PROMPT
from src.exceptions import GenerationError


def _chat_response(content='テスト回答'):
    mock_response = Mock()
    mock_response.json.return_value = {'message': {'content': content}}
    mock_response.raise_for_status = Mock()
    return mock_response


@pytest.fixture
def brain():
    """Create Brain instance"""
    return Brain()


def test_brain_initialization(brain):
    """Test that Brain initializes correctly"""
    assert brain.ollama_api_url == "http://localhost:11434/api/chat"
    assert brain.ollama_model == "qwen2.5:3b"


def test_generate_response_success(brain):
    """Test successful AI response generation"""
    with patch('src.http_client.requests.post', return_value=_chat_response()):
        result = brain.generate_response("テストプロンプト")

    assert result == 'テスト回答'


def test_generate_response_sends_generation_parameters(brain):
    """Test that Ollama /api/chat receives messages, options, and keep_alive."""
    with patch('src.http_client.requests.post', return_value=_chat_response()) as post:
        brain.generate_response("テストプロンプト")

    payload = post.call_args.kwargs["json"]
    assert payload["messages"] == [
        {"role": "system", "content": OLLAMA_SYSTEM_PROMPT},
        {"role": "user", "content": "テストプロンプト"},
    ]
    assert "prompt" not in payload
    assert "system" not in payload
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
        with patch('src.http_client.requests.post', return_value=_chat_response()) as post:
            brain_module.Brain().generate_response("テストプロンプト")

        payload = post.call_args.kwargs["json"]
        assert payload["messages"][0]["content"] == values["OLLAMA_SYSTEM_PROMPT"]
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
    with patch('src.http_client.requests.post', return_value=_chat_response('')):
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


def test_generate_response_empty_messages(brain):
    """An empty messages list is treated as an empty question."""
    result = brain.generate_response([])
    assert "質問が空です" in result


def test_generate_auxiliary_uses_shorter_options_and_caller_messages(brain):
    """Auxiliary calls do not attach the spoken-answer system prompt."""
    with patch('src.http_client.requests.post', return_value=_chat_response()) as post:
        messages = [
            {"role": "system", "content": "検索準備"},
            {"role": "user", "content": "質問：こんにちは"},
        ]
        brain.generate_auxiliary(messages)

    payload = post.call_args.kwargs["json"]
    assert payload["messages"] == messages
    assert payload["options"]["num_predict"] == 64
    assert payload["options"]["temperature"] == 0.0


def test_generate_auxiliary_empty_messages_raises(brain):
    with pytest.raises(GenerationError, match="empty"):
        brain.generate_auxiliary([])



def test_generate_response_sends_messages_array(brain, mocker):
    """A messages list is forwarded to /api/chat as-is."""
    mock_post = mocker.patch(
        'src.http_client.requests.post', return_value=_chat_response()
    )
    messages = [
        {"role": "system", "content": "システム"},
        {"role": "user", "content": "質問"},
    ]

    brain.generate_response(messages)

    json_body = mock_post.call_args.kwargs['json']
    assert json_body['messages'] == messages
    assert json_body['model'] == 'qwen2.5:3b'
    assert json_body['stream'] is False


def test_generate_response_does_not_slice_last_user_at_10000(brain, mocker):
    """The old prompt[:10000] guard is gone; the last user message is kept."""
    mock_post = mocker.patch(
        'src.http_client.requests.post', return_value=_chat_response()
    )
    long_question = "あ" * 10001

    brain.generate_response([
        {"role": "system", "content": "S"},
        {"role": "user", "content": long_question},
    ])

    json_body = mock_post.call_args.kwargs['json']
    assert json_body['messages'][-1]['content'] == long_question


def test_trim_messages_drops_oldest_history_keeps_system_and_last_user():
    """Over-budget messages drop the oldest turns, never system or the last user."""
    messages = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "old-q" + "x" * 100},
        {"role": "assistant", "content": "old-a" + "y" * 100},
        {"role": "user", "content": "new-q" + "z" * 20},
        {"role": "assistant", "content": "new-a"},
        {"role": "user", "content": "current-question"},
    ]

    trimmed = trim_messages_to_token_budget(
        messages, token_limit=80, char_to_token_ratio=1.0
    )

    assert trimmed[0] == {"role": "system", "content": "S"}
    assert trimmed[-1] == {"role": "user", "content": "current-question"}
    assert all("old-q" not in (m.get("content") or "") for m in trimmed)
    assert any("new-q" in (m.get("content") or "") for m in trimmed)


def test_trim_messages_never_drops_system_or_last_user_even_if_over_budget():
    """System and the latest user stay even when they alone exceed the budget."""
    messages = [
        {"role": "system", "content": "S" * 50},
        {"role": "user", "content": "Q" * 50},
    ]

    trimmed = trim_messages_to_token_budget(
        messages, token_limit=10, char_to_token_ratio=1.0
    )

    assert trimmed == messages
