#!/usr/bin/env python3
"""
Tests for centralized config module
"""

import pytest
import importlib


def test_default_config_values():
    """Test that config loads correct defaults"""
    import src.config as config

    assert config.MIC_DEVICE == "hw:0,0"
    assert config.SAMPLE_RATE == 16000
    assert config.CHANNELS == 1
    assert config.RECORD_SECONDS == 10
    assert config.WHISPER_MODEL_SIZE == "small"
    assert config.SEARXNG_URL == "http://localhost:8080"
    assert config.RERANKER_ENABLED is True
    assert config.RERANKER_MODEL_ID == (
        "hotchpotch/japanese-reranker-cross-encoder-xsmall-v1"
    )
    assert config.RERANKER_LOCAL_PATH == "./models/reranker"
    assert config.RERANK_TOP_K == 3
    assert config.OLLAMA_API_URL == "http://localhost:11434/api/generate"
    assert config.OLLAMA_MODEL == "qwen2.5:3b"
    assert config.SPEAKER_DEVICE == "plughw:0,0"
    assert config.PIPER_MODEL_PATH == "./models/tsukuyomi.onnx"
    assert config.PIPER_CONFIG_PATH == "./models/config.json"
    assert config.STATUS_LED_ENABLED is True
    assert config.STATUS_LED_PIN == 23
    assert config.PUSH_TO_TALK_ENABLED is True
    assert config.PTT_BUTTON_PIN == 17
    assert config.PTT_BOUNCE_TIME == 0.05
    assert config.PTT_MIN_RECORD_SECONDS == 0.5
    assert config.PTT_MAX_RECORD_SECONDS == 30
    assert config.DEBUG_AUDIO_DIR == ""


def test_config_reads_environment(monkeypatch):
    """Test that config respects environment variables"""
    monkeypatch.setenv("SEARXNG_URL", "http://custom:9090")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3:8b")
    monkeypatch.setenv("STATUS_LED_ENABLED", "false")
    monkeypatch.setenv("STATUS_LED_PIN", "21")
    monkeypatch.setenv("PUSH_TO_TALK_ENABLED", "false")
    monkeypatch.setenv("PTT_BUTTON_PIN", "27")
    monkeypatch.setenv("PTT_MAX_RECORD_SECONDS", "45")
    monkeypatch.setenv("RERANKER_ENABLED", "false")
    monkeypatch.setenv("RERANK_TOP_K", "5")
    monkeypatch.setenv("RERANKER_LOCAL_PATH", "./models/custom-reranker")

    import src.config
    importlib.reload(src.config)

    assert src.config.SEARXNG_URL == "http://custom:9090"
    assert src.config.OLLAMA_MODEL == "llama3:8b"
    assert src.config.STATUS_LED_ENABLED is False
    assert src.config.STATUS_LED_PIN == 21
    assert src.config.PUSH_TO_TALK_ENABLED is False
    assert src.config.PTT_BUTTON_PIN == 27
    assert src.config.PTT_MAX_RECORD_SECONDS == 45
    assert src.config.RERANKER_ENABLED is False
    assert src.config.RERANK_TOP_K == 5
    assert src.config.RERANKER_LOCAL_PATH == "./models/custom-reranker"

    # Reset
    monkeypatch.delenv("SEARXNG_URL")
    monkeypatch.delenv("OLLAMA_MODEL")
    monkeypatch.delenv("STATUS_LED_ENABLED")
    monkeypatch.delenv("STATUS_LED_PIN")
    monkeypatch.delenv("PUSH_TO_TALK_ENABLED")
    monkeypatch.delenv("PTT_BUTTON_PIN")
    monkeypatch.delenv("PTT_MAX_RECORD_SECONDS")
    monkeypatch.delenv("RERANKER_ENABLED")
    monkeypatch.delenv("RERANK_TOP_K")
    monkeypatch.delenv("RERANKER_LOCAL_PATH")
    importlib.reload(src.config)


def test_validate_url_valid():
    """Test validate_url with valid URLs"""
    from src.config import validate_url

    assert validate_url("http://localhost:8080", "test") == "http://localhost:8080"
    assert validate_url("https://example.com", "test") == "https://example.com"


def test_validate_url_invalid():
    """Test validate_url with invalid scheme"""
    from src.config import validate_url

    with pytest.raises(ValueError, match="must use http or https"):
        validate_url("ftp://example.com", "test")
