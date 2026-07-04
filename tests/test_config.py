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
    assert config.OLLAMA_API_URL == "http://localhost:11434/api/generate"
    assert config.OLLAMA_MODEL == "qwen2.5:3b"
    assert config.SPEAKER_DEVICE == "plughw:0,0"
    assert config.PIPER_MODEL_PATH == "./models/tsukuyomi.onnx"
    assert config.PIPER_CONFIG_PATH == "./models/config.json"
    assert config.STATUS_LED_ENABLED is True
    assert config.STATUS_LED_PIN == 17
    assert config.DEBUG_AUDIO_DIR == ""


def test_config_reads_environment(monkeypatch):
    """Test that config respects environment variables"""
    monkeypatch.setenv("SEARXNG_URL", "http://custom:9090")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3:8b")
    monkeypatch.setenv("STATUS_LED_ENABLED", "false")
    monkeypatch.setenv("STATUS_LED_PIN", "21")

    import src.config
    importlib.reload(src.config)

    assert src.config.SEARXNG_URL == "http://custom:9090"
    assert src.config.OLLAMA_MODEL == "llama3:8b"
    assert src.config.STATUS_LED_ENABLED is False
    assert src.config.STATUS_LED_PIN == 21

    # Reset
    monkeypatch.delenv("SEARXNG_URL")
    monkeypatch.delenv("OLLAMA_MODEL")
    monkeypatch.delenv("STATUS_LED_ENABLED")
    monkeypatch.delenv("STATUS_LED_PIN")
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
