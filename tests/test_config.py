#!/usr/bin/env python3
"""
Tests for centralized config module
"""

import pytest
from unittest.mock import patch
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
    assert config.DEBUG_AUDIO_DIR == ""
    assert isinstance(config.WAKE_WORDS, list)
    assert len(config.WAKE_WORDS) > 0
    assert config.WAKE_WORD_RECORD_SECONDS == 3
    assert config.SILENCE_THRESHOLD == 0.03


def test_config_reads_environment(monkeypatch):
    """Test that config respects environment variables"""
    monkeypatch.setenv("SEARXNG_URL", "http://custom:9090")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3:8b")

    import src.config
    importlib.reload(src.config)

    assert src.config.SEARXNG_URL == "http://custom:9090"
    assert src.config.OLLAMA_MODEL == "llama3:8b"

    # Reset
    monkeypatch.delenv("SEARXNG_URL")
    monkeypatch.delenv("OLLAMA_MODEL")
    importlib.reload(src.config)


def test_wake_words_config_from_env(monkeypatch):
    """WAKE_WORDS should be parsed from the WAKE_WORDS env variable."""
    monkeypatch.setenv("WAKE_WORDS", "ハロー,ねえ")

    import src.config
    importlib.reload(src.config)

    assert "ハロー" in src.config.WAKE_WORDS
    assert "ねえ" in src.config.WAKE_WORDS

    monkeypatch.delenv("WAKE_WORDS")
    importlib.reload(src.config)


def test_wake_word_record_seconds_from_env(monkeypatch):
    """WAKE_WORD_RECORD_SECONDS should be read from the environment."""
    monkeypatch.setenv("WAKE_WORD_RECORD_SECONDS", "5")

    import src.config
    importlib.reload(src.config)

    assert src.config.WAKE_WORD_RECORD_SECONDS == 5

    monkeypatch.delenv("WAKE_WORD_RECORD_SECONDS")
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
