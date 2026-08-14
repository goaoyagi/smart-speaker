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
    assert config.SEARCH_CANDIDATE_LIMIT == 10
    assert config.RERANKER_ENABLED is True
    assert config.RERANKER_MODEL_ID == (
        "hotchpotch/japanese-reranker-cross-encoder-xsmall-v1"
    )
    assert config.RERANKER_LOCAL_PATH == "./models/reranker"
    assert config.RERANK_TOP_K == 3
    assert config.OLLAMA_API_URL == "http://localhost:11434/api/generate"
    assert config.OLLAMA_MODEL == "qwen2.5:3b"
    assert config.OLLAMA_NUM_CTX == 8192
    assert config.OLLAMA_NUM_PREDICT == 512
    assert config.OLLAMA_TEMPERATURE == 0.3
    assert config.OLLAMA_REPEAT_PENALTY == 1.1
    assert config.OLLAMA_KEEP_ALIVE == "30m"
    assert config.OLLAMA_SYSTEM_PROMPT == (
        "あなたは日本語専用の音声アシスタントです。"
        "回答はすべて日本語のみで行い、アルファベット（英語の単語や文）を含めてはいけません。"
        "必要であればカタカナや日本語表現に翻訳して出力してください。"
    )
    assert config.SPEAKER_DEVICE == "plughw:0,0"
    assert config.PIPER_MODEL_PATH == "./models/piper/tsukuyomi.onnx"
    assert config.PIPER_CONFIG_PATH == "./models/piper/tsukuyomi.onnx.json"
    assert config.STATUS_LED_ENABLED is True
    assert config.STATUS_LED_PIN == 23
    assert config.PUSH_TO_TALK_ENABLED is True
    assert config.PTT_BUTTON_PIN == 24
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
    monkeypatch.setenv("SEARCH_CANDIDATE_LIMIT", "7")
    monkeypatch.setenv("OLLAMA_NUM_CTX", "4096")
    monkeypatch.setenv("OLLAMA_NUM_PREDICT", "256")
    monkeypatch.setenv("OLLAMA_TEMPERATURE", "0.7")
    monkeypatch.setenv("OLLAMA_REPEAT_PENALTY", "1.3")
    monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "10m")
    monkeypatch.setenv("OLLAMA_SYSTEM_PROMPT", "カスタム指示")

    import src.config
    importlib.reload(src.config)

    assert src.config.SEARXNG_URL == "http://custom:9090"
    assert src.config.OLLAMA_MODEL == "llama3:8b"
    assert src.config.OLLAMA_NUM_CTX == 4096
    assert src.config.OLLAMA_NUM_PREDICT == 256
    assert src.config.OLLAMA_TEMPERATURE == 0.7
    assert src.config.OLLAMA_REPEAT_PENALTY == 1.3
    assert src.config.OLLAMA_KEEP_ALIVE == "10m"
    assert src.config.OLLAMA_SYSTEM_PROMPT == "カスタム指示"
    assert src.config.STATUS_LED_ENABLED is False
    assert src.config.STATUS_LED_PIN == 21
    assert src.config.PUSH_TO_TALK_ENABLED is False
    assert src.config.PTT_BUTTON_PIN == 27
    assert src.config.PTT_MAX_RECORD_SECONDS == 45
    assert src.config.RERANKER_ENABLED is False
    assert src.config.RERANK_TOP_K == 5
    assert src.config.RERANKER_LOCAL_PATH == "./models/custom-reranker"
    assert src.config.SEARCH_CANDIDATE_LIMIT == 7

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
    monkeypatch.delenv("SEARCH_CANDIDATE_LIMIT")
    monkeypatch.delenv("OLLAMA_NUM_CTX")
    monkeypatch.delenv("OLLAMA_NUM_PREDICT")
    monkeypatch.delenv("OLLAMA_TEMPERATURE")
    monkeypatch.delenv("OLLAMA_REPEAT_PENALTY")
    monkeypatch.delenv("OLLAMA_KEEP_ALIVE")
    monkeypatch.delenv("OLLAMA_SYSTEM_PROMPT")
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
