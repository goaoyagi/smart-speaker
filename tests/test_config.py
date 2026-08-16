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
    assert config.RERANK_MIN_SCORE == 0.0
    assert config.FETCH_PAGE_ENABLED is True
    assert config.FETCH_PAGE_TOP_N == 3
    assert config.FETCH_PAGE_TIMEOUT == 3
    assert config.PASSAGE_CHARS == 250
    assert config.OLLAMA_API_URL == "http://localhost:11434/api/chat"
    assert config.OLLAMA_MODEL == "qwen2.5:3b"
    assert config.OLLAMA_NUM_CTX == 8192
    assert config.OLLAMA_NUM_PREDICT == 512
    assert config.OLLAMA_TEMPERATURE == 0.3
    assert config.OLLAMA_REPEAT_PENALTY == 1.1
    assert config.OLLAMA_KEEP_ALIVE == "30m"
    assert config.OLLAMA_SYSTEM_PROMPT == (
        "あなたは日本語専用の音声アシスタントです。"
        "最終的に発話する回答は日本語のみで行い、英語の単語や文、アルファベットの羅列は含めないでください。"
        "単位や略語はカタカナか日本語で書いてください。"
        "結論を先に述べ、文数は質問に合わせてください。"
        "事実や数値は1文で足りれば1文で止め、多くても3文までにしてください。"
        "挨拶やお礼は1〜2文で答えてください。"
        "分からないことは推測せず、「分かりません」と1〜2文で答えてください。"
        "仕組みや手順の説明だけ、3〜5文で具体的に説明してください。"
        "「それ」「さっきの」などの指示語は、これまでの会話を参照して解釈してください。"
        "ユーザーが話題を変えたら、前の話題には触れずに新しい質問だけに答えてください。"
        "検索の有無や「検索結果」という言い方は発話に出さないでください。"
        "検索結果が質問と無関係な場合は使わず、これまでの会話とあなたの知識で答えてください。"
        "検索結果を使って答える場合は、検索結果に書かれていないことを推測で補ってはいけません。"
    )
    assert config.OLLAMA_AUX_NUM_PREDICT == 64
    assert config.OLLAMA_AUX_TEMPERATURE == 0.0
    assert config.QUERY_PREP_ENABLED is True
    assert config.CONTEXT_CHAR_BUDGET == 2000
    assert config.CHAR_TO_TOKEN_RATIO == 1.0
    assert config.CONVERSATION_MAX_TURNS == 5
    assert config.CONVERSATION_ANSWER_CLIP == 400
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
    monkeypatch.setenv("RERANK_MIN_SCORE", "0.4")
    monkeypatch.setenv("FETCH_PAGE_ENABLED", "false")
    monkeypatch.setenv("FETCH_PAGE_TOP_N", "2")
    monkeypatch.setenv("FETCH_PAGE_TIMEOUT", "5")
    monkeypatch.setenv("PASSAGE_CHARS", "300")
    monkeypatch.setenv("RERANKER_LOCAL_PATH", "./models/custom-reranker")
    monkeypatch.setenv("SEARCH_CANDIDATE_LIMIT", "7")
    monkeypatch.setenv("CONTEXT_CHAR_BUDGET", "1500")
    monkeypatch.setenv("CHAR_TO_TOKEN_RATIO", "0.7")
    monkeypatch.setenv("CONVERSATION_MAX_TURNS", "4")
    monkeypatch.setenv("CONVERSATION_ANSWER_CLIP", "300")
    monkeypatch.setenv("QUERY_PREP_ENABLED", "false")
    monkeypatch.setenv("OLLAMA_AUX_NUM_PREDICT", "32")
    monkeypatch.setenv("OLLAMA_AUX_TEMPERATURE", "0.2")

    import src.config
    importlib.reload(src.config)

    assert src.config.SEARXNG_URL == "http://custom:9090"
    assert src.config.OLLAMA_MODEL == "llama3:8b"
    assert src.config.CONTEXT_CHAR_BUDGET == 1500
    assert src.config.CHAR_TO_TOKEN_RATIO == 0.7
    assert src.config.CONVERSATION_MAX_TURNS == 4
    assert src.config.CONVERSATION_ANSWER_CLIP == 300
    assert src.config.QUERY_PREP_ENABLED is False
    assert src.config.OLLAMA_AUX_NUM_PREDICT == 32
    assert src.config.OLLAMA_AUX_TEMPERATURE == 0.2
    assert src.config.STATUS_LED_ENABLED is False
    assert src.config.STATUS_LED_PIN == 21
    assert src.config.PUSH_TO_TALK_ENABLED is False
    assert src.config.PTT_BUTTON_PIN == 27
    assert src.config.PTT_MAX_RECORD_SECONDS == 45
    assert src.config.RERANKER_ENABLED is False
    assert src.config.RERANK_TOP_K == 5
    assert src.config.RERANK_MIN_SCORE == 0.4
    assert src.config.FETCH_PAGE_ENABLED is False
    assert src.config.FETCH_PAGE_TOP_N == 2
    assert src.config.FETCH_PAGE_TIMEOUT == 5
    assert src.config.PASSAGE_CHARS == 300
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
    monkeypatch.delenv("RERANK_MIN_SCORE")
    monkeypatch.delenv("FETCH_PAGE_ENABLED")
    monkeypatch.delenv("FETCH_PAGE_TOP_N")
    monkeypatch.delenv("FETCH_PAGE_TIMEOUT")
    monkeypatch.delenv("PASSAGE_CHARS")
    monkeypatch.delenv("RERANKER_LOCAL_PATH")
    monkeypatch.delenv("SEARCH_CANDIDATE_LIMIT")
    monkeypatch.delenv("CONTEXT_CHAR_BUDGET")
    monkeypatch.delenv("CHAR_TO_TOKEN_RATIO")
    monkeypatch.delenv("CONVERSATION_MAX_TURNS")
    monkeypatch.delenv("CONVERSATION_ANSWER_CLIP")
    monkeypatch.delenv("QUERY_PREP_ENABLED")
    monkeypatch.delenv("OLLAMA_AUX_NUM_PREDICT")
    monkeypatch.delenv("OLLAMA_AUX_TEMPERATURE")
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
