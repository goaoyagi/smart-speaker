#!/usr/bin/env python3
"""
Centralized configuration - All environment variables in one place.

Eliminates duplicated os.getenv() calls scattered across modules.
Also provides shared URL validation (previously duplicated in retriever and brain).
"""

import os
from urllib.parse import urlparse

import sys

# Load .env file manually from the project root directory, unless running unit tests
is_testing = "pytest" in sys.modules or any("pytest" in arg for arg in sys.argv)
if not is_testing:
    _env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(_env_path):
        with open(_env_path, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line or _line.startswith("#"):
                    continue
                if "=" in _line:
                    _key, _val = _line.split("=", 1)
                    _key = _key.strip()
                    _val = _val.strip()
                    if _val.startswith(('"', "'")) and _val.endswith(_val[0]):
                        _val = _val[1:-1]
                    if _key not in os.environ:
                        os.environ[_key] = _val




# Listener (Whisper) settings
MIC_DEVICE = os.getenv("MIC_DEVICE", "hw:0,0")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
CHANNELS = int(os.getenv("CHANNELS", "1"))
RECORD_SECONDS = int(os.getenv("RECORD_SECONDS", "10"))
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")
DEBUG_AUDIO_DIR = os.getenv("DEBUG_AUDIO_DIR", "")

# Retriever (SearXNG) settings
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8080")
# How many SearXNG hits to fetch before reranking (then truncated to RERANK_TOP_K).
SEARCH_CANDIDATE_LIMIT = int(os.getenv("SEARCH_CANDIDATE_LIMIT", "10"))

# Reranker (Optimum/ONNX) settings — enabled by default as part of the standard RAG flow
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "true").lower() in ("1", "true", "yes")
RERANKER_MODEL_ID = os.getenv(
    "RERANKER_MODEL_ID",
    "hotchpotch/japanese-reranker-cross-encoder-xsmall-v1",
)
RERANKER_LOCAL_PATH = os.getenv("RERANKER_LOCAL_PATH", "./models/reranker")
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "3"))
RERANK_MIN_SCORE = float(os.getenv("RERANK_MIN_SCORE", "0.0"))

# Page fetch + passage reranking settings (retriever.py)
FETCH_PAGE_ENABLED = os.getenv("FETCH_PAGE_ENABLED", "true").lower() in ("1", "true", "yes")
FETCH_PAGE_TOP_N = int(os.getenv("FETCH_PAGE_TOP_N", "3"))
FETCH_PAGE_TIMEOUT = int(os.getenv("FETCH_PAGE_TIMEOUT", "3"))
PASSAGE_CHARS = int(os.getenv("PASSAGE_CHARS", "250"))

# Brain (Ollama) settings
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
# Brain (Ollama) generation settings
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "512"))
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.3"))
OLLAMA_REPEAT_PENALTY = float(os.getenv("OLLAMA_REPEAT_PENALTY", "1.1"))
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
OLLAMA_SYSTEM_PROMPT = os.getenv(
    "OLLAMA_SYSTEM_PROMPT",
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
    "検索結果を使って答える場合は、検索結果に書かれていないことを推測で補ってはいけません。",
)
OLLAMA_AUX_NUM_PREDICT = int(os.getenv("OLLAMA_AUX_NUM_PREDICT", "64"))
OLLAMA_AUX_TEMPERATURE = float(os.getenv("OLLAMA_AUX_TEMPERATURE", "0.0"))
QUERY_PREP_ENABLED = os.getenv("QUERY_PREP_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
)
CONTEXT_CHAR_BUDGET = int(os.getenv("CONTEXT_CHAR_BUDGET", "2000"))
CHAR_TO_TOKEN_RATIO = float(os.getenv("CHAR_TO_TOKEN_RATIO", "1.0"))

# Speaker (Piper-Plus) settings
SPEAKER_DEVICE = os.getenv("SPEAKER_DEVICE", "plughw:0,0")
PIPER_MODEL_PATH = os.getenv("PIPER_MODEL_PATH", "./models/piper/tsukuyomi.onnx")
PIPER_CONFIG_PATH = os.getenv("PIPER_CONFIG_PATH", "./models/piper/tsukuyomi.onnx.json")

# Conversation history (multi-turn dialogue) settings
CONVERSATION_MAX_TURNS = int(os.getenv("CONVERSATION_MAX_TURNS", "5"))
CONVERSATION_ANSWER_CLIP = int(os.getenv("CONVERSATION_ANSWER_CLIP", "400"))

# Status LED (GPIO) settings
STATUS_LED_ENABLED = os.getenv("STATUS_LED_ENABLED", "true").lower() in ("1", "true", "yes")
STATUS_LED_PIN = int(os.getenv("STATUS_LED_PIN", "23"))

# Push-to-talk (GPIO button) settings
PUSH_TO_TALK_ENABLED = os.getenv("PUSH_TO_TALK_ENABLED", "true").lower() in ("1", "true", "yes")
PTT_BUTTON_PIN = int(os.getenv("PTT_BUTTON_PIN", "24"))
PTT_BOUNCE_TIME = float(os.getenv("PTT_BOUNCE_TIME", "0.05"))
PTT_MIN_RECORD_SECONDS = float(os.getenv("PTT_MIN_RECORD_SECONDS", "0.5"))
PTT_MAX_RECORD_SECONDS = float(os.getenv("PTT_MAX_RECORD_SECONDS", "30"))

# URL validation
_ALLOWED_SCHEMES = {"http", "https"}


def validate_url(url, name):
    """Validate that a URL uses an allowed scheme."""
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(f"{name} must use http or https (got {parsed.scheme!r})")
    return url
