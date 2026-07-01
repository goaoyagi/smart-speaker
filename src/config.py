#!/usr/bin/env python3
"""
Centralized configuration - All environment variables in one place.

Eliminates duplicated os.getenv() calls scattered across modules.
Also provides shared URL validation (previously duplicated in retriever and brain).
"""

import os
from urllib.parse import urlparse

# Listener (Whisper) settings
MIC_DEVICE = os.getenv("MIC_DEVICE", "hw:0,0")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
CHANNELS = int(os.getenv("CHANNELS", "1"))
RECORD_SECONDS = int(os.getenv("RECORD_SECONDS", "10"))
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")
DEBUG_AUDIO_DIR = os.getenv("DEBUG_AUDIO_DIR", "")

# Retriever (SearXNG) settings
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8080")

# Brain (Ollama) settings
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

# Speaker (Piper-Plus) settings
SPEAKER_DEVICE = os.getenv("SPEAKER_DEVICE", "plughw:0,0")
PIPER_MODEL_PATH = os.getenv("PIPER_MODEL_PATH", "./models/tsukuyomi.onnx")
PIPER_CONFIG_PATH = os.getenv("PIPER_CONFIG_PATH", "./models/config.json")

# Conversation history (multi-turn dialogue) settings
CONVERSATION_MAX_TURNS = int(os.getenv("CONVERSATION_MAX_TURNS", "3"))
CONVERSATION_ANSWER_CLIP = int(os.getenv("CONVERSATION_ANSWER_CLIP", "200"))

# URL validation
_ALLOWED_SCHEMES = {"http", "https"}


def validate_url(url, name):
    """Validate that a URL uses an allowed scheme."""
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(f"{name} must use http or https (got {parsed.scheme!r})")
    return url
