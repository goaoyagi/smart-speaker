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

# Brain (Ollama) settings
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

# Speaker (Piper-Plus) settings
SPEAKER_DEVICE = os.getenv("SPEAKER_DEVICE", "plughw:0,0")
PIPER_MODEL_PATH = os.getenv("PIPER_MODEL_PATH", "./models/tsukuyomi.onnx")
PIPER_CONFIG_PATH = os.getenv("PIPER_CONFIG_PATH", "./models/config.json")

# Status LED (GPIO) settings
STATUS_LED_ENABLED = os.getenv("STATUS_LED_ENABLED", "true").lower() in ("1", "true", "yes")
STATUS_LED_PIN = int(os.getenv("STATUS_LED_PIN", "23"))

# Push-to-talk (GPIO button) settings
PUSH_TO_TALK_ENABLED = os.getenv("PUSH_TO_TALK_ENABLED", "true").lower() in ("1", "true", "yes")
PTT_BUTTON_PIN = int(os.getenv("PTT_BUTTON_PIN", "17"))
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
