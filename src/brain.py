#!/usr/bin/env python3
"""
Brain module - AI response generation using Ollama
"""

import logging
import requests
import os
from urllib.parse import urlparse
from exceptions import GenerationError

logger = logging.getLogger(__name__)

# Configuration
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
_ALLOWED_SCHEMES = {"http", "https"}


def _validate_url(url, name):
    """Validate that a URL uses an allowed scheme."""
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(f"{name} must use http or https (got {parsed.scheme!r})")
    return url


class Brain:
    def __init__(self):
        print("Initializing Brain (Ollama)...")
        self.ollama_api_url = _validate_url(OLLAMA_API_URL, "OLLAMA_API_URL")
        self.ollama_model = OLLAMA_MODEL
        print("Brain initialized!")

    def generate_response(self, prompt):
        """Generate response using Ollama/Qwen 2.5 with search context"""
        if not isinstance(prompt, str) or not prompt.strip():
            return "申し訳ありませんが、質問が空です。"
        # Truncate excessively long prompts to prevent resource exhaustion
        prompt = prompt[:10000]

        print("Generating response with AI...")

        try:
            response = requests.post(
                self.ollama_api_url,
                json={
                    'model': self.ollama_model,
                    'prompt': prompt,
                    'stream': False
                },
                timeout=120
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.ConnectionError as e:
            raise GenerationError(
                f"Cannot connect to Ollama at {self.ollama_api_url}: {e}"
            ) from e
        except requests.exceptions.Timeout as e:
            raise GenerationError(
                f"Ollama request timed out after 120s: {e}"
            ) from e
        except requests.exceptions.HTTPError as e:
            raise GenerationError(
                f"Ollama returned an error (HTTP {response.status_code}): {e}"
            ) from e
        except (ValueError, requests.exceptions.JSONDecodeError) as e:
            raise GenerationError(
                f"Invalid JSON response from Ollama: {e}"
            ) from e

        answer = data.get('response', '').strip()
        if not answer:
            raise GenerationError("Ollama returned an empty response")

        logger.info("AI Response: %s", answer)
        return answer
