#!/usr/bin/env python3
"""
Brain module - AI response generation using Ollama
"""

import logging

from .config import OLLAMA_API_URL, OLLAMA_MODEL, validate_url
from .http_client import http_post_json
from .audio_utils import log_init, log_ready
from .exceptions import GenerationError

logger = logging.getLogger(__name__)


class Brain:
    def __init__(self):
        log_init("Brain (Ollama)")
        self.ollama_api_url = validate_url(OLLAMA_API_URL, "OLLAMA_API_URL")
        self.ollama_model = OLLAMA_MODEL
        log_ready("Brain")

    def generate_response(self, prompt):
        """Generate response using Ollama/Qwen 2.5 with search context"""
        if not isinstance(prompt, str) or not prompt.strip():
            return "申し訳ありませんが、質問が空です。"
        prompt = prompt[:10000]

        print("Generating response with AI...")

        data = http_post_json(
            self.ollama_api_url,
            error_class=GenerationError,
            service_name="Ollama",
            json_body={
                'model': self.ollama_model,
                'prompt': prompt,
                'stream': False
            },
            timeout=120
        )

        answer = data.get('response', '').strip()
        if not answer:
            raise GenerationError("Ollama returned an empty response")

        logger.info("AI Response: %s", answer)
        return answer
