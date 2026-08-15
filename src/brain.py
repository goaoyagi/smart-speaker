#!/usr/bin/env python3
"""
Brain module - AI response generation using Ollama
"""

import logging

from .config import (
    OLLAMA_API_URL,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
    OLLAMA_REPEAT_PENALTY,
    OLLAMA_SYSTEM_PROMPT,
    OLLAMA_TEMPERATURE,
    validate_url,
)
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
        """Generate a response through the legacy single-prompt interface."""
        if not isinstance(prompt, str) or not prompt.strip():
            return "申し訳ありませんが、質問が空です。"
        return self.generate_response_messages([
            {'role': 'system', 'content': OLLAMA_SYSTEM_PROMPT},
            {'role': 'user', 'content': prompt},
        ])

    def generate_response_messages(self, messages):
        """Generate a response using Ollama's chat messages interface."""
        if not isinstance(messages, list) or not messages:
            return "申し訳ありませんが、質問が空です。"

        logger.info("Generating response with AI...")

        data = http_post_json(
            self.ollama_api_url,
            error_class=GenerationError,
            service_name="Ollama",
            json_body={
                'model': self.ollama_model,
                'messages': messages,
                'stream': False,
                'keep_alive': OLLAMA_KEEP_ALIVE,
                'options': {
                    'num_ctx': OLLAMA_NUM_CTX,
                    'num_predict': OLLAMA_NUM_PREDICT,
                    'temperature': OLLAMA_TEMPERATURE,
                    'repeat_penalty': OLLAMA_REPEAT_PENALTY,
                },
            },
            timeout=120
        )

        message = data.get('message', {})
        answer = message.get('content', '').strip() if isinstance(message, dict) else ''
        if not answer:
            raise GenerationError("Ollama returned an empty response")

        logger.info("AI Response: %s", answer)
        return answer
