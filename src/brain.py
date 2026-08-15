#!/usr/bin/env python3
"""
Brain module - AI response generation using Ollama
"""

import logging

from .config import (
    CHAR_TO_TOKEN_RATIO,
    OLLAMA_API_URL,
    OLLAMA_AUX_NUM_PREDICT,
    OLLAMA_AUX_TEMPERATURE,
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


def estimate_message_tokens(messages, char_to_token_ratio=None):
    """Estimate tokens from total characters (conservative; Japanese ≈ 1:1)."""
    ratio = CHAR_TO_TOKEN_RATIO if char_to_token_ratio is None else char_to_token_ratio
    total_chars = sum(len(message.get("content") or "") for message in messages)
    return total_chars * ratio


def trim_messages_to_token_budget(messages, token_limit=None, char_to_token_ratio=None):
    """Drop oldest history turns until estimated tokens fit the context window.

    The system message and the last user message are never dropped.
    """
    if token_limit is None:
        token_limit = OLLAMA_NUM_CTX - OLLAMA_NUM_PREDICT

    messages = [dict(message) for message in messages]
    if not messages:
        return messages

    system = []
    rest = messages
    if messages[0].get("role") == "system":
        system = [messages[0]]
        rest = messages[1:]

    last = []
    middle = rest
    if rest:
        last = [rest[-1]]
        middle = rest[:-1]

    dropped = False
    while estimate_message_tokens(
        system + middle + last, char_to_token_ratio
    ) > token_limit and middle:
        drop = 2 if len(middle) >= 2 else 1
        middle = middle[drop:]
        dropped = True

    if dropped:
        logger.warning(
            "Dropped oldest conversation turns to fit the token budget (%s)",
            token_limit,
        )

    return system + middle + last


class Brain:
    def __init__(self):
        log_init("Brain (Ollama)")
        self.ollama_api_url = validate_url(OLLAMA_API_URL, "OLLAMA_API_URL")
        self.ollama_model = OLLAMA_MODEL
        log_ready("Brain")

    def generate_response(self, prompt):
        """Generate a response with Ollama /api/chat.

        ``prompt`` is a messages list on the main path. A string is still
        accepted so ``compose_prompt`` can be wired back from ``main.py``.
        """
        if isinstance(prompt, list):
            if not prompt:
                return "申し訳ありませんが、質問が空です。"
            messages = prompt
        else:
            if not isinstance(prompt, str) or not prompt.strip():
                return "申し訳ありませんが、質問が空です。"
            messages = [
                {"role": "system", "content": OLLAMA_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]

        logger.info("Generating response with AI...")
        answer = self._chat(
            messages,
            num_predict=OLLAMA_NUM_PREDICT,
            temperature=OLLAMA_TEMPERATURE,
            timeout=120,
        )
        logger.info("AI Response: %s", answer)
        return answer

    def generate_auxiliary(self, messages, num_predict=None, temperature=None):
        """Short non-spoken LLM call for search gating and query rewrite.

        Does not attach ``OLLAMA_SYSTEM_PROMPT``. Callers must pass their own
        messages and must not speak the result.
        """
        if not messages:
            raise GenerationError("Auxiliary prompt is empty")
        logger.info("Running auxiliary LLM call...")
        return self._chat(
            messages,
            num_predict=OLLAMA_AUX_NUM_PREDICT if num_predict is None else num_predict,
            temperature=(
                OLLAMA_AUX_TEMPERATURE if temperature is None else temperature
            ),
            timeout=30,
        )

    def _chat(self, messages, *, num_predict, temperature, timeout):
        """POST /api/chat and return the assistant content."""
        messages = trim_messages_to_token_budget(messages)
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
                    'num_predict': num_predict,
                    'temperature': temperature,
                    'repeat_penalty': OLLAMA_REPEAT_PENALTY,
                },
            },
            timeout=timeout,
        )
        answer = (data.get('message') or {}).get('content', '').strip()
        if not answer:
            raise GenerationError("Ollama returned an empty response")
        return answer
