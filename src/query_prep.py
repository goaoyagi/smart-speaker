#!/usr/bin/env python3
"""
Query preparation - search-need gating and follow-up query rewrite.

This is a pre-generation auxiliary LLM call. The result is never spoken.
Failures degrade to "search the original question".
"""

import logging
import re

from .audio_utils import log_init, log_ready
from .config import QUERY_PREP_ENABLED
from .exceptions import GenerationError

logger = logging.getLogger(__name__)

_PREP_SYSTEM_PROMPT = (
    "あなたは音声アシスタントの検索準備係です。発話はしません。"
    "次の形式だけを返してください。"
    "1行目: NEED または SKIP。"
    "2行目: 検索クエリ。SKIP のときは元の質問をそのまま書いてください。"
    "SKIP にするのは、挨拶、お礼、雑談、復唱の依頼、"
    "ウェブ検索では知り得ない個人の情報（飼い犬の名前、本人の予定など）です。"
    "事実、作り方、仕組みの説明、指示語を含む続きの質問は NEED です。"
    "続きの質問は、これまでの会話を使って単独で検索できるクエリに書き換えてください。"
    "例: 直前が東京タワーの高さで「それはいつ完成しましたか？」なら"
    "2行目は「東京タワー 完成年」です。"
)

_SKIP_MARKERS = ("SKIP", "NO_SEARCH", "NO SEARCH")
_SKIP_JA = ("不要", "検索しない", "検索不要")
_NEED_MARKERS = ("NEED", "SEARCH")
_QUERY_LABEL = re.compile(
    r"^(?:2行目[:：]?|検索クエリ|query|QUERY)[:：]?\s*",
    re.IGNORECASE,
)
_NEED_PREFIX = re.compile(r"^(?:NEED|SEARCH|要)\s+", re.IGNORECASE)


def parse_prepare_output(text, original_query):
    """Parse a NEED/SKIP reply into ``(should_search, search_query)``."""
    if not isinstance(text, str) or not text.strip():
        return True, original_query

    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    first = lines[0]
    first_upper = first.upper()

    should_search = True
    if any(marker in first_upper for marker in _SKIP_MARKERS) or any(
        marker in first for marker in _SKIP_JA
    ):
        should_search = False
    elif first_upper.startswith(_NEED_MARKERS) or first.startswith("要"):
        should_search = True

    if not should_search:
        return False, original_query

    search_query = original_query
    if len(lines) >= 2:
        candidate = _QUERY_LABEL.sub("", lines[1]).strip()
        if candidate:
            search_query = candidate
    else:
        rest = _NEED_PREFIX.sub("", first).strip()
        if rest and rest.upper() not in _NEED_MARKERS:
            search_query = rest

    return True, search_query


class QueryPrep:
    def __init__(self, brain):
        log_init("QueryPrep")
        self.brain = brain
        log_ready("QueryPrep")

    def prepare(self, query, history_messages=None):
        """Return ``(should_search, search_query)``. Never raises."""
        if not QUERY_PREP_ENABLED:
            return True, query

        user_content = f"質問：{query}"
        messages = [
            {"role": "system", "content": _PREP_SYSTEM_PROMPT},
        ]
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": user_content})

        try:
            raw = self.brain.generate_auxiliary(messages)
        except GenerationError as error:
            logger.warning(
                "Query prep failed, searching the original question: %s",
                error,
            )
            return True, query

        should_search, search_query = parse_prepare_output(raw, query)
        if should_search:
            logger.info("Search needed; query=%s", search_query)
        else:
            logger.info("Search skipped for: %s", query)
        return should_search, search_query
