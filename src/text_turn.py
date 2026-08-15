#!/usr/bin/env python3
"""
Text-in / text-out conversation turn shared by main.py and eval.py.

Order: search-need / query rewrite → search → compose → generate → speech
normalization. The final spoken answer is generated once.
"""

import logging
import time

from .exceptions import SearchError
from .speech_normalize import normalize_for_speech

logger = logging.getLogger(__name__)


def run_text_turn(
    query,
    history_messages,
    *,
    retriever,
    composer,
    brain,
    query_prep,
    use_search=True,
    on_phase=None,
):
    """Prepare search, retrieve, generate, and normalize for speech.

    ``on_phase`` is called with ``prep``, ``search``, or ``generate`` so the
    caller can update LEDs. Search failures degrade to empty context.
    Generation failures propagate.
    """

    def phase(name):
        if on_phase:
            on_phase(name)

    result = {
        "answer": None,
        "search_results": [],
        "search_query": query,
        "skipped_search": False,
        "prep_ms": 0.0,
        "search_ms": 0.0,
        "generate_ms": 0.0,
        "degraded": False,
        "degrade_reason": None,
    }

    should_search = use_search
    search_query = query
    if use_search:
        phase("prep")
        started = time.perf_counter()
        should_search, search_query = query_prep.prepare(query, history_messages)
        result["prep_ms"] = (time.perf_counter() - started) * 1000
        result["search_query"] = search_query
        result["skipped_search"] = not should_search

    search_results = []
    if should_search:
        phase("search")
        started = time.perf_counter()
        try:
            search_results = retriever.search_web(search_query)
        except SearchError as error:
            logger.warning("Search failed, proceeding without context: %s", error)
            search_results = []
            result["degraded"] = True
            result["degrade_reason"] = str(error)
        result["search_ms"] = (time.perf_counter() - started) * 1000
    result["search_results"] = search_results

    phase("generate")
    messages = composer.compose_messages(query, search_results, history_messages)
    started = time.perf_counter()
    answer = brain.generate_response(messages)
    result["generate_ms"] = (time.perf_counter() - started) * 1000
    result["answer"] = normalize_for_speech(answer)
    return result
