#!/usr/bin/env python3
"""
Retriever module - Web search using SearXNG
"""

import logging

from .config import SEARXNG_URL, validate_url
from .http_client import http_get_json
from .audio_utils import log_init, log_ready
from .exceptions import SearchError

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self):
        log_init("Retriever (SearXNG)")
        self.searxng_url = validate_url(SEARXNG_URL, "SEARXNG_URL")
        log_ready("Retriever")

    def search_web(self, query):
        """Search web using local SearXNG"""
        if not isinstance(query, str) or not query.strip():
            print("Empty or invalid query")
            return []
        query = query.strip()[:500]

        print(f"Searching web for: {query}")

        params = {
            'q': query,
            'format': 'json',
            'language': 'ja'
        }

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        data = http_get_json(
            f"{self.searxng_url}/search",
            error_class=SearchError,
            service_name="SearXNG",
            params=params,
            headers=headers,
            timeout=10
        )

        results = []
        for result in data.get('results', [])[:5]:
            results.append({
                'title': result.get('title', ''),
                'content': result.get('content', ''),
                'url': result.get('url', '')
            })

        logger.info("Found %d results", len(results))
        return results
