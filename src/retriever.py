#!/usr/bin/env python3
"""
Retriever module - Web search using SearXNG
"""

import logging
import requests
import os
from urllib.parse import urlparse
from exceptions import SearchError

logger = logging.getLogger(__name__)

# Configuration
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8080")
_ALLOWED_SCHEMES = {"http", "https"}


def _validate_url(url, name):
    """Validate that a URL uses an allowed scheme."""
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(f"{name} must use http or https (got {parsed.scheme!r})")
    return url


class Retriever:
    def __init__(self):
        print("Initializing Retriever (SearXNG)...")
        self.searxng_url = _validate_url(SEARXNG_URL, "SEARXNG_URL")
        print("Retriever initialized!")

    def search_web(self, query):
        """Search web using local SearXNG"""
        # Validate and truncate query to prevent abuse
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

        try:
            response = requests.get(
                f"{self.searxng_url}/search",
                params=params,
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.ConnectionError as e:
            raise SearchError(
                f"Cannot connect to SearXNG at {self.searxng_url}: {e}"
            ) from e
        except requests.exceptions.Timeout as e:
            raise SearchError(
                f"SearXNG request timed out: {e}"
            ) from e
        except requests.exceptions.HTTPError as e:
            raise SearchError(
                f"SearXNG returned an error (HTTP {response.status_code}): {e}"
            ) from e
        except (ValueError, requests.exceptions.JSONDecodeError) as e:
            raise SearchError(
                f"Invalid JSON response from SearXNG: {e}"
            ) from e

        # Extract search results
        results = []
        for result in data.get('results', [])[:5]:
            results.append({
                'title': result.get('title', ''),
                'content': result.get('content', ''),
                'url': result.get('url', '')
            })

        logger.info("Found %d results", len(results))
        return results
