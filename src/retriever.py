#!/usr/bin/env python3
"""
Retriever module - Web search using SearXNG
"""

import requests
import os
from urllib.parse import urlparse

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
            
            # Extract search results
            results = []
            for result in data.get('results', [])[:5]:  # Top 5 results
                results.append({
                    'title': result.get('title', ''),
                    'content': result.get('content', ''),
                    'url': result.get('url', '')
                })
            
            print(f"Found {len(results)} results")
            return results
            
        except Exception as e:
            print(f"Search error: {e}")
            return []
