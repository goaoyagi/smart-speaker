#!/usr/bin/env python3
"""
Retriever module - Web search using SearXNG
"""

import requests
import os

# Configuration
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8080")


class Retriever:
    def __init__(self):
        print("Initializing Retriever (SearXNG)...")
        self.searxng_url = SEARXNG_URL
        print("Retriever initialized!")
    
    def search_web(self, query):
        """Search web using local SearXNG"""
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
