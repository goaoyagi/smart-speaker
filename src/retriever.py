#!/usr/bin/env python3
"""
Retriever module - Web search using SearXNG
"""

import logging
import os

from .config import (
    RERANKER_ENABLED,
    RERANKER_LOCAL_PATH,
    RERANKER_MODEL_ID,
    RERANK_TOP_K,
    SEARXNG_URL,
    validate_url,
)
from .http_client import http_get_json
from .audio_utils import log_init, log_ready
from .exceptions import SearchError

logger = logging.getLogger(__name__)


class _Reranker:
    """Lazy-loaded ONNX reranker with a permanent failure fallback."""

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._unavailable = False

    def rerank(self, query, results):
        if self._unavailable or not RERANKER_ENABLED or len(results) <= 1:
            return results

        try:
            self._load()
            pairs = [result.get("title", "") + " " + result.get("content", "") for result in results]
            inputs = self._tokenizer(
                [query] * len(pairs),
                pairs,
                padding=True,
                truncation=True,
                return_tensors="np",
            )
            outputs = self._model(**inputs)
            scores = self._scores(outputs.logits)
            ranked = sorted(zip(scores, results), key=lambda item: item[0], reverse=True)
            limit = RERANK_TOP_K if RERANK_TOP_K > 0 else len(ranked)
            return [result for _, result in ranked[:limit]]
        except Exception:
            self._unavailable = True
            logger.warning("Reranking failed; using unranked search results", exc_info=True)
            return results

    def _load(self):
        if self._unavailable:
            raise RuntimeError("Reranker is unavailable")
        if self._model is not None and self._tokenizer is not None:
            return

        try:
            from optimum.onnxruntime import ORTModelForSequenceClassification
            from transformers import AutoTokenizer

            model_path = RERANKER_LOCAL_PATH if os.path.isdir(RERANKER_LOCAL_PATH) else RERANKER_MODEL_ID
            self._tokenizer = AutoTokenizer.from_pretrained(model_path)
            self._model = ORTModelForSequenceClassification.from_pretrained(model_path)
        except Exception:
            self._unavailable = True
            raise

    @staticmethod
    def _scores(logits):
        if hasattr(logits, "detach"):
            logits = logits.detach().cpu().numpy()
        if hasattr(logits, "tolist"):
            logits = logits.tolist()

        scores = []
        for row in logits:
            if isinstance(row, (list, tuple)):
                scores.append(row[-1])
            else:
                scores.append(row)
        return scores


class Retriever:
    def __init__(self):
        log_init("Retriever (SearXNG)")
        self.searxng_url = validate_url(SEARXNG_URL, "SEARXNG_URL")
        self._reranker = _Reranker()
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
        return self._reranker.rerank(query, results)
