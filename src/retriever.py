#!/usr/bin/env python3
"""
Retriever module - Web search using SearXNG with optional ONNX Japanese reranking
"""

import logging
import os

import numpy as np

from .config import (
    SEARXNG_URL,
    RERANKER_ENABLED,
    RERANKER_MODEL_ID,
    RERANKER_LOCAL_PATH,
    RERANK_TOP_K,
    validate_url,
)
from .http_client import http_get_json
from .audio_utils import log_init, log_ready
from .exceptions import SearchError

logger = logging.getLogger(__name__)

# How many SearXNG hits to fetch before reranking (then truncated to RERANK_TOP_K).
_SEARCH_CANDIDATE_LIMIT = 10


class Retriever:
    def __init__(self):
        log_init("Retriever (SearXNG)")
        self.searxng_url = validate_url(SEARXNG_URL, "SEARXNG_URL")
        self.reranker_enabled = RERANKER_ENABLED
        self.rerank_top_k = max(1, RERANK_TOP_K)
        self._reranker_model = None
        self._reranker_tokenizer = None
        self._reranker_load_attempted = False
        self._reranker_available = False
        log_ready("Retriever")

    def search_web(self, query):
        """Search web using local SearXNG, then optionally rerank results."""
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

        # Fetch extra candidates when reranking; keep the historical top-5 otherwise.
        candidate_limit = (
            _SEARCH_CANDIDATE_LIMIT if self.reranker_enabled else 5
        )
        results = []
        for result in data.get('results', [])[:candidate_limit]:
            results.append({
                'title': result.get('title', ''),
                'content': result.get('content', ''),
                'url': result.get('url', '')
            })

        logger.info("Found %d results", len(results))

        if self.reranker_enabled and results:
            results = self._rerank(query, results)

        return results

    def _rerank(self, query, results):
        """Rerank search results by relevance; on failure return the original list."""
        try:
            if not self._ensure_reranker():
                return results

            passages = [
                f"{r.get('title', '')} {r.get('content', '')}".strip() or r.get('url', '')
                for r in results
            ]
            scores = self._score_pairs(query, passages)
            ranked = sorted(
                zip(scores, results),
                key=lambda item: item[0],
                reverse=True,
            )
            top = [item[1] for item in ranked[: self.rerank_top_k]]
            logger.info(
                "Reranked %d results to top %d",
                len(results),
                len(top),
            )
            return top
        except Exception as e:
            logger.warning("Reranking failed; using original search order: %s", e)
            return results

    def _ensure_reranker(self):
        """Lazily load the ONNX reranker. Returns True when ready to score."""
        if self._reranker_available:
            return True
        if self._reranker_load_attempted:
            return False

        self._reranker_load_attempted = True
        try:
            from transformers import AutoTokenizer  # dynamic import
            from optimum.onnxruntime import ORTModelForSequenceClassification  # dynamic import
        except ImportError as e:
            logger.warning(
                "Reranker dependencies unavailable (optimum/transformers); skipping: %s",
                e,
            )
            return False

        model_source, from_local = self._resolve_model_source()
        try:
            self._reranker_tokenizer = AutoTokenizer.from_pretrained(model_source)
            # Local optimum-cli exports are already ONNX; Hub ids need on-the-fly export.
            load_kwargs = {} if from_local else {"export": True}
            self._reranker_model = ORTModelForSequenceClassification.from_pretrained(
                model_source,
                **load_kwargs,
            )
            self._reranker_available = True
            logger.info("Reranker loaded from %s", model_source)
            return True
        except Exception as e:
            logger.warning("Failed to load reranker from %s: %s", model_source, e)
            self._reranker_model = None
            self._reranker_tokenizer = None
            return False

    def _resolve_model_source(self):
        """Prefer a local ONNX export when present; otherwise use the Hub model id.

        Returns ``(source, from_local)``.
        """
        local_path = RERANKER_LOCAL_PATH
        if local_path and os.path.isdir(local_path) and os.listdir(local_path):
            return local_path, True
        return RERANKER_MODEL_ID, False

    def _score_pairs(self, query, passages):
        """Score (query, passage) pairs with the cross-encoder; higher is more relevant."""
        inputs = self._reranker_tokenizer(
            [(query, passage) for passage in passages],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
        )
        outputs = self._reranker_model(**inputs)
        logits = np.asarray(outputs.logits).reshape(-1)
        return logits.tolist()
