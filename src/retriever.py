#!/usr/bin/env python3
"""
Retriever module - Web search using SearXNG with optional ONNX Japanese reranking
"""

import logging
import os

from .config import (
    SEARXNG_URL,
    SEARCH_CANDIDATE_LIMIT,
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
            pairs = [
                (result.get("title", "") + " " + result.get("content", "")).strip()
                or result.get("url", "")
                for result in results
            ]
            inputs = self._tokenizer(
                [query] * len(pairs),
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="np",
            )
            outputs = self._model(**inputs)
            scores = self._scores(outputs.logits)
            ranked = sorted(
                zip(scores, results),
                key=lambda item: item[0],
                reverse=True,
            )
            # 0 or negative TOP_K means sort only (no truncate).
            limit = RERANK_TOP_K if RERANK_TOP_K > 0 else len(ranked)
            top = [result for _, result in ranked[:limit]]
            logger.info("Reranked %d results to top %d", len(results), len(top))
            return top
        except Exception:
            self._unavailable = True
            logger.warning(
                "Reranking failed; using unranked search results",
                exc_info=True,
            )
            return results

    def _load(self):
        if self._unavailable:
            raise RuntimeError("Reranker is unavailable")
        if self._model is not None and self._tokenizer is not None:
            return

        try:
            # Dynamic imports: missing deps must not break process startup.
            from optimum.onnxruntime import ORTModelForSequenceClassification
            from transformers import AutoTokenizer

            model_source, from_local = self._resolve_model_source()
            self._tokenizer = AutoTokenizer.from_pretrained(model_source)
            # Local optimum-cli exports are already ONNX; Hub ids need on-the-fly export.
            load_kwargs = {} if from_local else {"export": True}
            self._model = ORTModelForSequenceClassification.from_pretrained(
                model_source,
                **load_kwargs,
            )
            logger.info("Reranker loaded from %s", model_source)
        except Exception:
            self._unavailable = True
            raise

    @staticmethod
    def _resolve_model_source():
        """Prefer a local ONNX export when present; otherwise use the Hub model id.

        Returns ``(source, from_local)``.
        """
        local_path = RERANKER_LOCAL_PATH
        if local_path and os.path.isdir(local_path) and os.listdir(local_path):
            return local_path, True
        return RERANKER_MODEL_ID, False

    @staticmethod
    def _scores(logits):
        """Normalize cross-encoder logits of varying shapes to a flat score list.

        Accepts torch tensors, numpy arrays, and nested lists such as (n,),
        (n, 1), or (n, 2) — for multi-logit rows the last column is used.
        """
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
        """Search web using local SearXNG, then optionally rerank results."""
        if not isinstance(query, str) or not query.strip():
            logger.warning("Empty or invalid query")
            return []
        query = query.strip()[:500]

        logger.info("Searching web for: %s", query)

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
        for result in data.get('results', [])[:SEARCH_CANDIDATE_LIMIT]:
            results.append({
                'title': result.get('title', ''),
                'content': result.get('content', ''),
                'url': result.get('url', '')
            })

        logger.info("Found %d results", len(results))
        return self._reranker.rerank(query, results)
