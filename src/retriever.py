#!/usr/bin/env python3
"""
Retriever module - Web search using SearXNG with optional ONNX Japanese reranking
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor

from .config import (
    SEARXNG_URL,
    SEARCH_CANDIDATE_LIMIT,
    RERANKER_ENABLED,
    RERANKER_MODEL_ID,
    RERANKER_LOCAL_PATH,
    RERANK_TOP_K,
    FETCH_PAGE_ENABLED,
    FETCH_PAGE_TOP_N,
    FETCH_PAGE_TIMEOUT,
    PASSAGE_CHARS,
    RERANK_MIN_SCORE,
    validate_url,
)
from .http_client import http_get_json, http_get_text
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
        return self._rerank(query, results)

    def rerank_without_limit(self, query, results):
        return self._rerank(query, results, top_k=0)

    def _rerank(self, query, results, top_k=None):
        if self._unavailable or not RERANKER_ENABLED or not results:
            return results
        if len(results) <= 1 and RERANK_MIN_SCORE <= 0.0:
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
            logger.info("Reranker scores: %s", [round(float(score), 4) for score in scores])

            ranked = sorted(
                zip(scores, results),
                key=lambda item: item[0],
                reverse=True,
            )

            filtered = [
                (score, result)
                for score, result in ranked
                if float(score) >= RERANK_MIN_SCORE
            ]

            if not filtered:
                logger.info(
                    "All reranked results were below score threshold %.3f",
                    RERANK_MIN_SCORE,
                )
                return []

            # 0 or negative TOP_K means sort only (no truncate).
            limit_setting = RERANK_TOP_K if top_k is None else top_k
            limit = limit_setting if limit_setting > 0 else len(filtered)
            top = [result for _, result in filtered[:limit]]
            logger.info(
                "Reranked %d results to top %d (threshold %.3f)",
                len(results),
                len(top),
                RERANK_MIN_SCORE,
            )
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
        """Search web using local SearXNG, then optionally rerank and fetch passages."""
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

        logger.info("Found %d raw results", len(results))
        reranked_results = self._reranker.rerank(query, results)

        if not FETCH_PAGE_ENABLED:
            return reranked_results

        return self._fetch_and_rerank_passages(query, reranked_results)

    def _fetch_and_rerank_passages(self, query, results):
        if not results:
            return []

        top_n = FETCH_PAGE_TOP_N if FETCH_PAGE_TOP_N > 0 else len(results)
        fetch_targets = results[:top_n]
        passage_candidates = list(results[top_n:])

        with ThreadPoolExecutor(max_workers=max(1, len(fetch_targets))) as executor:
            for candidates in executor.map(self._result_to_passages, fetch_targets):
                passage_candidates.extend(candidates)

        if not passage_candidates:
            return []

        reranked_passages = self._reranker.rerank_without_limit(query, passage_candidates)
        logger.info(
            "Passage enrichment built %d candidates and returned %d",
            len(passage_candidates),
            len(reranked_passages),
        )
        return reranked_passages

    def _result_to_passages(self, result):
        fallback = {
            'title': result.get('title', ''),
            'content': result.get('content', ''),
            'url': result.get('url', ''),
        }
        raw_url = fallback['url']
        if not raw_url:
            return [fallback]

        try:
            url = validate_url(raw_url, "search result URL")
        except ValueError:
            logger.warning("Skipping non-http(s) URL during page fetch: %s", raw_url)
            return [fallback]

        try:
            html = http_get_text(
                url,
                error_class=SearchError,
                service_name="PageFetch",
                timeout=FETCH_PAGE_TIMEOUT,
            )
        except SearchError:
            logger.warning("Page fetch failed; falling back to snippet for %s", url, exc_info=True)
            return [fallback]

        extracted = self._extract_page_text(html, url)
        if not extracted:
            return [fallback]

        passages = self._split_passages(extracted)
        if not passages:
            return [fallback]

        return [
            {
                'title': fallback['title'],
                'content': passage,
                'url': url,
            }
            for passage in passages
        ]

    @staticmethod
    def _extract_page_text(html, url):
        try:
            import trafilatura
        except ImportError:
            logger.warning("trafilatura is unavailable; using snippets", exc_info=True)
            return ""

        try:
            extracted = trafilatura.extract(html, output_format="txt")
        except Exception:
            logger.warning("Page extraction failed; using snippet for %s", url, exc_info=True)
            return ""

        if not extracted:
            logger.warning("Page extraction produced empty content; using snippet for %s", url)
            return ""

        lines = [line.strip() for line in extracted.splitlines() if line.strip()]
        return "\n".join(lines)

    @staticmethod
    def _split_passages(text):
        cleaned = text.strip()
        if not cleaned:
            return []

        max_chars = PASSAGE_CHARS if PASSAGE_CHARS > 0 else len(cleaned)
        passages = []
        start = 0

        while start < len(cleaned):
            end = min(start + max_chars, len(cleaned))
            if end < len(cleaned):
                sentence_break = cleaned.rfind("。", start, end)
                newline_break = cleaned.rfind("\n", start, end)
                split_at = max(sentence_break, newline_break)
                if split_at > start:
                    end = split_at + 1

            passage = cleaned[start:end].strip()
            if passage:
                passages.append(passage)

            if end <= start:
                end = min(start + max_chars, len(cleaned))
            start = end

        return passages
