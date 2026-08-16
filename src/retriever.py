#!/usr/bin/env python3
"""
Retriever module - Web search using SearXNG with optional ONNX Japanese reranking,
page-body fetch, and passage rerank.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import (
    SEARXNG_URL,
    SEARCH_CANDIDATE_LIMIT,
    RERANKER_ENABLED,
    RERANKER_MODEL_ID,
    RERANKER_LOCAL_PATH,
    RERANK_TOP_K,
    RERANK_MIN_SCORE,
    FETCH_PAGE_ENABLED,
    FETCH_PAGE_TOP_N,
    FETCH_PAGE_TIMEOUT,
    PASSAGE_CHARS,
    validate_url,
)
from .http_client import http_get_json, http_get_text
from .audio_utils import log_init, log_ready
from .exceptions import SearchError

logger = logging.getLogger(__name__)

_PAGE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}

_SENTENCE_SEPS = ("。", "！", "？", "\n")


def split_passages(text, max_chars):
    """Split text into passages of about ``max_chars``, preferring sentence breaks."""
    text = (text or "").strip()
    if not text or max_chars <= 0:
        return []
    if len(text) <= max_chars:
        return [text]

    passages = []
    start = 0
    length = len(text)
    while start < length:
        while start < length and text[start].isspace():
            start += 1
        if start >= length:
            break
        end = min(start + max_chars, length)
        if end < length:
            window = text[start:end]
            break_at = max(window.rfind(sep) for sep in _SENTENCE_SEPS)
            if break_at >= 0:
                end = start + break_at + 1
        chunk = text[start:end].strip()
        if chunk:
            passages.append(chunk)
        start = end
    return passages


def extract_page_text(html):
    """Extract main text from HTML via trafilatura. Empty string on failure."""
    if not html or not isinstance(html, str):
        return ""
    try:
        import trafilatura
    except ImportError:
        logger.warning("trafilatura is not installed; keeping search snippets")
        return ""
    try:
        text = trafilatura.extract(html)
    except Exception:
        logger.warning("trafilatura extract failed", exc_info=True)
        return ""
    return (text or "").strip()


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
            texts = [
                (result.get("title", "") + " " + result.get("content", "")).strip()
                or result.get("url", "")
                for result in results
            ]
            ranked = self._score_and_sort(query, results, texts)
            kept = [(score, result) for score, result in ranked if score >= RERANK_MIN_SCORE]
            if not kept:
                logger.info(
                    "All %d results below RERANK_MIN_SCORE=%s",
                    len(ranked),
                    RERANK_MIN_SCORE,
                )
                return []
            # 0 or negative TOP_K means sort only (no truncate).
            limit = RERANK_TOP_K if RERANK_TOP_K > 0 else len(kept)
            top = [result for _, result in kept[:limit]]
            logger.info("Reranked %d results to top %d", len(results), len(top))
            return top
        except Exception:
            self._unavailable = True
            logger.warning(
                "Reranking failed; using unranked search results",
                exc_info=True,
            )
            return results

    def rerank_passages(self, query, passages):
        """Rerank passages by query. Raises on inference failure so the caller can degrade.

        Does not latch ``_unavailable``; result rerank already succeeded on this turn.
        """
        if self._unavailable or not RERANKER_ENABLED or len(passages) <= 1:
            return passages

        texts = [passage.get("content", "") for passage in passages]
        ranked = self._score_and_sort(query, passages, texts)
        kept = [(score, passage) for score, passage in ranked if score >= RERANK_MIN_SCORE]
        if not kept:
            logger.info(
                "All %d passages below RERANK_MIN_SCORE=%s",
                len(ranked),
                RERANK_MIN_SCORE,
            )
            return []
        return [passage for _, passage in kept]

    def _score_and_sort(self, query, items, texts):
        """Score query-text pairs and return ``(score, item)`` sorted descending."""
        self._load()
        inputs = self._tokenizer(
            [query] * len(texts),
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
        )
        outputs = self._model(**inputs)
        scores = self._scores(outputs.logits)
        logger.info("Reranker scores: %s", scores)
        return sorted(
            zip(scores, items),
            key=lambda item: item[0],
            reverse=True,
        )

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
        ranked = self._reranker.rerank(query, results)
        if not FETCH_PAGE_ENABLED or not ranked:
            return ranked
        pages = self._fetch_pages(ranked)
        return self._enrich_with_passages(query, pages)

    def _fetch_pages(self, results):
        """Fetch top URLs in parallel and replace snippets with extracted text."""
        top_n = FETCH_PAGE_TOP_N if FETCH_PAGE_TOP_N > 0 else len(results)
        targets = results[:top_n]
        fetched = [None] * len(targets)
        workers = min(len(targets), top_n) or 1
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_idx = {
                pool.submit(self._fetch_one_page, item): idx
                for idx, item in enumerate(targets)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    fetched[idx] = future.result()
                except Exception:
                    logger.warning(
                        "Page fetch worker failed; keeping snippet",
                        exc_info=True,
                    )
                    fetched[idx] = targets[idx]
        return fetched

    def _fetch_one_page(self, result):
        """Fetch one URL. On any failure, return the original snippet result."""
        url = result.get("url") or ""
        try:
            validate_url(url, "page URL")
        except ValueError:
            logger.warning("Skipping page fetch for non-http URL: %s", url)
            return result
        try:
            html = http_get_text(
                url,
                error_class=SearchError,
                service_name="page",
                headers=_PAGE_HEADERS,
                timeout=FETCH_PAGE_TIMEOUT,
            )
        except SearchError as exc:
            logger.warning("Page fetch failed for %s; keeping snippet: %s", url, exc)
            return result
        text = extract_page_text(html)
        if not text:
            logger.warning("Empty page extract for %s; keeping snippet", url)
            return result
        updated = dict(result)
        updated["content"] = text
        logger.info("Extracted %d chars from %s", len(text), url)
        return updated

    def _enrich_with_passages(self, query, pages):
        """Split fetched text into passages and rerank. Degrade to page text on failure."""
        if self._reranker._unavailable or not RERANKER_ENABLED:
            return pages
        try:
            passages = []
            for page in pages:
                chunks = split_passages(page.get("content", ""), PASSAGE_CHARS)
                if not chunks:
                    passages.append(page)
                    continue
                for chunk in chunks:
                    passages.append({
                        "title": page.get("title", ""),
                        "content": chunk,
                        "url": page.get("url", ""),
                    })
            if not passages:
                return pages
            return self._reranker.rerank_passages(query, passages)
        except Exception:
            logger.warning(
                "Passage rerank failed; using fetched page text",
                exc_info=True,
            )
            return pages
