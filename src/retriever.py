#!/usr/bin/env python3
"""
Retriever module - Web search using SearXNG with optional ONNX Japanese reranking
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
    FETCH_PAGE_ENABLED,
    FETCH_PAGE_TOP_N,
    FETCH_PAGE_TIMEOUT,
    PASSAGE_CHARS,
    RERANK_MIN_SCORE,
    CONTEXT_CHAR_BUDGET,
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
        """Search web using local SearXNG, rerank, fetch pages, and re-rank passages."""
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
        return self._fetch_and_rerank_passages(query, ranked)

    def _fetch_page_text(self, url):
        """Fetch a single page and extract its main text via trafilatura.

        Returns the extracted text, or None on any failure.
        """
        try:
            validate_url(url, "page URL")
        except ValueError:
            logger.warning("Skipping non-http/https URL: %s", url)
            return None

        try:
            html = http_get_text(
                url,
                error_class=SearchError,
                service_name="Page fetch",
                timeout=FETCH_PAGE_TIMEOUT,
            )
        except SearchError as e:
            logger.warning("Page fetch failed for %s: %s", url, e)
            return None

        try:
            import trafilatura
        except ImportError:
            logger.warning("trafilatura not available; cannot extract page text")
            return None

        try:
            text = trafilatura.extract(html, include_comments=False, include_tables=False)
            if not text or not text.strip():
                logger.warning("trafilatura extracted empty text from %s", url)
                return None
            return text.strip()
        except Exception as e:
            logger.warning("trafilatura extraction failed for %s: %s", url, e)
            return None

    @staticmethod
    def _split_passages(text, char_limit=None):
        """Split text into passages of roughly char_limit characters.

        Prefers splitting on sentence-ending punctuation or newlines.
        """
        if char_limit is None:
            char_limit = PASSAGE_CHARS
        if len(text) <= char_limit:
            return [text]

        passages = []
        remaining = text
        while len(remaining) > char_limit:
            split_at = char_limit
            chunk = remaining[:char_limit]
            for sep in ("。", "！", "？", "\n", "、", " "):
                idx = chunk.rfind(sep)
                if idx > char_limit // 2:
                    split_at = idx + 1
                    break
            passages.append(remaining[:split_at].strip())
            remaining = remaining[split_at:].strip()
        if remaining:
            passages.append(remaining)
        return [p for p in passages if p]

    def _fetch_and_rerank_passages(self, query, ranked_results):
        """Fetch pages for top results, extract text, split into passages,
        and re-rank passages against the query.

        Failures are non-fatal: each failing URL degrades to its snippet,
        and a total second-stage failure returns the original results.
        """
        if not FETCH_PAGE_ENABLED or not ranked_results:
            return ranked_results

        top_n = ranked_results[:FETCH_PAGE_TOP_N]

        # Fetch pages in parallel
        url_to_text = {}
        with ThreadPoolExecutor(max_workers=FETCH_PAGE_TOP_N) as executor:
            future_to_result = {
                executor.submit(self._fetch_page_text, r["url"]): r
                for r in top_n
            }
            for future in as_completed(future_to_result):
                result = future_to_result[future]
                try:
                    text = future.result()
                except Exception as e:
                    logger.warning("Unexpected error fetching %s: %s", result["url"], e)
                    text = None
                url_to_text[result["url"]] = text

        # Build passages: extracted text → split, or fall back to snippet
        all_passages = []
        for result in top_n:
            text = url_to_text.get(result["url"])
            if text:
                passages = self._split_passages(text)
                for passage in passages:
                    all_passages.append({
                        "title": result["title"],
                        "content": passage,
                        "url": result["url"],
                    })
            else:
                all_passages.append({
                    "title": result["title"],
                    "content": result["content"],
                    "url": result["url"],
                })

        if not all_passages:
            return ranked_results

        # If no page text was extracted, skip the second rerank
        any_fetched = any(url_to_text.values())
        if not any_fetched:
            logger.info("No page text extracted; skipping passage re-rank")
            return ranked_results

        # Second-stage rerank: query vs passages
        try:
            scored = self._reranker.rerank(query, all_passages)
        except Exception:
            logger.warning(
                "Passage re-ranking failed; using extracted text as-is",
                exc_info=True,
            )
            scored = all_passages

        # Filter by RERANK_MIN_SCORE
        if RERANK_MIN_SCORE > 0.0 and self._reranker._model is not None:
            try:
                scored = self._filter_by_score(query, scored)
            except Exception:
                logger.warning(
                    "Score filtering failed; using unfiltered passages",
                    exc_info=True,
                )

        if not scored:
            logger.info("All passages dropped by RERANK_MIN_SCORE; returning empty")
            return []

        # Clip to CONTEXT_CHAR_BUDGET
        kept = []
        total_chars = 0
        for item in scored:
            item_chars = len(item.get("title", "")) + len(item.get("content", ""))
            if total_chars + item_chars > CONTEXT_CHAR_BUDGET and kept:
                break
            kept.append(item)
            total_chars += item_chars

        if len(kept) < len(scored):
            logger.warning(
                "Dropped %s passages to fit CONTEXT_CHAR_BUDGET (%s)",
                len(scored) - len(kept),
                CONTEXT_CHAR_BUDGET,
            )

        logger.info(
            "Page fetch: %d results → %d passages → %d kept",
            len(top_n),
            len(all_passages),
            len(kept),
        )
        return kept

    def _filter_by_score(self, query, results):
        """Filter results whose cross-encoder score is below RERANK_MIN_SCORE."""
        if not results:
            return results
        try:
            self._reranker._load()
            pairs = [
                (result.get("title", "") + " " + result.get("content", "")).strip()
                or result.get("url", "")
                for result in results
            ]
            inputs = self._reranker._tokenizer(
                [query] * len(pairs),
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="np",
            )
            outputs = self._reranker._model(**inputs)
            scores = self._reranker._scores(outputs.logits)
            logger.info("Passage scores: min=%.4f max=%.4f", min(scores), max(scores))
            return [
                result
                for score, result in zip(scores, results)
                if score >= RERANK_MIN_SCORE
            ]
        except Exception:
            raise
