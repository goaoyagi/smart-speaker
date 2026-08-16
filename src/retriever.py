#!/usr/bin/env python3
"""
Retriever module - Web search using SearXNG with optional ONNX Japanese reranking
and page content extraction.
"""

import concurrent.futures
import logging
import os
import re

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
from .http_client import http_get_json, http_get_html
from .audio_utils import log_init, log_ready
from .exceptions import SearchError

logger = logging.getLogger(__name__)


def split_into_passages(text, max_chars=PASSAGE_CHARS):
    """Split text into passages of approximately max_chars characters.

    Prioritizes paragraph and sentence boundaries (newlines, punctuation).
    """
    if not text or not isinstance(text, str):
        return []

    text = text.strip()
    if not text:
        return []

    if max_chars <= 0:
        max_chars = 250

    if len(text) <= max_chars:
        return [text]

    paragraphs = [p.strip() for p in re.split(r'\n+', text) if p.strip()]
    if not paragraphs:
        return [text[:max_chars]]

    passages = []
    current_passage = ""

    for para in paragraphs:
        sentences = [s.strip() for s in re.split(r'(?<=[。！？!?\n])\s*', para) if s.strip()]
        if not sentences:
            sentences = [para]

        for sentence in sentences:
            if not sentence:
                continue
            if len(sentence) > max_chars:
                if current_passage:
                    passages.append(current_passage)
                    current_passage = ""
                for i in range(0, len(sentence), max_chars):
                    chunk = sentence[i:i + max_chars].strip()
                    if chunk:
                        passages.append(chunk)
                continue

            if not current_passage:
                current_passage = sentence
            elif len(current_passage) + len(sentence) <= max_chars:
                current_passage += sentence
            else:
                passages.append(current_passage)
                current_passage = sentence

    if current_passage:
        passages.append(current_passage)

    return [p for p in passages if p]


class _Reranker:
    """Lazy-loaded ONNX reranker with a permanent failure fallback."""

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._unavailable = False

    def rerank(self, query, results, top_k=None, min_score=None):
        if top_k is None:
            top_k = RERANK_TOP_K
        if min_score is None:
            min_score = RERANK_MIN_SCORE

        if not results:
            return []

        if self._unavailable or not RERANKER_ENABLED:
            limit = top_k if top_k > 0 else len(results)
            return results[:limit]

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
            logger.debug("Reranker scores: %s", scores)

            scored = [
                (score, result)
                for score, result in zip(scores, results)
                if score >= min_score
            ]
            if not scored:
                logger.info(
                    "All %d results scored below RERANK_MIN_SCORE (%s)",
                    len(results),
                    min_score,
                )
                return []

            ranked = sorted(
                scored,
                key=lambda item: item[0],
                reverse=True,
            )
            limit = top_k if top_k > 0 else len(ranked)
            top = [result for _, result in ranked[:limit]]
            logger.info("Reranked %d results to top %d", len(results), len(top))
            return top
        except Exception:
            self._unavailable = True
            logger.warning(
                "Reranking failed; using unranked search results",
                exc_info=True,
            )
            limit = top_k if top_k > 0 else len(results)
            return results[:limit]

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

    @staticmethod
    def _fetch_single_page(url):
        """Fetch a single page HTML string, returning None on failure."""
        if not url or not isinstance(url, str):
            return None
        try:
            validate_url(url, "Page URL")
        except Exception:
            logger.warning("Invalid URL scheme for page fetch: %s", url)
            return None

        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            return http_get_html(
                url,
                error_class=SearchError,
                service_name="WebPage",
                headers=headers,
                timeout=FETCH_PAGE_TIMEOUT,
            )
        except Exception as e:
            logger.warning("Failed to fetch page %s: %s", url, e)
            return None

    @staticmethod
    def _extract_page_text(html, url=""):
        """Extract main text from HTML using trafilatura, returning None on failure."""
        if not html:
            return None
        try:
            import trafilatura
            extracted = trafilatura.extract(
                html,
                include_links=False,
                include_images=False,
                include_tables=True,
            )
            if extracted and extracted.strip():
                return extracted.strip()
            return None
        except ImportError:
            logger.warning("trafilatura not installed; skipping page text extraction")
            return None
        except Exception as e:
            logger.warning("Failed to extract text from %s: %s", url, e)
            return None

    def search_web(self, query):
        """Search web using local SearXNG, optionally fetch page text, and rerank."""
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
        if not results:
            return []

        # Stage 1: Result Reranking
        ranked_results = self._reranker.rerank(
            query,
            results,
            top_k=RERANK_TOP_K,
            min_score=RERANK_MIN_SCORE,
        )
        if not ranked_results:
            return []

        if not FETCH_PAGE_ENABLED:
            return ranked_results

        # Stage 2: Page fetching, text extraction, and passage reranking
        target_results = (
            ranked_results[:FETCH_PAGE_TOP_N]
            if FETCH_PAGE_TOP_N > 0
            else ranked_results
        )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, len(target_results))
        ) as executor:
            html_list = list(
                executor.map(
                    lambda r: self._fetch_single_page(r.get("url", "")),
                    target_results,
                )
            )

        candidate_passages = []
        for result, html in zip(target_results, html_list):
            extracted_text = self._extract_page_text(html, result.get("url", ""))
            if extracted_text:
                passages = split_into_passages(extracted_text, max_chars=PASSAGE_CHARS)
                for passage in passages:
                    candidate_passages.append({
                        "title": result.get("title", ""),
                        "content": passage,
                        "url": result.get("url", ""),
                    })
            else:
                candidate_passages.append({
                    "title": result.get("title", ""),
                    "content": result.get("content", ""),
                    "url": result.get("url", ""),
                })

        if not candidate_passages:
            return []

        return self._reranker.rerank(
            query,
            candidate_passages,
            top_k=RERANK_TOP_K,
            min_score=RERANK_MIN_SCORE,
        )
