#!/usr/bin/env python3
"""
Retriever module - Web search using SearXNG with optional ONNX Japanese reranking,
plus page-content fetch and passage-level reranking for deeper factual context.
"""

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor

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
    CONTEXT_CHAR_BUDGET,
    validate_url,
)
from .http_client import http_get_json, http_get_text
from .audio_utils import log_init, log_ready
from .exceptions import SearchError

logger = logging.getLogger(__name__)

_PAGE_FETCH_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}


_PARAGRAPH_SPLIT_RE = re.compile(r'\n+')
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[。！？!?\n])\s*')


def _split_into_passages(text, size):
    """Split text into ~``size``-char passages, preferring paragraph and
    sentence boundaries over a hard cut.

    A sentence longer than ``size`` (e.g. body text with no punctuation)
    is hard-chunked so a single passage can never grow past ``size``
    characters, no matter how the source text is punctuated.
    """
    if not text or not isinstance(text, str):
        return []

    text = text.strip()
    if not text:
        return []

    if size <= 0:
        size = PASSAGE_CHARS

    if len(text) <= size:
        return [text]

    passages = []
    current = ""

    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(text) if p.strip()] or [text]
    for paragraph in paragraphs:
        sentences = [
            s.strip() for s in _SENTENCE_SPLIT_RE.split(paragraph) if s.strip()
        ] or [paragraph]

        for sentence in sentences:
            if len(sentence) > size:
                if current:
                    passages.append(current)
                    current = ""
                for i in range(0, len(sentence), size):
                    chunk = sentence[i:i + size].strip()
                    if chunk:
                        passages.append(chunk)
                continue

            if not current:
                current = sentence
            elif len(current) + len(sentence) <= size:
                current += sentence
            else:
                passages.append(current)
                current = sentence

    if current:
        passages.append(current)

    return [p for p in passages if p]


def _clip_to_char_budget(items, budget=None):
    """Keep leading ``items`` until the combined title+content length would
    exceed ``budget``, dropping the rest from the back.

    Passage-level reranking can return many small passages (up to
    ``FETCH_PAGE_TOP_N`` pages, each split into ``PASSAGE_CHARS``-sized
    chunks), so the final list is bounded here the same way
    ``composer.clip_search_results`` bounds the composed prompt. At least
    one item is always kept, even if it alone exceeds the budget, so a
    single oversized passage cannot degrade the result to nothing.
    """
    if budget is None:
        budget = CONTEXT_CHAR_BUDGET

    kept = []
    total_chars = 0
    for item in items:
        item_chars = len(item.get("title", "")) + len(item.get("content", ""))
        if kept and total_chars + item_chars > budget:
            break
        kept.append(item)
        total_chars += item_chars

    if len(kept) < len(items):
        logger.warning(
            "Dropped %d passages to fit CONTEXT_CHAR_BUDGET (%s)",
            len(items) - len(kept),
            budget,
        )
    return kept


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
            filtered = [item for item in ranked if item[0] >= min_score]
            if len(filtered) < len(ranked):
                logger.info(
                    "Dropped %d results below RERANK_MIN_SCORE (%s)",
                    len(ranked) - len(filtered),
                    min_score,
                )
            # 0 or negative TOP_K means sort (and filter) only, no truncate.
            limit = top_k if top_k > 0 else len(filtered)
            top = [result for _, result in filtered[:limit]]
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
        ranked = self._reranker.rerank(query, results)

        if not FETCH_PAGE_ENABLED or not ranked:
            return ranked

        return self._fetch_and_rerank_passages(query, ranked)

    def _fetch_and_rerank_passages(self, query, results):
        """Fetch top result pages, split into passages, and rerank them.

        Any per-URL failure (fetch or extraction) is non-fatal: that result's
        original snippet is kept instead of dropping it from the context.
        """
        top_n = FETCH_PAGE_TOP_N if FETCH_PAGE_TOP_N > 0 else len(results)
        to_fetch = results[:top_n]
        rest = results[top_n:]

        if not to_fetch:
            return results

        candidates = []
        with ThreadPoolExecutor(max_workers=len(to_fetch)) as executor:
            future_to_result = {
                executor.submit(self._fetch_passages_for_result, result): result
                for result in to_fetch
            }
            for future in future_to_result:
                result = future_to_result[future]
                try:
                    passages = future.result()
                except Exception:
                    passages = None
                    logger.warning(
                        "Passage fetch worker failed for %s",
                        result.get("url", ""),
                        exc_info=True,
                    )
                candidates.extend(passages if passages else [result])

        candidates.extend(rest)
        reranked = self._reranker.rerank(query, candidates)
        return _clip_to_char_budget(reranked)

    def _fetch_passages_for_result(self, result):
        """Fetch and extract passages for a single search result.

        Returns None (caller falls back to the original snippet) if the URL
        is missing/invalid, the fetch fails, or extraction yields no text.
        """
        url = result.get("url", "")
        if not url:
            return None
        try:
            validate_url(url, "fetch_page_url")
        except ValueError:
            return None

        try:
            html = http_get_text(
                url,
                error_class=SearchError,
                service_name="page fetch",
                headers=_PAGE_FETCH_HEADERS,
                timeout=FETCH_PAGE_TIMEOUT,
            )
        except SearchError:
            logger.warning("Failed to fetch page %s", url, exc_info=True)
            return None

        text = self._extract_text(html)
        if not text:
            return None

        passages = _split_into_passages(text, PASSAGE_CHARS)
        if not passages:
            return None

        return [
            {"title": result.get("title", ""), "content": passage, "url": url}
            for passage in passages
        ]

    @staticmethod
    def _extract_text(html):
        try:
            # Dynamic import: missing dep must not break the search pipeline.
            import trafilatura

            return trafilatura.extract(html)
        except Exception:
            logger.warning("Page text extraction failed", exc_info=True)
            return None
