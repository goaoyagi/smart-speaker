#!/usr/bin/env python3
"""
Tests for retriever module
"""

import pytest
import requests
from unittest.mock import Mock, patch, MagicMock
from src.retriever import Retriever
from src.exceptions import SearchError


@pytest.fixture
def retriever(monkeypatch):
    """Create Retriever with reranking disabled (offline unit-test policy)."""
    monkeypatch.setattr("src.retriever.RERANKER_ENABLED", False)
    return Retriever()


def test_retriever_initialization(retriever):
    """Test that Retriever initializes correctly"""
    assert retriever.searxng_url == "http://localhost:8080"
    assert retriever.reranker_enabled is False


def test_search_web_success(retriever):
    """Test successful web search"""
    mock_response = Mock()
    mock_response.json.return_value = {
        'results': [
            {'title': 'Test', 'content': 'Test content', 'url': 'http://test.com'}
        ]
    }
    mock_response.raise_for_status = Mock()

    with patch('src.http_client.requests.get', return_value=mock_response):
        results = retriever.search_web("test query")

    assert len(results) == 1
    assert results[0]['title'] == 'Test'


def test_search_web_connection_error(retriever):
    """Test web search raises SearchError on connection failure"""
    with patch('src.http_client.requests.get',
               side_effect=requests.exceptions.ConnectionError("Connection refused")):
        with pytest.raises(SearchError, match="Cannot connect to SearXNG"):
            retriever.search_web("test query")


def test_search_web_timeout(retriever):
    """Test web search raises SearchError on timeout"""
    with patch('src.http_client.requests.get',
               side_effect=requests.exceptions.Timeout("timed out")):
        with pytest.raises(SearchError, match="timed out"):
            retriever.search_web("test query")


def test_search_web_reranks_and_keeps_top_k(monkeypatch, mock_search_results):
    """When enabled, search results are reranked and truncated to RERANK_TOP_K."""
    monkeypatch.setattr("src.retriever.RERANKER_ENABLED", True)
    monkeypatch.setattr("src.retriever.RERANK_TOP_K", 1)

    extra = {
        'title': '無関係',
        'content': '関係ない内容',
        'url': 'http://example.com/3',
    }
    api_results = mock_search_results + [extra]

    mock_response = Mock()
    mock_response.json.return_value = {'results': api_results}
    mock_response.raise_for_status = Mock()

    r = Retriever()
    # Inject a ready mock reranker (no real model download).
    r._reranker_available = True
    r._reranker_load_attempted = True
    r._score_pairs = MagicMock(return_value=[0.1, 0.9, 0.2])

    with patch('src.http_client.requests.get', return_value=mock_response):
        results = r.search_web("テスト質問")

    assert len(results) == 1
    assert results[0]['title'] == 'テスト結果2'
    r._score_pairs.assert_called_once()


def test_rerank_skips_on_import_error(monkeypatch, mock_search_results):
    """Missing optimum/transformers must not break search (non-fatal)."""
    monkeypatch.setattr("src.retriever.RERANKER_ENABLED", True)

    mock_response = Mock()
    mock_response.json.return_value = {'results': mock_search_results}
    mock_response.raise_for_status = Mock()

    r = Retriever()
    r._reranker_load_attempted = False
    r._reranker_available = False

    # sys.modules[name] = None makes ``import name`` raise ImportError.
    with patch.dict(
        'sys.modules',
        {'transformers': None, 'optimum': None, 'optimum.onnxruntime': None},
    ):
        with patch('src.http_client.requests.get', return_value=mock_response):
            results = r.search_web("テスト質問")

    assert len(results) == 2
    assert results[0]['title'] == 'テスト結果1'
    assert r._reranker_available is False


def test_rerank_skips_on_inference_failure(monkeypatch, mock_search_results):
    """Inference failures fall back to the original SearXNG order."""
    monkeypatch.setattr("src.retriever.RERANKER_ENABLED", True)

    mock_response = Mock()
    mock_response.json.return_value = {'results': mock_search_results}
    mock_response.raise_for_status = Mock()

    r = Retriever()
    r._reranker_available = True
    r._reranker_load_attempted = True
    r._score_pairs = MagicMock(side_effect=RuntimeError("onnx blew up"))

    with patch('src.http_client.requests.get', return_value=mock_response):
        results = r.search_web("テスト質問")

    assert results == mock_search_results


def test_resolve_model_source_prefers_local(tmp_path, monkeypatch):
    """Local ONNX export directory is preferred over the Hub model id."""
    monkeypatch.setattr("src.retriever.RERANKER_ENABLED", False)
    local = tmp_path / "reranker"
    local.mkdir()
    (local / "model.onnx").write_text("dummy")
    monkeypatch.setattr("src.retriever.RERANKER_LOCAL_PATH", str(local))

    r = Retriever()
    source, from_local = r._resolve_model_source()
    assert from_local is True
    assert source == str(local)


def test_resolve_model_source_falls_back_to_model_id(tmp_path, monkeypatch):
    """Missing local cache falls back to RERANKER_MODEL_ID."""
    monkeypatch.setattr("src.retriever.RERANKER_ENABLED", False)
    monkeypatch.setattr("src.retriever.RERANKER_LOCAL_PATH", str(tmp_path / "missing"))
    monkeypatch.setattr(
        "src.retriever.RERANKER_MODEL_ID",
        "hotchpotch/japanese-reranker-cross-encoder-xsmall-v1",
    )

    r = Retriever()
    source, from_local = r._resolve_model_source()
    assert from_local is False
    assert source == "hotchpotch/japanese-reranker-cross-encoder-xsmall-v1"
