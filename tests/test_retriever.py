#!/usr/bin/env python3
"""
Tests for retriever module
"""

import pytest
import requests
from unittest.mock import Mock, patch
from types import SimpleNamespace
from src.retriever import Retriever, _Reranker
from src.exceptions import SearchError


@pytest.fixture
def retriever():
    """Create Retriever instance"""
    return Retriever()


def _mock_search_response(results):
    mock_response = Mock()
    mock_response.json.return_value = {'results': results}
    mock_response.raise_for_status = Mock()
    return mock_response


def test_retriever_initialization(retriever):
    """Test that Retriever initializes correctly"""
    assert retriever.searxng_url == "http://localhost:8080"
    assert isinstance(retriever._reranker, _Reranker)


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


def test_search_web_reranks_and_truncates(retriever, mocker):
    mocker.patch("src.retriever.RERANK_TOP_K", 2)
    mocker.patch.object(retriever._reranker, "_load")
    retriever._reranker._tokenizer = Mock(return_value={"input_ids": Mock()})
    retriever._reranker._model = Mock(
        return_value=SimpleNamespace(logits=[[0.1], [0.9], [0.3]])
    )

    results = [
        {'title': 'Low', 'content': 'low', 'url': 'http://low'},
        {'title': 'High', 'content': 'high', 'url': 'http://high'},
        {'title': 'Middle', 'content': 'middle', 'url': 'http://middle'},
    ]
    with patch('src.http_client.requests.get', return_value=_mock_search_response(results)):
        ranked = retriever.search_web("test query")

    assert [result['title'] for result in ranked] == ['High', 'Middle']
    retriever._reranker._tokenizer.assert_called_once()
    retriever._reranker._model.assert_called_once()


def test_search_web_reranks_when_top_k_equals_result_count(retriever, mocker):
    mocker.patch("src.retriever.RERANK_TOP_K", 3)
    mocker.patch.object(retriever._reranker, "_load")
    retriever._reranker._tokenizer = Mock(return_value={"input_ids": Mock()})
    retriever._reranker._model = Mock(
        return_value=SimpleNamespace(logits=[[0.1], [0.9], [0.3]])
    )

    results = [
        {'title': 'Low', 'content': 'low', 'url': 'http://low'},
        {'title': 'High', 'content': 'high', 'url': 'http://high'},
        {'title': 'Middle', 'content': 'middle', 'url': 'http://middle'},
    ]
    with patch('src.http_client.requests.get', return_value=_mock_search_response(results)):
        ranked = retriever.search_web("test query")

    assert [result['title'] for result in ranked] == ['High', 'Middle', 'Low']


def test_search_web_top_k_zero_ranks_without_truncating(retriever, mocker):
    mocker.patch("src.retriever.RERANK_TOP_K", 0)
    mocker.patch.object(retriever._reranker, "_load")
    retriever._reranker._tokenizer = Mock(return_value={"input_ids": Mock()})
    retriever._reranker._model = Mock(
        return_value=SimpleNamespace(logits=[[0.1], [0.9], [0.3]])
    )

    results = [
        {'title': 'Low', 'content': 'low', 'url': 'http://low'},
        {'title': 'High', 'content': 'high', 'url': 'http://high'},
        {'title': 'Middle', 'content': 'middle', 'url': 'http://middle'},
    ]
    with patch('src.http_client.requests.get', return_value=_mock_search_response(results)):
        ranked = retriever.search_web("test query")

    assert [result['title'] for result in ranked] == ['High', 'Middle', 'Low']


def test_search_web_reranking_disabled(retriever, mocker):
    mocker.patch("src.retriever.RERANKER_ENABLED", False)
    load = mocker.patch.object(retriever._reranker, "_load")
    results = [
        {'title': 'First', 'content': 'first', 'url': 'http://first'},
        {'title': 'Second', 'content': 'second', 'url': 'http://second'},
        {'title': 'Third', 'content': 'third', 'url': 'http://third'},
        {'title': 'Fourth', 'content': 'fourth', 'url': 'http://fourth'},
    ]
    with patch('src.http_client.requests.get', return_value=_mock_search_response(results)):
        returned = retriever.search_web("test query")

    assert returned == results
    load.assert_not_called()


def test_search_web_import_error_falls_back_permanently(retriever, mocker):
    load = mocker.patch.object(
        retriever._reranker, "_load", side_effect=ImportError("missing optimum")
    )
    warning = mocker.patch("src.retriever.logger.warning")
    results = [
        {'title': 'First', 'content': 'first', 'url': 'http://first'},
        {'title': 'Second', 'content': 'second', 'url': 'http://second'},
        {'title': 'Third', 'content': 'third', 'url': 'http://third'},
        {'title': 'Fourth', 'content': 'fourth', 'url': 'http://fourth'},
    ]
    with patch('src.http_client.requests.get', return_value=_mock_search_response(results)):
        assert retriever.search_web("test query") == results
        assert retriever.search_web("test query") == results

    assert load.call_count == 1
    assert retriever._reranker._unavailable is True
    warning.assert_called_once()


def test_search_web_inference_failure_falls_back_permanently(retriever, mocker):
    mocker.patch.object(retriever._reranker, "_load")
    retriever._reranker._tokenizer = Mock(return_value={"input_ids": Mock()})
    retriever._reranker._model = Mock(side_effect=RuntimeError("inference failed"))
    results = [
        {'title': 'First', 'content': 'first', 'url': 'http://first'},
        {'title': 'Second', 'content': 'second', 'url': 'http://second'},
        {'title': 'Third', 'content': 'third', 'url': 'http://third'},
        {'title': 'Fourth', 'content': 'fourth', 'url': 'http://fourth'},
    ]
    with patch('src.http_client.requests.get', return_value=_mock_search_response(results)):
        assert retriever.search_web("test query") == results
        # Second call must not retry inference after the permanent latch.
        assert retriever.search_web("test query") == results

    assert retriever._reranker._unavailable is True
    assert retriever._reranker._model.call_count == 1


def test_search_web_empty_results_skips_reranking(retriever, mocker):
    load = mocker.patch.object(retriever._reranker, "_load")
    with patch('src.http_client.requests.get', return_value=_mock_search_response([])):
        assert retriever.search_web("test query") == []

    load.assert_not_called()


def test_search_web_respects_candidate_limit(retriever, mocker):
    mocker.patch("src.retriever.RERANKER_ENABLED", False)
    mocker.patch("src.retriever.SEARCH_CANDIDATE_LIMIT", 2)
    results = [
        {'title': 'A', 'content': 'a', 'url': 'http://a'},
        {'title': 'B', 'content': 'b', 'url': 'http://b'},
        {'title': 'C', 'content': 'c', 'url': 'http://c'},
    ]
    with patch('src.http_client.requests.get', return_value=_mock_search_response(results)):
        returned = retriever.search_web("test query")

    assert [r['title'] for r in returned] == ['A', 'B']


def test_resolve_model_source_prefers_local(tmp_path, monkeypatch):
    local = tmp_path / "reranker"
    local.mkdir()
    (local / "model.onnx").write_text("dummy")
    monkeypatch.setattr("src.retriever.RERANKER_LOCAL_PATH", str(local))

    source, from_local = _Reranker._resolve_model_source()
    assert from_local is True
    assert source == str(local)


def test_resolve_model_source_falls_back_to_model_id(tmp_path, monkeypatch):
    monkeypatch.setattr("src.retriever.RERANKER_LOCAL_PATH", str(tmp_path / "missing"))
    monkeypatch.setattr(
        "src.retriever.RERANKER_MODEL_ID",
        "hotchpotch/japanese-reranker-cross-encoder-xsmall-v1",
    )

    source, from_local = _Reranker._resolve_model_source()
    assert from_local is False
    assert source == "hotchpotch/japanese-reranker-cross-encoder-xsmall-v1"


@pytest.mark.parametrize(
    "logits, expected",
    [
        ([[0.1], [0.9], [0.3]], [0.1, 0.9, 0.3]),
        ([[0.2, 0.8], [0.7, 0.1]], [0.8, 0.1]),
        ([0.5, 0.1, 0.9], [0.5, 0.1, 0.9]),
    ],
)
def test_scores_absorbs_logit_shapes(logits, expected):
    assert _Reranker._scores(logits) == expected


def test_scores_absorbs_torch_like_tensor():
    class _FakeTensor:
        def __init__(self, data):
            self._data = data

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self

        def tolist(self):
            return self._data

    logits = _FakeTensor([[0.1], [0.9]])
    assert _Reranker._scores(logits) == [0.1, 0.9]


class TestSplitPassages:
    def test_short_text_returns_single(self):
        assert Retriever._split_passages("短いテキスト", char_limit=250) == ["短いテキスト"]

    def test_splits_on_sentence_end(self):
        text = "A" * 200 + "。" + "B" * 200 + "。" + "C" * 50
        passages = Retriever._split_passages(text, char_limit=250)
        assert len(passages) >= 2
        assert all(len(p) <= 260 for p in passages)

    def test_splits_on_newline(self):
        text = "A" * 200 + "\n" + "B" * 200
        passages = Retriever._split_passages(text, char_limit=250)
        assert len(passages) == 2

    def test_empty_passages_removed(self):
        text = "。" * 300
        passages = Retriever._split_passages(text, char_limit=250)
        assert all(p for p in passages)


class TestFetchAndRerankPassages:
    def test_disabled_returns_original(self, retriever, mocker):
        mocker.patch("src.retriever.FETCH_PAGE_ENABLED", False)
        results = [
            {"title": "T1", "content": "C1", "url": "http://a.com"},
            {"title": "T2", "content": "C2", "url": "http://b.com"},
        ]
        returned = retriever._fetch_and_rerank_passages("query", results)
        assert returned == results

    def test_empty_results_returns_empty(self, retriever, mocker):
        mocker.patch("src.retriever.FETCH_PAGE_ENABLED", True)
        assert retriever._fetch_and_rerank_passages("query", []) == []

    def test_fetch_success_replaces_snippet_with_passages(self, retriever, mocker):
        mocker.patch("src.retriever.FETCH_PAGE_ENABLED", True)
        mocker.patch("src.retriever.FETCH_PAGE_TOP_N", 2)
        mocker.patch("src.retriever.PASSAGE_CHARS", 500)
        mocker.patch("src.retriever.RERANK_MIN_SCORE", 0.0)
        mocker.patch("src.retriever.CONTEXT_CHAR_BUDGET", 99999)

        def fake_fetch(url):
            if url == "http://a.com":
                return "これは抽出された本文です。とても長い文章が含まれています。"
            return None

        mocker.patch.object(retriever, "_fetch_page_text", side_effect=fake_fetch)
        mocker.patch.object(retriever._reranker, "_load")
        retriever._reranker._tokenizer = Mock(return_value={"input_ids": Mock()})
        retriever._reranker._model = Mock(
            return_value=SimpleNamespace(logits=[[0.9], [0.3]])
        )

        results = [
            {"title": "T1", "content": "snippet1", "url": "http://a.com"},
            {"title": "T2", "content": "snippet2", "url": "http://b.com"},
        ]
        returned = retriever._fetch_and_rerank_passages("query", results)

        assert len(returned) >= 1
        assert returned[0]["content"] == "これは抽出された本文です。とても長い文章が含まれています。"

    def test_one_fetch_failure_others_continue(self, retriever, mocker):
        mocker.patch("src.retriever.FETCH_PAGE_ENABLED", True)
        mocker.patch("src.retriever.FETCH_PAGE_TOP_N", 2)
        mocker.patch("src.retriever.PASSAGE_CHARS", 500)
        mocker.patch("src.retriever.RERANK_MIN_SCORE", 0.0)
        mocker.patch("src.retriever.CONTEXT_CHAR_BUDGET", 99999)

        def fake_fetch(url):
            if url == "http://a.com":
                return "抽出された本文です。"
            raise SearchError("fail")

        mocker.patch.object(retriever, "_fetch_page_text", side_effect=fake_fetch)
        mocker.patch.object(retriever._reranker, "_load")
        retriever._reranker._tokenizer = Mock(return_value={"input_ids": Mock()})
        retriever._reranker._model = Mock(
            return_value=SimpleNamespace(logits=[[0.9], [0.5]])
        )

        results = [
            {"title": "T1", "content": "snippet1", "url": "http://a.com"},
            {"title": "T2", "content": "snippet2", "url": "http://b.com"},
        ]
        returned = retriever._fetch_and_rerank_passages("query", results)

        contents = {r["content"] for r in returned}
        assert "抽出された本文です。" in contents
        assert "snippet2" in contents

    def test_second_rerank_failure_degrades(self, retriever, mocker):
        mocker.patch("src.retriever.FETCH_PAGE_ENABLED", True)
        mocker.patch("src.retriever.FETCH_PAGE_TOP_N", 1)
        mocker.patch("src.retriever.RERANK_MIN_SCORE", 0.0)
        mocker.patch("src.retriever.CONTEXT_CHAR_BUDGET", 99999)

        mocker.patch.object(retriever, "_fetch_page_text", return_value="抽出本文")
        mocker.patch.object(
            retriever._reranker,
            "rerank",
            side_effect=RuntimeError("rerank failed"),
        )
        warning = mocker.patch("src.retriever.logger.warning")

        results = [
            {"title": "T1", "content": "snippet1", "url": "http://a.com"},
        ]
        returned = retriever._fetch_and_rerank_passages("query", results)

        assert len(returned) == 1
        assert returned[0]["content"] == "抽出本文"
        assert any("Passage re-ranking failed" in str(c) for c in warning.call_args_list)

    def test_min_score_filters_low_scores(self, retriever, mocker):
        mocker.patch("src.retriever.FETCH_PAGE_ENABLED", True)
        mocker.patch("src.retriever.FETCH_PAGE_TOP_N", 1)
        mocker.patch("src.retriever.RERANK_MIN_SCORE", 0.5)
        mocker.patch("src.retriever.CONTEXT_CHAR_BUDGET", 99999)

        mocker.patch.object(retriever, "_fetch_page_text", return_value="抽出本文")
        mocker.patch.object(retriever._reranker, "_load")
        retriever._reranker._tokenizer = Mock(return_value={"input_ids": Mock()})
        retriever._reranker._model = Mock(
            return_value=SimpleNamespace(logits=[[0.3]])
        )

        results = [
            {"title": "T1", "content": "snippet1", "url": "http://a.com"},
        ]
        returned = retriever._fetch_and_rerank_passages("query", results)

        assert returned == []

    def test_all_dropped_by_score_returns_empty(self, retriever, mocker):
        mocker.patch("src.retriever.FETCH_PAGE_ENABLED", True)
        mocker.patch("src.retriever.FETCH_PAGE_TOP_N", 2)
        mocker.patch("src.retriever.RERANK_MIN_SCORE", 0.9)
        mocker.patch("src.retriever.CONTEXT_CHAR_BUDGET", 99999)

        mocker.patch.object(retriever, "_fetch_page_text", return_value="抽出本文")
        mocker.patch.object(retriever._reranker, "_load")
        retriever._reranker._tokenizer = Mock(return_value={"input_ids": Mock()})
        retriever._reranker._model = Mock(
            return_value=SimpleNamespace(logits=[[0.1], [0.2]])
        )

        results = [
            {"title": "T1", "content": "s1", "url": "http://a.com"},
            {"title": "T2", "content": "s2", "url": "http://b.com"},
        ]
        returned = retriever._fetch_and_rerank_passages("query", results)

        assert returned == []

    def test_context_char_budget_clips(self, retriever, mocker):
        mocker.patch("src.retriever.FETCH_PAGE_ENABLED", True)
        mocker.patch("src.retriever.FETCH_PAGE_TOP_N", 1)
        mocker.patch("src.retriever.RERANK_MIN_SCORE", 0.0)
        mocker.patch("src.retriever.CONTEXT_CHAR_BUDGET", 30)

        mocker.patch.object(
            retriever, "_fetch_page_text", return_value="長い本文です。" * 20
        )
        mocker.patch.object(retriever._reranker, "_load")
        retriever._reranker._tokenizer = Mock(return_value={"input_ids": Mock()})
        retriever._reranker._model = Mock(
            return_value=SimpleNamespace(logits=[[0.9]] * 20)
        )

        results = [
            {"title": "T1", "content": "s1", "url": "http://a.com"},
        ]
        returned = retriever._fetch_and_rerank_passages("query", results)

        total = sum(len(r["title"]) + len(r["content"]) for r in returned)
        assert total <= 30 or len(returned) == 1

    def test_non_http_url_skipped(self, retriever, mocker):
        mocker.patch("src.retriever.FETCH_PAGE_ENABLED", True)
        mocker.patch("src.retriever.FETCH_PAGE_TOP_N", 1)
        mocker.patch("src.retriever.RERANK_MIN_SCORE", 0.0)
        mocker.patch("src.retriever.CONTEXT_CHAR_BUDGET", 99999)

        http_get_text_spy = mocker.patch("src.retriever.http_get_text")

        results = [
            {"title": "T1", "content": "snippet1", "url": "ftp://a.com"},
        ]
        returned = retriever._fetch_and_rerank_passages("query", results)

        http_get_text_spy.assert_not_called()
        assert returned[0]["content"] == "snippet1"
