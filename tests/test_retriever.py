#!/usr/bin/env python3
"""
Tests for retriever module
"""

import pytest
import requests
from unittest.mock import Mock, patch
from types import SimpleNamespace
from src.retriever import Retriever, _Reranker, split_passages, extract_page_text
from src.exceptions import SearchError


@pytest.fixture(autouse=True)
def disable_page_fetch(mocker):
    """Keep existing snippet-path tests isolated from page fetch."""
    mocker.patch("src.retriever.FETCH_PAGE_ENABLED", False)


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


def test_split_passages_prefers_sentence_breaks():
    text = "あいうえお。かきくけこ。さしすせそ。"
    assert split_passages(text, 10) == ["あいうえお。", "かきくけこ。", "さしすせそ。"]


def test_split_passages_short_text_is_single_chunk():
    assert split_passages("短い本文", 250) == ["短い本文"]


def test_split_passages_empty_or_invalid():
    assert split_passages("", 250) == []
    assert split_passages("本文", 0) == []


def test_extract_page_text_uses_trafilatura(mocker):
    fake = mocker.Mock()
    fake.extract.return_value = "抽出本文"
    mocker.patch.dict("sys.modules", {"trafilatura": fake})

    assert extract_page_text("<html>page</html>") == "抽出本文"
    fake.extract.assert_called_once_with("<html>page</html>")


def test_extract_page_text_import_error_returns_empty(mocker):
    mocker.patch.dict("sys.modules", {"trafilatura": None})
    assert extract_page_text("<html>page</html>") == ""


def test_extract_page_text_rejects_non_string():
    assert extract_page_text(None) == ""
    assert extract_page_text(123) == ""


def test_search_web_uses_extracted_page_text(retriever, mocker):
    mocker.patch("src.retriever.FETCH_PAGE_ENABLED", True)
    mocker.patch("src.retriever.RERANKER_ENABLED", False)
    mocker.patch("src.retriever.http_get_text", return_value="<html>page</html>")
    mocker.patch("src.retriever.extract_page_text", return_value="抽出された本文です")
    results = [
        {"title": "Test", "content": "snippet", "url": "http://test.com"},
    ]
    with patch("src.http_client.requests.get", return_value=_mock_search_response(results)):
        returned = retriever.search_web("test query")

    assert returned[0]["content"] == "抽出された本文です"
    assert returned[0]["title"] == "Test"
    assert returned[0]["url"] == "http://test.com"


def test_search_web_keeps_snippet_when_one_page_fails(retriever, mocker):
    mocker.patch("src.retriever.FETCH_PAGE_ENABLED", True)
    mocker.patch("src.retriever.RERANKER_ENABLED", False)

    def fake_get_text(url, **_kwargs):
        if "fail" in url:
            raise SearchError("timed out")
        return "<html>ok</html>"

    mocker.patch("src.retriever.http_get_text", side_effect=fake_get_text)
    mocker.patch("src.retriever.extract_page_text", return_value="本文です")
    warning = mocker.patch("src.retriever.logger.warning")
    results = [
        {"title": "Ok", "content": "ok-snippet", "url": "http://ok.com"},
        {"title": "Fail", "content": "fail-snippet", "url": "http://fail.com"},
    ]
    with patch("src.http_client.requests.get", return_value=_mock_search_response(results)):
        returned = retriever.search_web("test query")

    by_title = {item["title"]: item["content"] for item in returned}
    assert by_title["Ok"] == "本文です"
    assert by_title["Fail"] == "fail-snippet"
    warning.assert_called()


def test_search_web_fetch_disabled_keeps_snippets(retriever, mocker):
    mocker.patch("src.retriever.FETCH_PAGE_ENABLED", False)
    fetch = mocker.patch("src.retriever.http_get_text")
    mocker.patch("src.retriever.RERANKER_ENABLED", False)
    results = [
        {"title": "Test", "content": "snippet", "url": "http://test.com"},
    ]
    with patch("src.http_client.requests.get", return_value=_mock_search_response(results)):
        returned = retriever.search_web("test query")

    assert returned[0]["content"] == "snippet"
    fetch.assert_not_called()


def test_search_web_skips_non_http_url(retriever, mocker):
    mocker.patch("src.retriever.FETCH_PAGE_ENABLED", True)
    mocker.patch("src.retriever.RERANKER_ENABLED", False)
    fetch = mocker.patch("src.retriever.http_get_text")
    results = [
        {"title": "Ftp", "content": "ftp-snippet", "url": "ftp://example.com/file"},
    ]
    with patch("src.http_client.requests.get", return_value=_mock_search_response(results)):
        returned = retriever.search_web("test query")

    assert returned[0]["content"] == "ftp-snippet"
    fetch.assert_not_called()


def test_search_web_passage_rerank_orders_chunks(retriever, mocker):
    mocker.patch("src.retriever.FETCH_PAGE_ENABLED", True)
    mocker.patch("src.retriever.PASSAGE_CHARS", 10)
    mocker.patch.object(retriever._reranker, "_load")
    retriever._reranker._tokenizer = Mock(return_value={"input_ids": Mock()})
    retriever._reranker._model = Mock(
        return_value=SimpleNamespace(logits=[[0.1], [0.9], [0.3]])
    )
    mocker.patch("src.retriever.http_get_text", return_value="<html>page</html>")
    mocker.patch(
        "src.retriever.extract_page_text",
        return_value="あいうえお。かきくけこ。さしすせそ。",
    )
    results = [
        {"title": "Doc", "content": "snippet", "url": "http://doc.com"},
    ]
    with patch("src.http_client.requests.get", return_value=_mock_search_response(results)):
        returned = retriever.search_web("test query")

    assert [item["content"] for item in returned] == [
        "かきくけこ。",
        "さしすせそ。",
        "あいうえお。",
    ]


def test_search_web_passage_rerank_failure_degrades_to_page_text(retriever, mocker):
    mocker.patch("src.retriever.FETCH_PAGE_ENABLED", True)
    mocker.patch("src.retriever.PASSAGE_CHARS", 10)
    mocker.patch.object(retriever._reranker, "_load")
    retriever._reranker._tokenizer = Mock(return_value={"input_ids": Mock()})
    retriever._reranker._model = Mock(side_effect=RuntimeError("passage failed"))
    warning = mocker.patch("src.retriever.logger.warning")
    mocker.patch("src.retriever.http_get_text", return_value="<html>page</html>")
    mocker.patch(
        "src.retriever.extract_page_text",
        return_value="あいうえお。かきくけこ。さしすせそ。",
    )
    results = [
        {"title": "Doc", "content": "snippet", "url": "http://doc.com"},
    ]
    with patch("src.http_client.requests.get", return_value=_mock_search_response(results)):
        returned = retriever.search_web("test query")

    assert returned[0]["content"] == "あいうえお。かきくけこ。さしすせそ。"
    assert retriever._reranker._unavailable is False
    warning.assert_called()


def test_search_web_drops_results_below_min_score(retriever, mocker):
    mocker.patch("src.retriever.RERANK_MIN_SCORE", 0.5)
    mocker.patch.object(retriever._reranker, "_load")
    retriever._reranker._tokenizer = Mock(return_value={"input_ids": Mock()})
    retriever._reranker._model = Mock(
        return_value=SimpleNamespace(logits=[[0.1], [0.9], [0.3]])
    )
    results = [
        {"title": "Low", "content": "low", "url": "http://low"},
        {"title": "High", "content": "high", "url": "http://high"},
        {"title": "Middle", "content": "middle", "url": "http://middle"},
    ]
    with patch("src.http_client.requests.get", return_value=_mock_search_response(results)):
        ranked = retriever.search_web("test query")

    assert [item["title"] for item in ranked] == ["High"]


def test_search_web_min_score_all_dropped_returns_empty(retriever, mocker):
    mocker.patch("src.retriever.RERANK_MIN_SCORE", 1.0)
    mocker.patch.object(retriever._reranker, "_load")
    retriever._reranker._tokenizer = Mock(return_value={"input_ids": Mock()})
    retriever._reranker._model = Mock(
        return_value=SimpleNamespace(logits=[[0.1], [0.9], [0.3]])
    )
    results = [
        {"title": "Low", "content": "low", "url": "http://low"},
        {"title": "High", "content": "high", "url": "http://high"},
        {"title": "Middle", "content": "middle", "url": "http://middle"},
    ]
    with patch("src.http_client.requests.get", return_value=_mock_search_response(results)):
        ranked = retriever.search_web("test query")

    assert ranked == []


def test_search_web_passage_min_score_all_dropped_returns_empty(retriever, mocker):
    mocker.patch("src.retriever.FETCH_PAGE_ENABLED", True)
    mocker.patch("src.retriever.RERANK_MIN_SCORE", 1.0)
    mocker.patch("src.retriever.PASSAGE_CHARS", 10)
    mocker.patch.object(retriever._reranker, "_load")
    retriever._reranker._tokenizer = Mock(return_value={"input_ids": Mock()})
    retriever._reranker._model = Mock(
        return_value=SimpleNamespace(logits=[[0.1], [0.2], [0.3]])
    )
    mocker.patch("src.retriever.http_get_text", return_value="<html>page</html>")
    mocker.patch(
        "src.retriever.extract_page_text",
        return_value="あいうえお。かきくけこ。さしすせそ。",
    )
    results = [
        {"title": "Doc", "content": "snippet", "url": "http://doc.com"},
    ]
    with patch("src.http_client.requests.get", return_value=_mock_search_response(results)):
        returned = retriever.search_web("test query")

    assert returned == []
