#!/usr/bin/env python3
"""
Tests for retriever module
"""

import pytest
import requests
from unittest.mock import Mock, patch
from types import SimpleNamespace
from src.retriever import Retriever, _Reranker, split_into_passages
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


def test_search_web_success(retriever, mocker):
    """Test successful web search with page fetch disabled"""
    mocker.patch("src.retriever.FETCH_PAGE_ENABLED", False)
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
    mocker.patch("src.retriever.FETCH_PAGE_ENABLED", False)
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
    mocker.patch("src.retriever.FETCH_PAGE_ENABLED", False)
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
    mocker.patch("src.retriever.FETCH_PAGE_ENABLED", False)
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
    mocker.patch("src.retriever.FETCH_PAGE_ENABLED", False)
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
    mocker.patch("src.retriever.FETCH_PAGE_ENABLED", False)
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
    mocker.patch("src.retriever.FETCH_PAGE_ENABLED", False)
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
    mocker.patch("src.retriever.FETCH_PAGE_ENABLED", False)
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


# Tests for passage splitting
def test_split_into_passages_empty():
    assert split_into_passages("") == []
    assert split_into_passages(None) == []
    assert split_into_passages("   ") == []


def test_split_into_passages_short():
    text = "富士山は日本で最も高い山です。"
    passages = split_into_passages(text, max_chars=250)
    assert passages == [text]


def test_split_into_passages_multi_sentence():
    sentence1 = "富士山は日本で最も高い山です。" * 5  # 80 chars
    sentence2 = "標高は3776メートルあります。" * 5    # 85 chars
    sentence3 = "静岡県と山梨県にまたがっています。" * 5  # 90 chars
    text = sentence1 + sentence2 + sentence3  # 255 chars
    passages = split_into_passages(text, max_chars=100)
    assert len(passages) >= 3
    for p in passages:
        assert len(p) <= 100


def test_split_into_passages_long_unbroken_text():
    text = "あ" * 600
    passages = split_into_passages(text, max_chars=250)
    assert len(passages) == 3
    assert len(passages[0]) == 250
    assert len(passages[1]) == 250
    assert len(passages[2]) == 100


# Tests for page fetching and two-stage reranking
def test_search_web_fetch_page_and_passage_rerank_success(retriever, mocker):
    """Test successful page fetching, passage extraction, and two-stage rerank"""
    mocker.patch("src.retriever.FETCH_PAGE_ENABLED", True)
    mocker.patch("src.retriever.RERANK_TOP_K", 2)
    mocker.patch("src.retriever.FETCH_PAGE_TOP_N", 2)
    mocker.patch("src.retriever.PASSAGE_CHARS", 100)
    mocker.patch.object(retriever._reranker, "_load")
    retriever._reranker._tokenizer = Mock(return_value={"input_ids": Mock()})

    # Model call 1 (stage 1 results): 2 results
    # Model call 2 (stage 2 passages): 3 candidate passages
    retriever._reranker._model = Mock(side_effect=[
        SimpleNamespace(logits=[[0.8], [0.6]]),
        SimpleNamespace(logits=[[0.95], [0.7], [0.4]]),
    ])

    results = [
        {'title': 'Page 1', 'content': 'snippet 1', 'url': 'http://example.com/1'},
        {'title': 'Page 2', 'content': 'snippet 2', 'url': 'http://example.com/2'},
    ]

    mocker.patch.object(retriever, "_fetch_single_page", side_effect=[
        "<html>page1</html>",
        "<html>page2</html>",
    ])
    mocker.patch.object(retriever, "_extract_page_text", side_effect=[
        "富士山の詳細な説明文です。パッセージ1の内容です。\n\n追加のパッセージ2の内容です。",
        "別の山の説明文です。パッセージ3の内容です。",
    ])

    with patch('src.http_client.requests.get', return_value=_mock_search_response(results)):
        final = retriever.search_web("富士山")

    assert len(final) == 2
    # Final results contain extracted passages
    assert "富士山" in final[0]['content']
    assert final[0]['title'] == 'Page 1'
    assert retriever._reranker._model.call_count == 2


def test_search_web_single_page_fetch_failure_degrades_to_snippet(retriever, mocker):
    """Test when 1 page fetch fails, that item falls back to snippet while other uses passage"""
    mocker.patch("src.retriever.FETCH_PAGE_ENABLED", True)
    mocker.patch("src.retriever.RERANK_TOP_K", 2)
    mocker.patch.object(retriever._reranker, "_load")
    retriever._reranker._tokenizer = Mock(return_value={"input_ids": Mock()})

    retriever._reranker._model = Mock(side_effect=[
        SimpleNamespace(logits=[[0.8], [0.6]]),
        SimpleNamespace(logits=[[0.9], [0.7]]),
    ])

    results = [
        {'title': 'Page 1', 'content': 'snippet 1', 'url': 'http://example.com/1'},
        {'title': 'Page 2', 'content': 'snippet 2', 'url': 'http://example.com/2'},
    ]

    # Page 1 succeeds, Page 2 fails (returns None)
    mocker.patch.object(retriever, "_fetch_single_page", side_effect=[
        "<html>page1</html>",
        None,
    ])
    mocker.patch.object(retriever, "_extract_page_text", side_effect=[
        "抽出された本文1",
    ])

    with patch('src.http_client.requests.get', return_value=_mock_search_response(results)):
        final = retriever.search_web("富士山")

    assert len(final) == 2
    assert final[0]['content'] == "抽出された本文1"
    assert final[1]['content'] == "snippet 2"


def test_search_web_two_stage_rerank_failure_degrades_gracefully(retriever, mocker):
    """Test graceful degradation when stage 2 rerank fails"""
    mocker.patch("src.retriever.FETCH_PAGE_ENABLED", True)
    mocker.patch("src.retriever.RERANK_TOP_K", 2)
    mocker.patch.object(retriever._reranker, "_load")
    retriever._reranker._tokenizer = Mock(return_value={"input_ids": Mock()})

    # Stage 1 succeeds, Stage 2 model inference raises Exception
    retriever._reranker._model = Mock(side_effect=[
        SimpleNamespace(logits=[[0.8], [0.6]]),
        RuntimeError("Inference crashed"),
    ])

    results = [
        {'title': 'Page 1', 'content': 'snippet 1', 'url': 'http://example.com/1'},
        {'title': 'Page 2', 'content': 'snippet 2', 'url': 'http://example.com/2'},
    ]

    mocker.patch.object(retriever, "_fetch_single_page", return_value="<html>ok</html>")
    mocker.patch.object(retriever, "_extract_page_text", return_value="抽出本文")

    with patch('src.http_client.requests.get', return_value=_mock_search_response(results)):
        final = retriever.search_web("富士山")

    # Degrades to candidate passages
    assert len(final) == 2
    assert final[0]['content'] == "抽出本文"


def test_search_web_rerank_min_score_filters_low_scores(retriever, mocker):
    """Test that items with score < RERANK_MIN_SCORE are dropped"""
    mocker.patch("src.retriever.FETCH_PAGE_ENABLED", False)
    mocker.patch("src.retriever.RERANK_TOP_K", 3)
    mocker.patch("src.retriever.RERANK_MIN_SCORE", 0.5)
    mocker.patch.object(retriever._reranker, "_load")
    retriever._reranker._tokenizer = Mock(return_value={"input_ids": Mock()})
    retriever._reranker._model = Mock(
        return_value=SimpleNamespace(logits=[[0.8], [0.2], [-0.1]])
    )

    results = [
        {'title': 'High', 'content': 'high', 'url': 'http://high'},
        {'title': 'Low', 'content': 'low', 'url': 'http://low'},
        {'title': 'Negative', 'content': 'negative', 'url': 'http://negative'},
    ]
    with patch('src.http_client.requests.get', return_value=_mock_search_response(results)):
        ranked = retriever.search_web("test query")

    assert [r['title'] for r in ranked] == ['High']


def test_search_web_rerank_min_score_all_dropped_returns_empty(retriever, mocker):
    """Test that when all scores are below RERANK_MIN_SCORE, empty list is returned"""
    mocker.patch("src.retriever.FETCH_PAGE_ENABLED", False)
    mocker.patch("src.retriever.RERANK_TOP_K", 3)
    mocker.patch("src.retriever.RERANK_MIN_SCORE", 0.9)
    mocker.patch.object(retriever._reranker, "_load")
    retriever._reranker._tokenizer = Mock(return_value={"input_ids": Mock()})
    retriever._reranker._model = Mock(
        return_value=SimpleNamespace(logits=[[0.3], [0.2], [0.1]])
    )

    results = [
        {'title': 'Low1', 'content': 'low1', 'url': 'http://low1'},
        {'title': 'Low2', 'content': 'low2', 'url': 'http://low2'},
        {'title': 'Low3', 'content': 'low3', 'url': 'http://low3'},
    ]
    with patch('src.http_client.requests.get', return_value=_mock_search_response(results)):
        ranked = retriever.search_web("test query")

    assert ranked == []


def test_search_web_trafilatura_extraction_failure_falls_back_to_snippet(retriever, mocker):
    """Test that if trafilatura extraction fails, snippet is retained"""
    mocker.patch("src.retriever.FETCH_PAGE_ENABLED", True)
    mocker.patch("src.retriever.RERANK_TOP_K", 1)
    mocker.patch.object(retriever._reranker, "_load")
    retriever._reranker._tokenizer = Mock(return_value={"input_ids": Mock()})
    retriever._reranker._model = Mock(return_value=SimpleNamespace(logits=[[0.8]]))

    results = [
        {'title': 'Page 1', 'content': 'fallback snippet', 'url': 'http://example.com/1'},
    ]

    mocker.patch.object(retriever, "_fetch_single_page", return_value="<html>empty</html>")
    mocker.patch.object(retriever, "_extract_page_text", return_value=None)

    with patch('src.http_client.requests.get', return_value=_mock_search_response(results)):
        final = retriever.search_web("富士山")

    assert len(final) == 1
    assert final[0]['content'] == "fallback snippet"
