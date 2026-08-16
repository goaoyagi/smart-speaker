#!/usr/bin/env python3
"""
Tests for retriever module
"""

import pytest
import sys
from unittest.mock import Mock
from types import SimpleNamespace
from src.retriever import Retriever, _Reranker
from src.exceptions import SearchError


@pytest.fixture
def retriever():
    """Create Retriever instance"""
    return Retriever()


def _mock_search_response(results):
    return {'results': results}


def test_retriever_initialization(retriever):
    """Test that Retriever initializes correctly"""
    assert retriever.searxng_url == "http://localhost:8080"
    assert isinstance(retriever._reranker, _Reranker)


def test_search_web_success(retriever):
    """Test successful web search"""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("src.retriever.FETCH_PAGE_ENABLED", False)
        mp.setattr(
            "src.retriever.http_get_json",
            lambda *args, **kwargs: {
                'results': [
                    {'title': 'Test', 'content': 'Test content', 'url': 'http://test.com'}
                ]
            },
        )
        results = retriever.search_web("test query")

    assert len(results) == 1
    assert results[0]['title'] == 'Test'


def test_search_web_connection_error(retriever):
    """Test web search raises SearchError on connection failure"""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "src.retriever.http_get_json",
            lambda *args, **kwargs: (_ for _ in ()).throw(SearchError("Cannot connect to SearXNG")),
        )
        with pytest.raises(SearchError, match="Cannot connect to SearXNG"):
            retriever.search_web("test query")


def test_search_web_timeout(retriever):
    """Test web search raises SearchError on timeout"""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "src.retriever.http_get_json",
            lambda *args, **kwargs: (_ for _ in ()).throw(SearchError("timed out")),
        )
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
    mocker.patch("src.retriever.http_get_json", return_value=_mock_search_response(results))
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
    mocker.patch("src.retriever.http_get_json", return_value=_mock_search_response(results))
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
    mocker.patch("src.retriever.http_get_json", return_value=_mock_search_response(results))
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
    mocker.patch("src.retriever.http_get_json", return_value=_mock_search_response(results))
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
    mocker.patch("src.retriever.http_get_json", return_value=_mock_search_response(results))
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
    mocker.patch("src.retriever.http_get_json", return_value=_mock_search_response(results))
    assert retriever.search_web("test query") == results
    # Second call must not retry inference after the permanent latch.
    assert retriever.search_web("test query") == results

    assert retriever._reranker._unavailable is True
    assert retriever._reranker._model.call_count == 1


def test_search_web_empty_results_skips_reranking(retriever, mocker):
    mocker.patch("src.retriever.FETCH_PAGE_ENABLED", False)
    load = mocker.patch.object(retriever._reranker, "_load")
    mocker.patch("src.retriever.http_get_json", return_value=_mock_search_response([]))
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
    mocker.patch("src.retriever.http_get_json", return_value=_mock_search_response(results))
    returned = retriever.search_web("test query")

    assert [r['title'] for r in returned] == ['A', 'B']


def test_fetch_page_success_and_passage_rerank(retriever, mocker, monkeypatch):
    fake_trafilatura = SimpleNamespace(
        extract=lambda html, output_format="txt": "段落一です。\n段落二です。\n段落三です。"
    )
    monkeypatch.setitem(sys.modules, "trafilatura", fake_trafilatura)
    mocker.patch("src.retriever.FETCH_PAGE_ENABLED", True)
    mocker.patch("src.retriever.FETCH_PAGE_TOP_N", 1)
    mocker.patch("src.retriever.PASSAGE_CHARS", 8)
    mocker.patch.object(retriever._reranker, "_load")
    retriever._reranker._tokenizer = Mock(return_value={"input_ids": Mock()})
    retriever._reranker._model = Mock(
        side_effect=[
            SimpleNamespace(logits=[[0.9], [0.8]]),
            SimpleNamespace(logits=[[0.1], [0.7], [0.6], [0.2]]),
        ]
    )
    mocker.patch(
        "src.retriever.http_get_json",
        return_value=_mock_search_response(
            [
                {'title': 'Top', 'content': 'snippet-top', 'url': 'http://top'},
                {'title': 'Second', 'content': 'snippet-second', 'url': 'http://second'},
            ]
        ),
    )
    mocker.patch("src.retriever.http_get_text", return_value="<html>dummy</html>")

    returned = retriever.search_web("test query")

    assert returned[0]['content'] == '段落一です。'
    assert any(item['content'] == 'snippet-second' for item in returned)


def test_fetch_page_failure_falls_back_to_snippet(retriever, mocker, monkeypatch):
    fake_trafilatura = SimpleNamespace(
        extract=lambda html, output_format="txt": "抽出本文です。"
    )
    monkeypatch.setitem(sys.modules, "trafilatura", fake_trafilatura)
    mocker.patch("src.retriever.FETCH_PAGE_ENABLED", True)
    mocker.patch("src.retriever.FETCH_PAGE_TOP_N", 2)
    mocker.patch.object(retriever._reranker, "_load")
    retriever._reranker._tokenizer = Mock(return_value={"input_ids": Mock()})
    retriever._reranker._model = Mock(
        side_effect=[
            SimpleNamespace(logits=[[0.9], [0.8]]),
            SimpleNamespace(logits=[[0.4], [0.3]]),
        ]
    )
    mocker.patch(
        "src.retriever.http_get_json",
        return_value=_mock_search_response(
            [
                {'title': 'First', 'content': 'first snippet', 'url': 'http://first'},
                {'title': 'Second', 'content': 'second snippet', 'url': 'http://second'},
            ]
        ),
    )
    mocker.patch(
        "src.retriever.http_get_text",
        side_effect=[SearchError("timeout"), "<html>ok</html>"],
    )

    returned = retriever.search_web("test query")

    assert any(item['content'] == 'first snippet' for item in returned)
    assert any(item['content'] == '抽出本文です。' for item in returned)


def test_passage_rerank_failure_degrades_to_unranked_passages(retriever, mocker, monkeypatch):
    fake_trafilatura = SimpleNamespace(
        extract=lambda html, output_format="txt": "抽出本文です。"
    )
    monkeypatch.setitem(sys.modules, "trafilatura", fake_trafilatura)
    mocker.patch("src.retriever.FETCH_PAGE_ENABLED", True)
    mocker.patch("src.retriever.FETCH_PAGE_TOP_N", 1)
    mocker.patch("src.retriever.RERANK_MIN_SCORE", 0.1)
    mocker.patch.object(retriever._reranker, "_load")
    retriever._reranker._tokenizer = Mock(return_value={"input_ids": Mock()})
    retriever._reranker._model = Mock(
        side_effect=[
            SimpleNamespace(logits=[[0.9]]),
            RuntimeError("second-stage failed"),
        ]
    )
    mocker.patch(
        "src.retriever.http_get_json",
        return_value=_mock_search_response(
            [{'title': 'Only', 'content': 'only snippet', 'url': 'http://only'}]
        ),
    )
    mocker.patch("src.retriever.http_get_text", return_value="<html>ok</html>")

    returned = retriever.search_web("test query")

    assert returned == [{'title': 'Only', 'content': '抽出本文です。', 'url': 'http://only'}]
    assert retriever._reranker._unavailable is True


def test_rerank_min_score_filters_and_allows_empty(mocker):
    reranker = _Reranker()
    mocker.patch("src.retriever.RERANK_MIN_SCORE", 0.5)
    mocker.patch("src.retriever.RERANK_TOP_K", 3)
    mocker.patch.object(reranker, "_load")
    reranker._tokenizer = Mock(return_value={"input_ids": Mock()})

    reranker._model = Mock(return_value=SimpleNamespace(logits=[[0.2], [0.8]]))
    filtered = reranker.rerank(
        "q",
        [
            {'title': 'Low', 'content': 'low', 'url': 'http://low'},
            {'title': 'High', 'content': 'high', 'url': 'http://high'},
        ],
    )
    assert [item['title'] for item in filtered] == ['High']

    reranker._model = Mock(return_value=SimpleNamespace(logits=[[0.1], [0.2]]))
    assert reranker.rerank(
        "q",
        [
            {'title': 'A', 'content': 'a', 'url': 'http://a'},
            {'title': 'B', 'content': 'b', 'url': 'http://b'},
        ],
    ) == []


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
