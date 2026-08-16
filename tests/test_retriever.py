#!/usr/bin/env python3
"""
Tests for retriever module
"""

import pytest
import requests
from unittest.mock import Mock, patch
from types import SimpleNamespace
from src.retriever import Retriever, _Reranker, _split_into_passages, _clip_to_char_budget
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
    """Test successful web search"""
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


def test_rerank_min_score_filters_low_scores(mocker):
    reranker = _Reranker()
    mocker.patch.object(reranker, "_load")
    reranker._tokenizer = Mock(return_value={"input_ids": Mock()})
    reranker._model = Mock(return_value=SimpleNamespace(logits=[[0.1], [0.9], [-0.2]]))

    results = [
        {'title': 'Low', 'content': 'low', 'url': 'http://low'},
        {'title': 'High', 'content': 'high', 'url': 'http://high'},
        {'title': 'Neg', 'content': 'neg', 'url': 'http://neg'},
    ]
    ranked = reranker.rerank("test query", results, top_k=0, min_score=0.0)
    assert [r['title'] for r in ranked] == ['High', 'Low']


def test_rerank_min_score_all_filtered_returns_empty(mocker):
    reranker = _Reranker()
    mocker.patch.object(reranker, "_load")
    reranker._tokenizer = Mock(return_value={"input_ids": Mock()})
    reranker._model = Mock(return_value=SimpleNamespace(logits=[[0.1], [0.2]]))

    results = [
        {'title': 'A', 'content': 'a', 'url': 'http://a'},
        {'title': 'B', 'content': 'b', 'url': 'http://b'},
    ]
    ranked = reranker.rerank("test query", results, top_k=0, min_score=0.5)
    assert ranked == []


def test_split_into_passages_respects_size_and_boundaries():
    text = "あ" * 100 + "。" + "い" * 100 + "。" + "う" * 100 + "。"
    passages = _split_into_passages(text, 150)
    assert len(passages) >= 2
    assert "".join(passages).replace("", "") != ""
    for passage in passages:
        assert passage
        assert len(passage) <= 150


def test_split_into_passages_empty_returns_empty_list():
    assert _split_into_passages("", 250) == []
    assert _split_into_passages(None, 250) == []
    assert _split_into_passages("   ", 250) == []
    assert _split_into_passages(123, 250) == []


def test_split_into_passages_shorter_than_size_returned_as_single_passage():
    text = "富士山は日本で最も高い山です。"
    assert _split_into_passages(text, 250) == [text]


def test_split_into_passages_hard_chunks_unbroken_text():
    """Text with no sentence/line delimiters must still be bounded by ``size``."""
    text = "あ" * 600
    passages = _split_into_passages(text, 250)
    assert [len(p) for p in passages] == [250, 250, 100]
    assert "".join(passages) == text


def test_split_into_passages_never_exceeds_size_across_many_sentences():
    sentence1 = "富士山は日本で最も高い山です。" * 5
    sentence2 = "標高は3776メートルあります。" * 5
    sentence3 = "静岡県と山梨県にまたがっています。" * 5
    text = sentence1 + sentence2 + sentence3
    passages = _split_into_passages(text, 100)
    assert len(passages) >= 3
    for passage in passages:
        assert len(passage) <= 100


def test_split_into_passages_respects_paragraph_breaks():
    text = "一段落目の文章です。" + "\n\n" + "二段落目の文章です。" * 20
    passages = _split_into_passages(text, 50)
    assert passages[0].startswith("一段落目の文章です。")
    for passage in passages:
        assert len(passage) <= 50


def test_split_into_passages_non_positive_size_falls_back_to_default(mocker):
    mocker.patch("src.retriever.PASSAGE_CHARS", 50)
    text = "あ" * 120
    passages = _split_into_passages(text, 0)
    assert all(len(p) <= 50 for p in passages)


def test_clip_to_char_budget_keeps_items_within_budget():
    items = [
        {"title": "A", "content": "x" * 10, "url": "http://a"},
        {"title": "B", "content": "x" * 10, "url": "http://b"},
        {"title": "C", "content": "x" * 10, "url": "http://c"},
    ]
    kept = _clip_to_char_budget(items, budget=25)
    assert [item["title"] for item in kept] == ["A", "B"]


def test_clip_to_char_budget_always_keeps_at_least_one_item():
    items = [
        {"title": "Huge", "content": "x" * 1000, "url": "http://huge"},
        {"title": "Second", "content": "x" * 1000, "url": "http://second"},
    ]
    kept = _clip_to_char_budget(items, budget=10)
    assert [item["title"] for item in kept] == ["Huge"]


def test_clip_to_char_budget_empty_input_returns_empty():
    assert _clip_to_char_budget([], budget=100) == []


def test_clip_to_char_budget_uses_context_char_budget_by_default(mocker):
    mocker.patch("src.retriever.CONTEXT_CHAR_BUDGET", 15)
    items = [
        {"title": "A", "content": "x" * 10, "url": "http://a"},
        {"title": "B", "content": "x" * 10, "url": "http://b"},
    ]
    kept = _clip_to_char_budget(items)
    assert [item["title"] for item in kept] == ["A"]


def test_fetch_page_disabled_skips_fetch(retriever, mocker):
    mocker.patch("src.retriever.FETCH_PAGE_ENABLED", False)
    mocker.patch("src.retriever.RERANKER_ENABLED", False)
    fetch = mocker.patch.object(retriever, "_fetch_passages_for_result")
    results = [{'title': 'A', 'content': 'a', 'url': 'http://a'}]
    with patch('src.http_client.requests.get', return_value=_mock_search_response(results)):
        returned = retriever.search_web("test query")

    assert returned == results
    fetch.assert_not_called()


def test_fetch_and_rerank_passages_uses_extracted_text(retriever, mocker):
    mocker.patch("src.retriever.RERANKER_ENABLED", False)
    mocker.patch("src.retriever.FETCH_PAGE_ENABLED", True)
    mocker.patch("src.retriever.FETCH_PAGE_TOP_N", 2)
    mocker.patch.object(Retriever, "_extract_text", return_value="本文" * 200)
    mocker.patch("src.retriever.http_get_text", return_value="<html>dummy</html>")

    results = [
        {'title': 'A', 'content': 'snippet-a', 'url': 'http://a'},
        {'title': 'B', 'content': 'snippet-b', 'url': 'http://b'},
    ]
    with patch('src.http_client.requests.get', return_value=_mock_search_response(results)):
        returned = retriever.search_web("test query")

    assert len(returned) > 0
    assert all(r['content'] not in ('snippet-a', 'snippet-b') for r in returned)
    assert all(r['url'] in ('http://a', 'http://b') for r in returned)


def test_fetch_partial_failure_falls_back_to_snippet(retriever, mocker):
    mocker.patch("src.retriever.RERANKER_ENABLED", False)
    mocker.patch("src.retriever.FETCH_PAGE_ENABLED", True)
    mocker.patch("src.retriever.FETCH_PAGE_TOP_N", 2)

    def fake_get_text(url, *args, **kwargs):
        if url == 'http://bad':
            raise SearchError("boom")
        return "<html>good</html>"

    mocker.patch("src.retriever.http_get_text", side_effect=fake_get_text)
    mocker.patch.object(Retriever, "_extract_text", return_value="本文" * 200)

    results = [
        {'title': 'Bad', 'content': 'bad-snippet', 'url': 'http://bad'},
        {'title': 'Good', 'content': 'good-snippet', 'url': 'http://good'},
    ]
    with patch('src.http_client.requests.get', return_value=_mock_search_response(results)):
        returned = retriever.search_web("test query")

    assert any(r['url'] == 'http://bad' and r['content'] == 'bad-snippet' for r in returned)
    assert any(r['url'] == 'http://good' and r['content'] != 'good-snippet' for r in returned)


def test_fetch_passages_for_result_skips_invalid_url(retriever):
    assert retriever._fetch_passages_for_result({'title': 'x', 'content': 'y', 'url': ''}) is None
    assert retriever._fetch_passages_for_result(
        {'title': 'x', 'content': 'y', 'url': 'ftp://bad-scheme'}
    ) is None


def test_fetch_and_rerank_passages_clips_to_context_char_budget(retriever, mocker):
    """The final passage list must fit CONTEXT_CHAR_BUDGET, even when the
    reranker (disabled here) would otherwise return every passage unpruned.
    """
    mocker.patch("src.retriever.RERANKER_ENABLED", False)
    mocker.patch("src.retriever.FETCH_PAGE_ENABLED", True)
    mocker.patch("src.retriever.FETCH_PAGE_TOP_N", 2)
    mocker.patch("src.retriever.PASSAGE_CHARS", 50)
    mocker.patch("src.retriever.CONTEXT_CHAR_BUDGET", 120)
    mocker.patch.object(Retriever, "_extract_text", return_value="長い本文です。" * 30)
    mocker.patch("src.retriever.http_get_text", return_value="<html>dummy</html>")

    results = [
        {'title': 'A', 'content': 'snippet-a', 'url': 'http://a'},
        {'title': 'B', 'content': 'snippet-b', 'url': 'http://b'},
    ]
    with patch('src.http_client.requests.get', return_value=_mock_search_response(results)):
        returned = retriever.search_web("test query")

    total_chars = sum(len(r["title"]) + len(r["content"]) for r in returned)
    assert len(returned) >= 1
    assert total_chars <= 120 or len(returned) == 1


def test_fetch_and_rerank_passages_keeps_single_oversized_passage(retriever, mocker):
    """A budget smaller than even the first passage must not empty the result."""
    mocker.patch("src.retriever.RERANKER_ENABLED", False)
    mocker.patch("src.retriever.FETCH_PAGE_ENABLED", True)
    mocker.patch("src.retriever.FETCH_PAGE_TOP_N", 1)
    mocker.patch("src.retriever.PASSAGE_CHARS", 500)
    mocker.patch("src.retriever.CONTEXT_CHAR_BUDGET", 5)
    mocker.patch.object(Retriever, "_extract_text", return_value="長い本文です。" * 30)
    mocker.patch("src.retriever.http_get_text", return_value="<html>dummy</html>")

    results = [{'title': 'A', 'content': 'snippet-a', 'url': 'http://a'}]
    with patch('src.http_client.requests.get', return_value=_mock_search_response(results)):
        returned = retriever.search_web("test query")

    assert len(returned) == 1
