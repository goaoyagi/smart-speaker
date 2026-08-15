#!/usr/bin/env python3
"""Tests for search-need gating and query rewrite."""

from unittest.mock import Mock

from src.exceptions import GenerationError
from src.query_prep import QueryPrep, parse_prepare_output


def test_parse_prepare_output_need_with_rewritten_query():
    should_search, query = parse_prepare_output(
        "NEED\n東京タワー 完成年",
        "それはいつ完成しましたか？",
    )
    assert should_search is True
    assert query == "東京タワー 完成年"


def test_parse_prepare_output_skip():
    should_search, query = parse_prepare_output(
        "SKIP\nこんにちは。今日もよろしくお願いします。",
        "こんにちは。今日もよろしくお願いします。",
    )
    assert should_search is False
    assert query == "こんにちは。今日もよろしくお願いします。"


def test_parse_prepare_output_japanese_skip_and_non_string_fallback():
    should_search, query = parse_prepare_output("検索不要", "挨拶")
    assert should_search is False
    assert query == "挨拶"

    should_search, query = parse_prepare_output(None, "元の質問")
    assert should_search is True
    assert query == "元の質問"


def test_parse_prepare_output_single_line_need_with_query():
    should_search, query = parse_prepare_output(
        "NEED 富士山 標高",
        "その山の高さは？",
    )
    assert should_search is True
    assert query == "富士山 標高"


def test_query_prep_prepare_uses_auxiliary_call(mocker):
    brain = Mock()
    brain.generate_auxiliary.return_value = "NEED\n信濃川 長さ"
    prep = QueryPrep(brain)

    should_search, query = prep.prepare(
        "長さはどれくらいですか？",
        [{"role": "user", "content": "日本で一番長い川は？"}],
    )

    assert should_search is True
    assert query == "信濃川 長さ"
    messages = brain.generate_auxiliary.call_args[0][0]
    assert messages[0]["role"] == "system"
    assert "NEED" in messages[0]["content"]
    assert messages[-1]["content"] == "質問：長さはどれくらいですか？"
    assert messages[1]["content"] == "日本で一番長い川は？"


def test_query_prep_prepare_degrades_on_generation_error():
    brain = Mock()
    brain.generate_auxiliary.side_effect = GenerationError("timeout")
    prep = QueryPrep(brain)

    should_search, query = prep.prepare("富士山の高さは？")

    assert should_search is True
    assert query == "富士山の高さは？"


def test_query_prep_prepare_can_be_disabled(mocker):
    mocker.patch("src.query_prep.QUERY_PREP_ENABLED", False)
    brain = Mock()
    prep = QueryPrep(brain)

    should_search, query = prep.prepare("こんにちは")

    assert should_search is True
    assert query == "こんにちは"
    brain.generate_auxiliary.assert_not_called()
