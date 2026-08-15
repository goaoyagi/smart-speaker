#!/usr/bin/env python3
"""Tests for the shared text-in / text-out conversation turn."""

from unittest.mock import Mock

import pytest

from src.exceptions import GenerationError, SearchError
from src.text_turn import run_text_turn


def _components():
    retriever = Mock()
    retriever.search_web.return_value = [{"title": "A", "content": "本文"}]
    composer = Mock()
    composer.compose_messages.return_value = [{"role": "user", "content": "prompt"}]
    brain = Mock()
    brain.generate_response.return_value = "木星です。"
    query_prep = Mock()
    query_prep.prepare.return_value = (True, "太陽系 一番大きい惑星")
    return retriever, composer, brain, query_prep


def test_run_text_turn_rewrites_query_then_generates_and_normalizes():
    retriever, composer, brain, query_prep = _components()
    brain.generate_response.return_value = "367kmです。"
    phases = []

    result = run_text_turn(
        "長さはどれくらいですか？",
        [{"role": "user", "content": "信濃川"}],
        retriever=retriever,
        composer=composer,
        brain=brain,
        query_prep=query_prep,
        on_phase=phases.append,
    )

    query_prep.prepare.assert_called_once()
    retriever.search_web.assert_called_once_with("太陽系 一番大きい惑星")
    composer.compose_messages.assert_called_once_with(
        "長さはどれくらいですか？",
        [{"title": "A", "content": "本文"}],
        [{"role": "user", "content": "信濃川"}],
    )
    assert result["answer"] == "367キロメートルです。"
    assert result["skipped_search"] is False
    assert phases == ["prep", "search", "generate"]


def test_run_text_turn_skips_search_when_gate_says_so():
    retriever, composer, brain, query_prep = _components()
    query_prep.prepare.return_value = (False, "こんにちは")
    phases = []

    result = run_text_turn(
        "こんにちは",
        [],
        retriever=retriever,
        composer=composer,
        brain=brain,
        query_prep=query_prep,
        on_phase=phases.append,
    )

    retriever.search_web.assert_not_called()
    composer.compose_messages.assert_called_once_with("こんにちは", [], [])
    assert result["skipped_search"] is True
    assert result["search_results"] == []
    assert phases == ["prep", "generate"]


def test_run_text_turn_search_failure_degrades_to_empty_context():
    retriever, composer, brain, query_prep = _components()
    retriever.search_web.side_effect = SearchError("offline")

    result = run_text_turn(
        "質問",
        [],
        retriever=retriever,
        composer=composer,
        brain=brain,
        query_prep=query_prep,
    )

    composer.compose_messages.assert_called_once_with("質問", [], [])
    assert result["degraded"] is True
    assert result["answer"] == "木星です。"


def test_run_text_turn_generation_failure_propagates():
    retriever, composer, brain, query_prep = _components()
    brain.generate_response.side_effect = GenerationError("timeout")

    with pytest.raises(GenerationError, match="timeout"):
        run_text_turn(
            "質問",
            [],
            retriever=retriever,
            composer=composer,
            brain=brain,
            query_prep=query_prep,
        )


def test_run_text_turn_use_search_false_skips_prep_and_search():
    retriever, composer, brain, query_prep = _components()

    result = run_text_turn(
        "質問",
        [],
        retriever=retriever,
        composer=composer,
        brain=brain,
        query_prep=query_prep,
        use_search=False,
    )

    query_prep.prepare.assert_not_called()
    retriever.search_web.assert_not_called()
    assert result["skipped_search"] is False
    assert result["search_query"] == "質問"
