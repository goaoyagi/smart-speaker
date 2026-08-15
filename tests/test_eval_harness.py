#!/usr/bin/env python3
"""
Tests for the pure scoring and reporting helpers in scripts/eval.py.

This is a cross-cutting test rather than a mirror of one ``src/`` module.
The evaluation runner itself is intentionally not exercised, so these tests
never connect to SearXNG or Ollama.
"""

import importlib.util
import json
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_PATH = PROJECT_ROOT / "scripts" / "eval.py"
CASES_PATH = PROJECT_ROOT / "scripts" / "eval_cases.json"


def _load_eval_module():
    spec = importlib.util.spec_from_file_location("eval_harness", EVAL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eval_harness = _load_eval_module()


def test_count_sentences_counts_japanese_and_ascii_punctuation():
    assert eval_harness.count_sentences("一文目です。二文目です！三文目です？") == 3
    assert eval_harness.count_sentences("") == 0


def test_find_latin_runs_detects_halfwidth_and_fullwidth_letters():
    assert eval_harness.find_latin_runs("ＡＩとAIを含む") == ["ＡＩ", "AI"]
    assert eval_harness.find_latin_runs("日本語だけです。") == []


def test_grade_answer_checks_sentence_bounds_and_keywords():
    spec = {
        "min_sentences": 2,
        "max_sentences": 3,
        "expect_keywords": ["東京", "晴れ"],
        "expect_any": [["傘", "雨具"]],
        "deny_keywords": ["雪"],
    }

    passed, failures = eval_harness.grade_answer(
        spec,
        "東京は晴れです。傘は不要です。",
    )

    assert passed is True
    assert failures == []

    passed, failures = eval_harness.grade_answer(
        spec,
        "東京は雪です。雪が降っています。傘は必要です。もう一文です。",
    )

    assert passed is False
    assert any("文数が過多" in failure for failure in failures)
    assert any("期待キーワード欠落" in failure for failure in failures)
    assert any("禁止キーワード出現" in failure for failure in failures)


def test_grade_answer_detects_latin_and_missing_any_keyword():
    passed, failures = eval_harness.grade_answer(
        {"min_sentences": 1, "expect_any": [["東京", "大阪"]]},
        "AIについて説明します。",
    )

    assert passed is False
    assert any("アルファベット混入" in failure for failure in failures)
    assert any("いずれか" in failure for failure in failures)


@pytest.mark.parametrize(
    ("previous_answer", "answer", "expected_pass"),
    [
        ("直前の回答です。", "直前の回答です。", True),
        ("直前の回答です。", "別の回答です。", False),
        (None, "回答です。", False),
    ],
)
def test_grade_answer_repeat_matches_mismatches_and_handles_empty_history(
    previous_answer, answer, expected_pass
):
    passed, failures = eval_harness.grade_answer(
        {"min_sentences": 1, "expect_repeat": True},
        answer,
        previous_answer,
    )

    assert passed is expected_pass
    if expected_pass:
        assert failures == []
    else:
        assert failures


def test_summarize_reports_pass_rate_categories_degraded_and_latin_violations():
    results = [
        {
            "category": "fact",
            "answer": "回答。",
            "passed": True,
            "degraded": False,
            "latin_runs": [],
            "answer_chars": 3,
            "sentences": 1,
            "search_ms": 10,
            "generate_ms": 20,
            "total_ms": 30,
            "warnings": [],
        },
        {
            "category": "fact",
            "answer": "AI。",
            "passed": False,
            "degraded": True,
            "latin_runs": ["AI"],
            "answer_chars": 3,
            "sentences": 1,
            "search_ms": 30,
            "generate_ms": 40,
            "total_ms": 70,
            "warnings": ["検索失敗"],
        },
        {
            "category": "followup",
            "answer": None,
            "passed": False,
            "degraded": False,
            "latin_runs": [],
            "answer_chars": 0,
            "sentences": 0,
            "search_ms": 0,
            "generate_ms": 0,
            "total_ms": 0,
            "warnings": [],
        },
    ]

    summary = eval_harness.summarize(results)

    assert summary["turns"] == 3
    assert summary["passed"] == 1
    assert summary["pass_rate"] == pytest.approx(1 / 3, abs=0.001)
    assert summary["errors"] == 1
    assert summary["degraded"] == 1
    assert summary["latin_violations"] == 1
    assert summary["warnings"] == 1
    assert summary["by_category"]["fact"] == {
        "turns": 2,
        "passed": 1,
        "pass_rate": 0.5,
    }
    assert summary["by_category"]["followup"]["pass_rate"] == 0.0


def test_diff_summaries_returns_deltas_for_counts_and_medians():
    baseline = {
        "pass_rate": 0.5,
        "latin_violations": 2,
        "degraded": 1,
        "warnings": 3,
        "answer_chars": {"median": 100},
        "sentences": {"median": 2},
        "generate_ms": {"median": 500},
        "total_ms": {"median": 700},
    }
    current = {
        "pass_rate": 0.75,
        "latin_violations": 1,
        "degraded": 0,
        "warnings": 2,
        "answer_chars": {"median": 120},
        "sentences": {"median": 3},
        "generate_ms": {"median": 450},
        "total_ms": {"median": 650},
    }

    diff = eval_harness.diff_summaries(baseline, current)

    assert diff["pass_rate"]["delta"] == 0.25
    assert diff["latin_violations"]["delta"] == -1
    assert diff["degraded"]["delta"] == -1
    assert diff["warnings"]["delta"] == -1
    assert diff["answer_chars_median"]["delta"] == 20
    assert diff["generate_ms_median"]["delta"] == -50


def test_select_cases_filters_by_id_and_category():
    cases = [
        {"id": "fact-01", "category": "fact"},
        {"id": "follow-01", "category": "followup"},
    ]

    assert eval_harness.select_cases(cases, only="fact-01") == [cases[0]]
    assert eval_harness.select_cases(cases, categories="followup") == [cases[1]]


def test_select_cases_rejects_unknown_id():
    with pytest.raises(ValueError, match="存在しないケースID"):
        eval_harness.select_cases([{"id": "fact-01"}], only="missing")


def test_load_cases_contains_unique_ids_and_turns():
    cases = eval_harness.load_cases(CASES_PATH)
    raw_cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]

    assert cases == raw_cases
    assert cases
    assert all(case.get("id") and case.get("turns") for case in cases)
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids))
