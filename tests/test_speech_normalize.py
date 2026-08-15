#!/usr/bin/env python3
"""Tests for spoken-form latin normalization."""

from src.speech_normalize import normalize_for_speech


def test_normalize_replaces_units_and_ok():
    assert normalize_for_speech("信濃川の長さは367kmです。") == (
        "信濃川の長さは367キロメートルです。"
    )
    assert normalize_for_speech("OKです。") == "オーケーです。"
    assert normalize_for_speech("100gです。") == "100グラムです。"
    assert normalize_for_speech("333mです。") == "333メートルです。"
    assert normalize_for_speech("5msです。") == "5ミリ秒です。"


def test_normalize_replaces_known_acronyms_and_fullwidth():
    assert "エーティーピー" in normalize_for_speech("ATPを作ります。")
    assert "ディーエヌエー" in normalize_for_speech("DNAです。")
    assert "オーケー" in normalize_for_speech("ＯＫ")


def test_normalize_spells_leftover_latin_letters():
    assert normalize_for_speech("AIです。") == "エーアイです。"
    assert "A" not in normalize_for_speech("AIです。")


def test_normalize_is_idempotent_and_keeps_japanese():
    original = "富士山の高さは3776メートルです。"
    assert normalize_for_speech(original) == original
    once = normalize_for_speech("367kmです。")
    assert normalize_for_speech(once) == once


def test_normalize_empty_and_none():
    assert normalize_for_speech("") == ""
    assert normalize_for_speech(None) is None
