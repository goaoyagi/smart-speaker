#!/usr/bin/env python3
"""
Tests for WakeWordDetector module
"""

import pytest
from unittest.mock import patch
from src.wake_word import WakeWordDetector


@pytest.fixture
def detector():
    """WakeWordDetector with two explicit wake words."""
    return WakeWordDetector(wake_words=["オッケースピーカー", "ねえスピーカー"])


class TestInit:
    def test_wake_words_stored(self, detector):
        """Provided wake words should be accessible via the property."""
        assert "オッケースピーカー" in detector.wake_words
        assert "ねえスピーカー" in detector.wake_words

    def test_normalised_to_lower_case(self):
        """Wake words are normalised to lower-case internally."""
        d = WakeWordDetector(wake_words=["Hello", "HEY"])
        assert "hello" in d.wake_words
        assert "hey" in d.wake_words

    def test_empty_wake_words(self):
        """Empty wake-words list should not raise; detect should return False."""
        d = WakeWordDetector(wake_words=[])
        assert d.detect("オッケースピーカー") is False

    def test_whitespace_entries_stripped(self):
        """Entries that are only whitespace should be ignored."""
        d = WakeWordDetector(wake_words=["  ", "hello", ""])
        assert d.wake_words == ["hello"]

    def test_uses_config_when_no_arg(self):
        """When wake_words is not provided, WAKE_WORDS from config is used."""
        with patch("src.wake_word.WAKE_WORDS", ["テスト"]):
            d = WakeWordDetector()
        assert "テスト" in d.wake_words


class TestDetect:
    def test_exact_match_returns_true(self, detector):
        """Exact wake word should be detected."""
        assert detector.detect("オッケースピーカー") is True

    def test_wake_word_in_sentence_returns_true(self, detector):
        """Wake word embedded in a sentence should be detected."""
        assert detector.detect("ねえスピーカー、今日の天気を教えて") is True

    def test_no_match_returns_false(self, detector):
        """Unrelated text should not trigger detection."""
        assert detector.detect("今日の天気はどうですか") is False

    def test_empty_text_returns_false(self, detector):
        """Empty string should not trigger detection."""
        assert detector.detect("") is False

    def test_none_text_returns_false(self, detector):
        """None should not trigger detection."""
        assert detector.detect(None) is False

    def test_case_insensitive_latin(self):
        """Latin wake words should match regardless of case."""
        d = WakeWordDetector(wake_words=["hey"])
        assert d.detect("HEY speaker") is True
        assert d.detect("Hey speaker") is True

    def test_second_wake_word_detected(self, detector):
        """Both configured wake words should trigger detection."""
        assert detector.detect("ねえスピーカー起きて") is True

    def test_partial_word_does_not_match_latin(self):
        """A wake word that is a substring of another word should not match."""
        d = WakeWordDetector(wake_words=["hey"])
        assert d.detect("heyday is great") is False

    def test_whitespace_only_text_returns_false(self, detector):
        """Whitespace-only input should not trigger detection."""
        assert detector.detect("   ") is False
