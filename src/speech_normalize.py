#!/usr/bin/env python3
"""
Deterministic spoken-form normalization for Piper TTS.

Generation may emit SI units or short acronyms. Piper is more stable when
those are Japanese readings, so this module rewrites them without an LLM.
"""

import logging
import re

logger = logging.getLogger(__name__)

_FULLWIDTH_LATIN = str.maketrans(
    "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"
    "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
)

# Longer keys first so "https" wins over "http", "ms" over "m", "km" over "m".
_WORD_REPLACEMENTS = (
    ("https", "エイチティーティーピーエス"),
    ("http", "エイチティーティーピー"),
    ("www", "ダブリューダブリューダブリュー"),
    ("atp", "エーティーピー"),
    ("dna", "ディーエヌエー"),
    ("rna", "アールエヌエー"),
    ("url", "ユーアールエル"),
    ("km", "キロメートル"),
    ("cm", "センチメートル"),
    ("mm", "ミリメートル"),
    ("ms", "ミリ秒"),
    ("kg", "キログラム"),
    ("mg", "ミリグラム"),
    ("hz", "ヘルツ"),
    ("ok", "オーケー"),
)

_LETTER_READING = {
    "A": "エー",
    "B": "ビー",
    "C": "シー",
    "D": "ディー",
    "E": "イー",
    "F": "エフ",
    "G": "ジー",
    "H": "エイチ",
    "I": "アイ",
    "J": "ジェー",
    "K": "ケー",
    "L": "エル",
    "M": "エム",
    "N": "エヌ",
    "O": "オー",
    "P": "ピー",
    "Q": "キュー",
    "R": "アール",
    "S": "エス",
    "T": "ティー",
    "U": "ユー",
    "V": "ブイ",
    "W": "ダブリュー",
    "X": "エックス",
    "Y": "ワイ",
    "Z": "ゼット",
}

_LATIN_RUN = re.compile(r"[A-Za-z]+")
_NUMBERED_METER = re.compile(r"(?<=\d)\s*m(?![A-Za-z])", re.IGNORECASE)
_NUMBERED_GRAM = re.compile(r"(?<=\d)\s*g(?![A-Za-z])", re.IGNORECASE)

# Pre-compiled patterns for word replacements.
_WORD_PATTERNS = tuple(
    (
        re.compile(
            rf"(?<![A-Za-z]){re.escape(source)}(?![A-Za-z])",
            re.IGNORECASE,
        ),
        target,
    )
    for source, target in _WORD_REPLACEMENTS
)


def _spell_latin_run(match):
    return "".join(_LETTER_READING[char] for char in match.group(0).upper())


def normalize_for_speech(text):
    """Rewrite latin units and leftover letters into Japanese readings."""
    if not text:
        return text

    normalized = text.translate(_FULLWIDTH_LATIN)
    for pattern, target in _WORD_PATTERNS:
        normalized = pattern.sub(target, normalized)
    normalized = _NUMBERED_METER.sub("メートル", normalized)
    normalized = _NUMBERED_GRAM.sub("グラム", normalized)
    normalized = _LATIN_RUN.sub(_spell_latin_run, normalized)

    if normalized != text:
        logger.info("Normalized spoken text: %s", normalized)
    return normalized
