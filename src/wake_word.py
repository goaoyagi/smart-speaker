#!/usr/bin/env python3
"""
WakeWordDetector module - Keyword-based wake word detection

Listens to a short audio recording, transcribes it via Whisper, and checks
whether any of the configured wake words appear in the transcription.

This approach requires no additional dependencies beyond the existing Listener
(Whisper) already present in the project.  For production deployments, a
dedicated always-on engine such as OpenWakeWord or Porcupine can be
substituted by replacing the `detect_from_audio` method.
"""

import logging
import re

from .config import WAKE_WORDS

logger = logging.getLogger(__name__)


class WakeWordDetector:
    """Detect wake words inside a transcribed text string.

    Args:
        wake_words: Iterable of trigger keywords/phrases (case-insensitive).
            Falls back to :data:`src.config.WAKE_WORDS` when *None*.
    """

    def __init__(self, wake_words=None):
        if wake_words is None:
            wake_words = WAKE_WORDS
        # Normalise to lower-case strings and strip whitespace
        self._wake_words = [w.strip().lower() for w in wake_words if w.strip()]
        if not self._wake_words:
            logger.warning(
                "WakeWordDetector initialised with no wake words; "
                "every utterance will be ignored."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def wake_words(self) -> list:
        """Return the list of normalised wake word strings."""
        return list(self._wake_words)

    def detect(self, text: str) -> bool:
        """Return *True* if any wake word is found in *text*.

        For ASCII-only wake words, word-boundary matching is used to avoid
        false positives (e.g. ``"hey"`` matching ``"heyday"``).  For
        non-ASCII wake words (e.g. Japanese), substring matching is used
        because those languages do not use spaces as word boundaries.

        Args:
            text: Transcribed speech string to search.

        Returns:
            ``True`` when at least one wake word is present, ``False``
            otherwise.
        """
        if not text or not self._wake_words:
            return False

        normalised = text.lower()
        for word in self._wake_words:
            if word.isascii():
                # Word-boundary match for Latin-script keywords
                if re.search(r"\b" + re.escape(word) + r"\b", normalised):
                    logger.info("Wake word detected: %r", word)
                    return True
            else:
                # Substring match for non-ASCII (e.g. Japanese) keywords
                if word in normalised:
                    logger.info("Wake word detected: %r", word)
                    return True

        return False
