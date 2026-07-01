#!/usr/bin/env python3
"""
Custom exceptions for the voice assistant pipeline.

Provides distinct error types so callers can differentiate between
failure modes (e.g. search service down vs. no results found).
"""


class VoiceAssistantError(Exception):
    """Base exception for all voice assistant errors."""


class ListenerError(VoiceAssistantError):
    """Raised when audio recording or transcription fails."""


class SearchError(VoiceAssistantError):
    """Raised when the SearXNG search request fails."""


class GenerationError(VoiceAssistantError):
    """Raised when the Ollama AI generation fails."""


class SpeakerError(VoiceAssistantError):
    """Raised when text-to-speech synthesis or playback fails."""
