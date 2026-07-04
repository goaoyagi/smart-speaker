#!/usr/bin/env python3
"""
Tests for main orchestrator
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock, call
import sys

# Mock external dependencies before import
sys.modules['faster_whisper'] = MagicMock()
sys.modules['piper'] = MagicMock()

from src.main import VoiceAssistant, main
from src.wake_word import WakeWordDetector
from src.exceptions import ListenerError, SearchError, GenerationError, SpeakerError


@pytest.fixture
def voice_assistant():
    """Create VoiceAssistant instance with mocked components"""
    with patch('src.main.Listener') as mock_listener_cls, \
         patch('src.main.Retriever') as mock_retriever_cls, \
         patch('src.main.Composer') as mock_composer_cls, \
         patch('src.main.Brain') as mock_brain_cls, \
         patch('src.main.Speaker') as mock_speaker_cls:
        assistant = VoiceAssistant()
        return assistant


def test_voice_assistant_initialization(voice_assistant):
    """Test that VoiceAssistant initializes all components"""
    assert voice_assistant.listener is not None
    assert voice_assistant.retriever is not None
    assert voice_assistant.composer is not None
    assert voice_assistant.brain is not None
    assert voice_assistant.speaker is not None


def test_voice_assistant_has_wake_word_detector(voice_assistant):
    """VoiceAssistant should expose a WakeWordDetector instance."""
    assert isinstance(voice_assistant.wake_word_detector, WakeWordDetector)


def test_cleanup(voice_assistant):
    """Test cleanup method does not raise"""
    voice_assistant.cleanup()


def test_listen_and_respond_low_audio(voice_assistant):
    """Test listen_and_respond when audio level is too low (silence)"""
    silent_audio = np.zeros(16000, dtype=np.float32)
    voice_assistant.listener.record_audio.return_value = silent_audio

    voice_assistant.listen_and_respond()

    voice_assistant.speaker.speak.assert_called_once_with("音声が検出されませんでした。")


def test_listen_and_respond_empty_transcription(voice_assistant):
    """Test listen_and_respond when transcription is empty"""
    audio = np.full(16000, 0.1, dtype=np.float32)
    voice_assistant.listener.record_audio.return_value = audio
    voice_assistant.listener.transcribe.return_value = "   "

    voice_assistant.listen_and_respond()

    voice_assistant.retriever.search_web.assert_not_called()


def test_listen_and_respond_successful_flow(voice_assistant):
    """Test the full successful conversation flow"""
    audio = np.full(16000, 0.5, dtype=np.float32)
    voice_assistant.listener.record_audio.return_value = audio
    voice_assistant.listener.transcribe.return_value = "今日の天気はどうですか"
    voice_assistant.retriever.search_web.return_value = [
        {'title': '天気予報', 'content': '晴れ', 'url': 'http://example.com'}
    ]
    voice_assistant.composer.compose_prompt.return_value = "プロンプト"
    voice_assistant.brain.generate_response.return_value = "今日は晴れです"

    voice_assistant.listen_and_respond()

    voice_assistant.listener.record_audio.assert_called_once()
    voice_assistant.listener.transcribe.assert_called_once_with(audio)
    voice_assistant.retriever.search_web.assert_called_once_with("今日の天気はどうですか")
    voice_assistant.composer.compose_prompt.assert_called_once()
    voice_assistant.brain.generate_response.assert_called_once_with("プロンプト")
    voice_assistant.speaker.speak.assert_called_once_with("今日は晴れです")


def test_listen_and_respond_recording_failure(voice_assistant):
    """Test that ListenerError propagates from record_audio"""
    voice_assistant.listener.record_audio.side_effect = ListenerError("mic broken")
    with pytest.raises(ListenerError):
        voice_assistant.listen_and_respond()


def test_listen_and_respond_search_failure_degrades_gracefully(voice_assistant):
    """Test that SearchError is caught and pipeline continues without context"""
    audio = np.random.uniform(-0.5, 0.5, 16000).astype(np.float32)
    voice_assistant.listener.record_audio.return_value = audio
    voice_assistant.listener.transcribe.return_value = "テスト"
    voice_assistant.retriever.search_web.side_effect = SearchError("offline")
    voice_assistant.composer.compose_prompt.return_value = "prompt"
    voice_assistant.brain.generate_response.return_value = "回答"

    voice_assistant.listen_and_respond()

    voice_assistant.composer.compose_prompt.assert_called_once_with("テスト", [])
    voice_assistant.speaker.speak.assert_called_once_with("回答")


def test_listen_and_respond_generation_failure(voice_assistant):
    """Test that GenerationError propagates from generate_response"""
    audio = np.random.uniform(-0.5, 0.5, 16000).astype(np.float32)
    voice_assistant.listener.record_audio.return_value = audio
    voice_assistant.listener.transcribe.return_value = "テスト"
    voice_assistant.retriever.search_web.return_value = []
    voice_assistant.composer.compose_prompt.return_value = "prompt"
    voice_assistant.brain.generate_response.side_effect = GenerationError("timeout")

    with pytest.raises(GenerationError):
        voice_assistant.listen_and_respond()


# ---------------------------------------------------------------------------
# run_continuous tests
# ---------------------------------------------------------------------------

def _make_audio(level=0.5):
    return np.full(16000, level, dtype=np.float32)


def test_run_continuous_triggers_listen_and_respond_on_wake_word(voice_assistant):
    """When wake word is detected, listen_and_respond should be called once."""
    # Simulate: first iteration detects wake word, second raises KeyboardInterrupt
    voice_assistant.listener.record_audio.side_effect = [
        _make_audio(0.5),   # wake word check
        _make_audio(0.5),   # full conversation recording (inside listen_and_respond)
        KeyboardInterrupt,  # stop loop
    ]
    voice_assistant.listener.transcribe.side_effect = [
        "オッケースピーカー",  # wake word check
        "今日の天気は",        # conversation transcription
    ]
    voice_assistant.wake_word_detector = WakeWordDetector(wake_words=["オッケースピーカー"])
    voice_assistant.retriever.search_web.return_value = []
    voice_assistant.composer.compose_prompt.return_value = "prompt"
    voice_assistant.brain.generate_response.return_value = "晴れです"

    with pytest.raises(KeyboardInterrupt):
        voice_assistant.run_continuous()

    # compose_prompt was called → listen_and_respond ran
    voice_assistant.composer.compose_prompt.assert_called_once()


def test_run_continuous_skips_when_no_wake_word(voice_assistant):
    """Utterances without the wake word should not trigger listen_and_respond."""
    call_count = 0

    def record_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count >= 3:
            raise KeyboardInterrupt
        return _make_audio(0.5)

    voice_assistant.listener.record_audio.side_effect = record_side_effect
    voice_assistant.listener.transcribe.return_value = "関係ない発話"
    voice_assistant.wake_word_detector = WakeWordDetector(wake_words=["オッケースピーカー"])

    with pytest.raises(KeyboardInterrupt):
        voice_assistant.run_continuous()

    voice_assistant.composer.compose_prompt.assert_not_called()


def test_run_continuous_skips_silent_audio(voice_assistant):
    """Silent audio clips should be skipped without transcription."""
    call_count = 0

    def record_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count >= 3:
            raise KeyboardInterrupt
        return np.zeros(16000, dtype=np.float32)

    voice_assistant.listener.record_audio.side_effect = record_side_effect

    with pytest.raises(KeyboardInterrupt):
        voice_assistant.run_continuous()

    voice_assistant.listener.transcribe.assert_not_called()


def test_run_continuous_continues_after_recording_error(voice_assistant):
    """A ListenerError during wake word recording should not crash the loop."""
    call_count = 0

    def record_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ListenerError("mic error")
        raise KeyboardInterrupt

    voice_assistant.listener.record_audio.side_effect = record_side_effect

    with pytest.raises(KeyboardInterrupt):
        voice_assistant.run_continuous()

    # Loop survived the ListenerError
    assert call_count == 2


def test_run_continuous_continues_after_pipeline_error(voice_assistant):
    """A GenerationError inside listen_and_respond should not crash the loop."""
    call_count = 0

    def record_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return _make_audio(0.5)
        raise KeyboardInterrupt

    def transcribe_side_effect(audio):
        if call_count == 1:
            return "オッケースピーカー"
        return "今日は何日"

    voice_assistant.listener.record_audio.side_effect = record_side_effect
    voice_assistant.listener.transcribe.side_effect = transcribe_side_effect
    voice_assistant.wake_word_detector = WakeWordDetector(wake_words=["オッケースピーカー"])
    voice_assistant.retriever.search_web.return_value = []
    voice_assistant.composer.compose_prompt.return_value = "prompt"
    voice_assistant.brain.generate_response.side_effect = GenerationError("timeout")

    with pytest.raises(KeyboardInterrupt):
        voice_assistant.run_continuous()

    # Loop continued past the GenerationError
    assert call_count >= 2


def test_main_function_calls_run_continuous():
    """main() should call run_continuous (not listen_and_respond) in Phase C."""
    with patch('src.main.Listener'), \
         patch('src.main.Retriever'), \
         patch('src.main.Composer'), \
         patch('src.main.Brain'), \
         patch('src.main.Speaker'):
        with patch.object(VoiceAssistant, 'run_continuous',
                          side_effect=KeyboardInterrupt) as mock_run_continuous, \
             patch.object(VoiceAssistant, 'cleanup') as mock_cleanup:
            main()
            mock_run_continuous.assert_called_once()
            mock_cleanup.assert_called_once()


def test_main_function_calls_cleanup_on_keyboard_interrupt():
    """main() should call cleanup even when interrupted."""
    with patch('src.main.Listener'), \
         patch('src.main.Retriever'), \
         patch('src.main.Composer'), \
         patch('src.main.Brain'), \
         patch('src.main.Speaker'):
        with patch.object(VoiceAssistant, 'run_continuous',
                          side_effect=KeyboardInterrupt), \
             patch.object(VoiceAssistant, 'cleanup') as mock_cleanup:
            main()
            mock_cleanup.assert_called_once()
