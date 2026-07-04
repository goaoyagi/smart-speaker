#!/usr/bin/env python3
"""
Tests for main orchestrator
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, call
import sys

# Mock external dependencies before import
sys.modules['faster_whisper'] = MagicMock()
sys.modules['piper'] = MagicMock()
sys.modules['gpiozero'] = MagicMock()

from src.main import VoiceAssistant, main  # noqa: E402
from src.status_led import LedState  # noqa: E402
from src.exceptions import ListenerError, SearchError, GenerationError  # noqa: E402


@pytest.fixture
def voice_assistant():
    """Create VoiceAssistant instance with mocked components"""
    with patch('src.main.Listener'), \
         patch('src.main.Retriever'), \
         patch('src.main.Composer'), \
         patch('src.main.Brain'), \
         patch('src.main.Speaker'), \
         patch('src.main.StatusLED'):
        assistant = VoiceAssistant()
        return assistant


def test_voice_assistant_initialization(voice_assistant):
    """Test that VoiceAssistant initializes all components"""
    assert voice_assistant.listener is not None
    assert voice_assistant.retriever is not None
    assert voice_assistant.composer is not None
    assert voice_assistant.brain is not None
    assert voice_assistant.speaker is not None
    assert voice_assistant.status_led is not None


def test_cleanup(voice_assistant):
    """Test cleanup method does not raise"""
    voice_assistant.cleanup()


def test_listen_and_respond_low_audio(voice_assistant):
    """Test listen_and_respond when audio level is too low (silence)"""
    silent_audio = np.zeros(16000, dtype=np.float32)
    voice_assistant.listener.record_audio.return_value = silent_audio

    voice_assistant.listen_and_respond()

    voice_assistant.speaker.speak.assert_called_once_with("音声が検出されませんでした。")
    assert voice_assistant.status_led.set_state.call_args_list == [
        call(LedState.LISTENING),
        call(LedState.IDLE),
    ]


def test_listen_and_respond_empty_transcription(voice_assistant):
    """Test listen_and_respond when transcription is empty"""
    audio = np.full(16000, 0.1, dtype=np.float32)
    voice_assistant.listener.record_audio.return_value = audio
    voice_assistant.listener.transcribe.return_value = "   "

    voice_assistant.listen_and_respond()

    voice_assistant.retriever.search_web.assert_not_called()
    assert voice_assistant.status_led.set_state.call_args_list == [
        call(LedState.LISTENING),
        call(LedState.IDLE),
    ]


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
    assert voice_assistant.status_led.set_state.call_args_list == [
        call(LedState.LISTENING),
        call(LedState.SEARCHING),
        call(LedState.THINKING),
        call(LedState.SPEAKING),
        call(LedState.IDLE),
    ]


def test_listen_and_respond_recording_failure(voice_assistant):
    """Test that ListenerError propagates from record_audio"""
    voice_assistant.listener.record_audio.side_effect = ListenerError("mic broken")
    with pytest.raises(ListenerError):
        voice_assistant.listen_and_respond()
    assert voice_assistant.status_led.set_state.call_args_list == [
        call(LedState.LISTENING),
        call(LedState.ERROR),
    ]


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
    assert voice_assistant.status_led.set_state.call_args_list == [
        call(LedState.LISTENING),
        call(LedState.SEARCHING),
        call(LedState.THINKING),
        call(LedState.SPEAKING),
        call(LedState.IDLE),
    ]


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
    assert voice_assistant.status_led.set_state.call_args_list == [
        call(LedState.LISTENING),
        call(LedState.SEARCHING),
        call(LedState.THINKING),
        call(LedState.ERROR),
    ]


def test_main_function():
    """Test the main() entry point"""
    with patch('src.main.Listener'), \
         patch('src.main.Retriever'), \
         patch('src.main.Composer'), \
         patch('src.main.Brain'), \
         patch('src.main.Speaker'), \
         patch('src.main.StatusLED'):
        with patch.object(VoiceAssistant, 'listen_and_respond') as mock_listen, \
             patch.object(VoiceAssistant, 'cleanup') as mock_cleanup:
            main()
            mock_listen.assert_called_once()
            mock_cleanup.assert_called_once()


def test_main_function_calls_cleanup_on_exception():
    """Test that main() calls cleanup even when listen_and_respond raises"""
    with patch('src.main.Listener'), \
         patch('src.main.Retriever'), \
         patch('src.main.Composer'), \
         patch('src.main.Brain'), \
         patch('src.main.Speaker'), \
         patch('src.main.StatusLED'):
        with patch.object(VoiceAssistant, 'listen_and_respond',
                          side_effect=ListenerError("fail")), \
             patch.object(VoiceAssistant, 'cleanup') as mock_cleanup:
            with pytest.raises(SystemExit):
                main()
            mock_cleanup.assert_called_once()
