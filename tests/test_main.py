#!/usr/bin/env python3
"""
Tests for main orchestrator
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys

# Mock external dependencies before import
sys.modules['faster_whisper'] = MagicMock()
sys.modules['piper'] = MagicMock()

# Mock the bare module imports used by src/main.py
sys.modules['listener'] = MagicMock()
sys.modules['retriever'] = MagicMock()
sys.modules['composer'] = MagicMock()
sys.modules['brain'] = MagicMock()
sys.modules['speaker'] = MagicMock()

from src.main import VoiceAssistant, main
import numpy as np


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


def test_cleanup(voice_assistant):
    """Test cleanup method does not raise"""
    voice_assistant.cleanup()


def test_listen_and_respond_low_audio(voice_assistant):
    """Test listen_and_respond when audio level is too low (silence)"""
    # Simulate very quiet audio (below 0.03 threshold)
    silent_audio = np.zeros(16000, dtype=np.float32)
    voice_assistant.listener.record_audio.return_value = silent_audio

    voice_assistant.listen_and_respond()

    voice_assistant.speaker.speak.assert_called_once_with("音声が検出されませんでした。")


def test_listen_and_respond_empty_transcription(voice_assistant):
    """Test listen_and_respond when transcription is empty"""
    # Audio with sufficient level but empty transcription
    audio = np.full(16000, 0.1, dtype=np.float32)
    voice_assistant.listener.record_audio.return_value = audio
    voice_assistant.listener.transcribe.return_value = "   "

    voice_assistant.listen_and_respond()

    # Should return early without calling retriever
    voice_assistant.retriever.search_web.assert_not_called()
    voice_assistant.speaker.speak.assert_not_called()


def test_listen_and_respond_successful_flow(voice_assistant):
    """Test the full successful conversation flow"""
    # Setup mocks for each step
    audio = np.full(16000, 0.5, dtype=np.float32)
    voice_assistant.listener.record_audio.return_value = audio
    voice_assistant.listener.transcribe.return_value = "今日の天気はどうですか"
    voice_assistant.retriever.search_web.return_value = [
        {'title': '天気予報', 'content': '晴れ', 'url': 'http://example.com'}
    ]
    voice_assistant.composer.compose_prompt.return_value = "プロンプト"
    voice_assistant.brain.generate_response.return_value = "今日は晴れです"

    voice_assistant.listen_and_respond()

    # Verify full pipeline was called
    voice_assistant.listener.record_audio.assert_called_once()
    voice_assistant.listener.transcribe.assert_called_once_with(audio)
    voice_assistant.retriever.search_web.assert_called_once_with("今日の天気はどうですか")
    voice_assistant.composer.compose_prompt.assert_called_once()
    voice_assistant.brain.generate_response.assert_called_once_with("プロンプト")
    voice_assistant.speaker.speak.assert_called_once_with("今日は晴れです")


def test_listen_and_respond_exception(voice_assistant):
    """Test listen_and_respond handles generic exceptions gracefully"""
    voice_assistant.listener.record_audio.side_effect = RuntimeError("Mic error")

    # Should not raise
    voice_assistant.listen_and_respond()


def test_listen_and_respond_keyboard_interrupt(voice_assistant):
    """Test listen_and_respond handles KeyboardInterrupt"""
    voice_assistant.listener.record_audio.side_effect = KeyboardInterrupt()

    # Should not raise
    voice_assistant.listen_and_respond()


def test_main_function():
    """Test the main() entry point"""
    with patch('src.main.Listener'), \
         patch('src.main.Retriever'), \
         patch('src.main.Composer'), \
         patch('src.main.Brain'), \
         patch('src.main.Speaker'):
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
         patch('src.main.Speaker'):
        with patch.object(VoiceAssistant, 'listen_and_respond', side_effect=RuntimeError("fail")), \
             patch.object(VoiceAssistant, 'cleanup') as mock_cleanup:
            with pytest.raises(RuntimeError):
                main()
            mock_cleanup.assert_called_once()
