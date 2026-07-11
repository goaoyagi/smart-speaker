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
         patch('src.main.StatusLED'), \
         patch('src.main.PushToTalkButton') as mock_button_cls:
        mock_button_cls.return_value.available = False
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
    assert voice_assistant.button is not None


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
         patch('src.main.StatusLED'), \
         patch('src.main.PushToTalkButton') as mock_button_cls:
        mock_button_cls.return_value.available = False
        with patch.object(VoiceAssistant, 'run') as mock_run, \
             patch.object(VoiceAssistant, 'cleanup') as mock_cleanup:
            main()
            mock_run.assert_called_once()
            mock_cleanup.assert_called_once()


def test_main_function_calls_cleanup_on_exception():
    """Test that main() calls cleanup even when run raises"""
    with patch('src.main.Listener'), \
         patch('src.main.Retriever'), \
         patch('src.main.Composer'), \
         patch('src.main.Brain'), \
         patch('src.main.Speaker'), \
         patch('src.main.StatusLED'), \
         patch('src.main.PushToTalkButton') as mock_button_cls:
        mock_button_cls.return_value.available = False
        with patch.object(VoiceAssistant, 'run',
                          side_effect=ListenerError("fail")), \
             patch.object(VoiceAssistant, 'cleanup') as mock_cleanup:
            with pytest.raises(SystemExit):
                main()
            mock_cleanup.assert_called_once()


def test_run_uses_fixed_recording_when_no_button(voice_assistant):
    """Test run() falls back to fixed recording when button is unavailable."""
    voice_assistant.button.available = False
    with patch.object(voice_assistant, 'listen_and_respond') as mock_listen:
        voice_assistant.run()
        mock_listen.assert_called_once()


def test_run_uses_push_to_talk_when_button_available(voice_assistant):
    """Test run() uses push-to-talk loop when button is available."""
    voice_assistant.button.available = True
    with patch.object(voice_assistant, 'run_push_to_talk') as mock_ptt:
        voice_assistant.run()
        mock_ptt.assert_called_once()


def test_push_to_talk_turn_successful_flow(voice_assistant):
    """Test a full push-to-talk turn from button release to response."""
    audio = np.full(16000, 0.5, dtype=np.float32)
    voice_assistant.button.wait_for_release.return_value = True
    voice_assistant.listener.stop_recording.return_value = audio
    voice_assistant.listener.transcribe.return_value = "今日の天気は"
    voice_assistant.retriever.search_web.return_value = []
    voice_assistant.composer.compose_prompt.return_value = "prompt"
    voice_assistant.brain.generate_response.return_value = "晴れです"

    with patch('src.main.time.monotonic', side_effect=[0.0, 2.0]):
        voice_assistant._push_to_talk_turn()

    voice_assistant.listener.start_recording.assert_called_once()
    voice_assistant.listener.stop_recording.assert_called_once()
    voice_assistant.speaker.speak.assert_called_once_with("晴れです")


def test_push_to_talk_turn_discards_short_press(voice_assistant):
    """Test that a very short button press is ignored."""
    voice_assistant.button.wait_for_release.return_value = True
    voice_assistant.listener.stop_recording.return_value = np.zeros(100)

    with patch('src.main.time.monotonic', side_effect=[0.0, 0.1]):
        voice_assistant._push_to_talk_turn()

    voice_assistant.listener.transcribe.assert_not_called()
    assert voice_assistant.status_led.set_state.call_args_list[-1] == call(LedState.IDLE)


def test_run_push_to_talk_loops_until_interrupt(voice_assistant):
    """Test run_push_to_talk loops and handles KeyboardInterrupt."""
    voice_assistant.button.wait_for_press.side_effect = [True, KeyboardInterrupt]

    with patch.object(voice_assistant, '_push_to_talk_turn') as mock_turn:
        voice_assistant.run_push_to_talk()
        mock_turn.assert_called_once()
