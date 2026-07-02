#!/usr/bin/env python3
"""
Tests for main orchestrator
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys

# Mock external dependencies before import
sys.modules['faster_whisper'] = MagicMock()
sys.modules['piper'] = MagicMock()

from src.main import VoiceAssistant, main
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
    """Test that VoiceAssistant initializes all components including history"""
    assert voice_assistant.listener is not None
    assert voice_assistant.retriever is not None
    assert voice_assistant.composer is not None
    assert voice_assistant.brain is not None
    assert voice_assistant.speaker is not None
    assert voice_assistant.history is not None


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


def test_listen_and_respond_saves_turn_to_history(voice_assistant):
    """Test that a completed turn is saved to conversation history"""
    audio = np.full(16000, 0.5, dtype=np.float32)
    voice_assistant.listener.record_audio.return_value = audio
    voice_assistant.listener.transcribe.return_value = "質問です"
    voice_assistant.retriever.search_web.return_value = []
    voice_assistant.composer.compose_prompt.return_value = "プロンプト"
    voice_assistant.brain.generate_response.return_value = "回答です"

    voice_assistant.listen_and_respond()

    turns = voice_assistant.history.get_history()
    assert len(turns) == 1
    assert turns[0]["user"] == "質問です"
    assert turns[0]["assistant"] == "回答です"


def test_listen_and_respond_passes_history_to_composer(voice_assistant):
    """Test that history summary is passed to compose_prompt on subsequent turns"""
    audio = np.full(16000, 0.5, dtype=np.float32)
    voice_assistant.listener.record_audio.return_value = audio
    voice_assistant.listener.transcribe.return_value = "質問"
    voice_assistant.retriever.search_web.return_value = []
    voice_assistant.composer.compose_prompt.return_value = "プロンプト"
    voice_assistant.brain.generate_response.return_value = "回答"

    # Seed history with a previous turn
    voice_assistant.history.add_turn("前の質問", "前の回答")

    voice_assistant.listen_and_respond()

    call_args = voice_assistant.composer.compose_prompt.call_args
    history_summary_arg = call_args.args[2]
    assert "前の質問" in history_summary_arg
    assert "前の回答" in history_summary_arg


def test_listen_and_respond_repeat_request_replays_last_response(voice_assistant):
    """Test that a repeat request speaks the last stored response without a full pipeline"""
    voice_assistant.history.add_turn("前の質問", "前の回答です")

    audio = np.full(16000, 0.5, dtype=np.float32)
    voice_assistant.listener.record_audio.return_value = audio
    voice_assistant.listener.transcribe.return_value = "もう一回言って"

    voice_assistant.listen_and_respond()

    voice_assistant.speaker.speak.assert_called_once_with("前の回答です")
    voice_assistant.retriever.search_web.assert_not_called()
    voice_assistant.brain.generate_response.assert_not_called()


def test_listen_and_respond_repeat_request_empty_history_continues(voice_assistant):
    """Test that a repeat request with no history falls through to the normal pipeline"""
    audio = np.full(16000, 0.5, dtype=np.float32)
    voice_assistant.listener.record_audio.return_value = audio
    voice_assistant.listener.transcribe.return_value = "もう一回言って"
    voice_assistant.retriever.search_web.return_value = []
    voice_assistant.composer.compose_prompt.return_value = "プロンプト"
    voice_assistant.brain.generate_response.return_value = "回答"

    voice_assistant.listen_and_respond()

    voice_assistant.brain.generate_response.assert_called_once()


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

    # First positional arg is the query, second is search_results (empty list)
    call_args = voice_assistant.composer.compose_prompt.call_args[0]
    assert call_args[0] == "テスト"
    assert call_args[1] == []
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
        with patch.object(VoiceAssistant, 'listen_and_respond',
                          side_effect=ListenerError("fail")), \
             patch.object(VoiceAssistant, 'cleanup') as mock_cleanup:
            with pytest.raises(SystemExit):
                main()
            mock_cleanup.assert_called_once()

