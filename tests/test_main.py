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


@pytest.fixture
def voice_assistant_with_history(voice_assistant):
    """VoiceAssistant pre-loaded with one history turn"""
    voice_assistant.history.add("前の質問", "前の回答")
    return voice_assistant


def test_voice_assistant_initialization(voice_assistant):
    """Test that VoiceAssistant initializes all components"""
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


def test_listen_and_respond_stores_history(voice_assistant):
    """Completed turn should be added to conversation history"""
    audio = np.full(16000, 0.5, dtype=np.float32)
    voice_assistant.listener.record_audio.return_value = audio
    voice_assistant.listener.transcribe.return_value = "質問"
    voice_assistant.retriever.search_web.return_value = []
    voice_assistant.composer.compose_prompt.return_value = "prompt"
    voice_assistant.brain.generate_response.return_value = "回答"

    assert voice_assistant.history.is_empty()
    voice_assistant.listen_and_respond()

    turns = voice_assistant.history.get_history()
    assert len(turns) == 1
    assert turns[0] == {"question": "質問", "answer": "回答"}


def test_listen_and_respond_history_passed_to_composer(voice_assistant_with_history):
    """Existing history text is forwarded to compose_prompt"""
    audio = np.full(16000, 0.5, dtype=np.float32)
    voice_assistant_with_history.listener.record_audio.return_value = audio
    voice_assistant_with_history.listener.transcribe.return_value = "新しい質問"
    voice_assistant_with_history.retriever.search_web.return_value = []
    voice_assistant_with_history.composer.compose_prompt.return_value = "prompt"
    voice_assistant_with_history.brain.generate_response.return_value = "新しい回答"

    voice_assistant_with_history.listen_and_respond()

    call_kwargs = voice_assistant_with_history.composer.compose_prompt.call_args
    history_arg = call_kwargs[1].get("history_text") or (
        call_kwargs[0][2] if len(call_kwargs[0]) > 2 else ""
    )
    assert "前の質問" in history_arg
    assert "前の回答" in history_arg


def test_listen_and_respond_repeat_command_with_history(voice_assistant_with_history):
    """'もう一回言って' should replay the last answer without calling the LLM"""
    audio = np.full(16000, 0.5, dtype=np.float32)
    voice_assistant_with_history.listener.record_audio.return_value = audio
    voice_assistant_with_history.listener.transcribe.return_value = "もう一回言って"

    voice_assistant_with_history.listen_and_respond()

    voice_assistant_with_history.brain.generate_response.assert_not_called()
    voice_assistant_with_history.speaker.speak.assert_called_once_with("前の回答")


def test_listen_and_respond_repeat_command_no_history(voice_assistant):
    """Repeat command with empty history should inform the user"""
    audio = np.full(16000, 0.5, dtype=np.float32)
    voice_assistant.listener.record_audio.return_value = audio
    voice_assistant.listener.transcribe.return_value = "もう一回言って"

    voice_assistant.listen_and_respond()

    voice_assistant.brain.generate_response.assert_not_called()
    voice_assistant.speaker.speak.assert_called_once()
    spoken = voice_assistant.speaker.speak.call_args[0][0]
    assert "履歴" in spoken


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

    call_args = voice_assistant.composer.compose_prompt.call_args
    assert call_args[0][0] == "テスト"
    assert call_args[0][1] == []
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
    """Test the main() entry point calls run_loop and cleanup"""
    with patch('src.main.Listener'), \
         patch('src.main.Retriever'), \
         patch('src.main.Composer'), \
         patch('src.main.Brain'), \
         patch('src.main.Speaker'):
        with patch.object(VoiceAssistant, 'run_loop') as mock_loop, \
             patch.object(VoiceAssistant, 'cleanup') as mock_cleanup:
            main()
            mock_loop.assert_called_once()
            mock_cleanup.assert_called_once()


def test_main_function_calls_cleanup_on_exception():
    """Test that main() calls cleanup even when run_loop raises"""
    with patch('src.main.Listener'), \
         patch('src.main.Retriever'), \
         patch('src.main.Composer'), \
         patch('src.main.Brain'), \
         patch('src.main.Speaker'):
        with patch.object(VoiceAssistant, 'run_loop',
                          side_effect=ListenerError("fail")), \
             patch.object(VoiceAssistant, 'cleanup') as mock_cleanup:
            with pytest.raises(SystemExit):
                main()
            mock_cleanup.assert_called_once()
