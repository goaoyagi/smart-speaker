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
from src.conversation_history import ConversationHistory
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


def test_voice_assistant_has_history(voice_assistant):
    """VoiceAssistant should expose a ConversationHistory instance"""
    assert isinstance(voice_assistant.history, ConversationHistory)
    assert voice_assistant.history.is_empty()


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


def test_listen_and_respond_passes_history_to_composer(voice_assistant):
    """compose_prompt should receive the VoiceAssistant's history object."""
    audio = np.full(16000, 0.5, dtype=np.float32)
    voice_assistant.listener.record_audio.return_value = audio
    voice_assistant.listener.transcribe.return_value = "質問"
    voice_assistant.retriever.search_web.return_value = []
    voice_assistant.composer.compose_prompt.return_value = "prompt"
    voice_assistant.brain.generate_response.return_value = "回答"

    voice_assistant.listen_and_respond()

    call_args = voice_assistant.composer.compose_prompt.call_args
    assert call_args[0][2] is voice_assistant.history


def test_listen_and_respond_stores_turn_in_history(voice_assistant):
    """After a successful turn, history should contain the Q&A pair."""
    audio = np.full(16000, 0.5, dtype=np.float32)
    voice_assistant.listener.record_audio.return_value = audio
    voice_assistant.listener.transcribe.return_value = "今日は何日ですか"
    voice_assistant.retriever.search_web.return_value = []
    voice_assistant.composer.compose_prompt.return_value = "prompt"
    voice_assistant.brain.generate_response.return_value = "三日です"

    voice_assistant.listen_and_respond()

    assert len(voice_assistant.history) == 1
    formatted = voice_assistant.history.format_for_prompt()
    assert "今日は何日ですか" in formatted
    assert "三日です" in formatted


def test_history_accumulates_across_turns(voice_assistant):
    """History should grow with each successful listen_and_respond call."""
    audio = np.full(16000, 0.5, dtype=np.float32)
    voice_assistant.listener.record_audio.return_value = audio
    voice_assistant.retriever.search_web.return_value = []
    voice_assistant.composer.compose_prompt.return_value = "prompt"

    for i in range(3):
        voice_assistant.listener.transcribe.return_value = f"質問{i}"
        voice_assistant.brain.generate_response.return_value = f"回答{i}"
        voice_assistant.listen_and_respond()

    assert len(voice_assistant.history) == 3


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


def test_listen_and_respond_generation_failure_does_not_store_history(voice_assistant):
    """Failed generation should not add a turn to history."""
    audio = np.full(16000, 0.5, dtype=np.float32)
    voice_assistant.listener.record_audio.return_value = audio
    voice_assistant.listener.transcribe.return_value = "質問"
    voice_assistant.retriever.search_web.return_value = []
    voice_assistant.composer.compose_prompt.return_value = "prompt"
    voice_assistant.brain.generate_response.side_effect = GenerationError("timeout")

    with pytest.raises(GenerationError):
        voice_assistant.listen_and_respond()

    assert voice_assistant.history.is_empty()


def test_main_function():
    """Test the main() entry point"""
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


# ---------------------------------------------------------------------------
# Phase C: continuous loop and exit word detection
# ---------------------------------------------------------------------------

def test_listen_and_respond_returns_true_on_success(voice_assistant):
    """Successful pipeline should return True (continue loop)."""
    audio = np.full(16000, 0.5, dtype=np.float32)
    voice_assistant.listener.record_audio.return_value = audio
    voice_assistant.listener.transcribe.return_value = "今日の天気は？"
    voice_assistant.retriever.search_web.return_value = []
    voice_assistant.composer.compose_prompt.return_value = "prompt"
    voice_assistant.brain.generate_response.return_value = "晴れです"

    result = voice_assistant.listen_and_respond()
    assert result is True


def test_listen_and_respond_returns_true_on_low_audio(voice_assistant):
    """Silent audio should return True (keep looping, just skip)."""
    silent_audio = np.zeros(16000, dtype=np.float32)
    voice_assistant.listener.record_audio.return_value = silent_audio

    result = voice_assistant.listen_and_respond()
    assert result is True


def test_listen_and_respond_returns_false_on_exit_word(voice_assistant):
    """Exit word in transcription should return False (stop loop)."""
    audio = np.full(16000, 0.5, dtype=np.float32)
    voice_assistant.listener.record_audio.return_value = audio
    voice_assistant.listener.transcribe.return_value = "終わり"

    result = voice_assistant.listen_and_respond()
    assert result is False


def test_listen_and_respond_exit_word_speaks_farewell(voice_assistant):
    """Exit word should trigger a farewell message."""
    audio = np.full(16000, 0.5, dtype=np.float32)
    voice_assistant.listener.record_audio.return_value = audio
    voice_assistant.listener.transcribe.return_value = "終了"

    voice_assistant.listen_and_respond()

    voice_assistant.speaker.speak.assert_called_once()
    spoken = voice_assistant.speaker.speak.call_args[0][0]
    assert "終了" in spoken or "終わり" in spoken or "またいつ" in spoken


def test_listen_and_respond_exit_word_skips_pipeline(voice_assistant):
    """Exit word should not trigger search or generation."""
    audio = np.full(16000, 0.5, dtype=np.float32)
    voice_assistant.listener.record_audio.return_value = audio
    voice_assistant.listener.transcribe.return_value = "ストップ"

    voice_assistant.listen_and_respond()

    voice_assistant.retriever.search_web.assert_not_called()
    voice_assistant.brain.generate_response.assert_not_called()


def test_listen_and_respond_exit_word_not_stored_in_history(voice_assistant):
    """Exit command should not be recorded in conversation history."""
    audio = np.full(16000, 0.5, dtype=np.float32)
    voice_assistant.listener.record_audio.return_value = audio
    voice_assistant.listener.transcribe.return_value = "終わり"

    voice_assistant.listen_and_respond()

    assert voice_assistant.history.is_empty()


def test_run_loop_stops_on_exit_word(voice_assistant):
    """run_loop() should stop when listen_and_respond returns False."""
    call_count = 0

    def mock_listen():
        nonlocal call_count
        call_count += 1
        if call_count >= 3:
            return False
        return True

    voice_assistant.listen_and_respond = mock_listen
    voice_assistant.run_loop()
    assert call_count == 3


def test_run_loop_continues_on_true(voice_assistant):
    """run_loop() should keep calling listen_and_respond while it returns True."""
    responses = [True, True, False]
    iterator = iter(responses)
    voice_assistant.listen_and_respond = lambda: next(iterator)
    voice_assistant.run_loop()
    # The iterator is exhausted only when all responses have been consumed,
    # confirming that listen_and_respond was called exactly 3 times.
    with pytest.raises(StopIteration):
        next(iterator)


def test_run_loop_stops_on_keyboard_interrupt(voice_assistant):
    """run_loop() should stop cleanly on KeyboardInterrupt."""
    voice_assistant.listen_and_respond = MagicMock(side_effect=KeyboardInterrupt)
    voice_assistant.run_loop()  # Should not raise


def test_run_loop_continues_on_search_error(voice_assistant):
    """Non-fatal SearchError in listen_and_respond should not stop the loop."""
    responses = [SearchError("offline"), True, False]
    iterator = iter(responses)

    def mock_listen():
        r = next(iterator)
        if isinstance(r, Exception):
            raise r
        return r

    voice_assistant.listen_and_respond = mock_listen
    voice_assistant.run_loop()  # Should complete without raising
