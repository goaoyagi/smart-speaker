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
from src.history_summarizer import HistorySummarizer
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


# ---------------------------------------------------------------------------
# Phase B: LLM-based history summarization
# ---------------------------------------------------------------------------

@pytest.fixture
def voice_assistant_with_summary():
    """Create VoiceAssistant with use_llm_summary=True and mocked components."""
    with patch('src.main.Listener'), \
         patch('src.main.Retriever'), \
         patch('src.main.Composer'), \
         patch('src.main.Brain'), \
         patch('src.main.Speaker'):
        assistant = VoiceAssistant(use_llm_summary=True)
        return assistant


def test_use_llm_summary_creates_summarizer(voice_assistant_with_summary):
    """When use_llm_summary=True, a HistorySummarizer should be attached."""
    assert isinstance(voice_assistant_with_summary._summarizer, HistorySummarizer)


def test_no_llm_summary_by_default(voice_assistant):
    """Default VoiceAssistant should have no summarizer."""
    assert voice_assistant._summarizer is None


def test_llm_summary_calls_summarizer_on_respond(voice_assistant_with_summary):
    """With LLM summary enabled, summarize() should be called before compose_prompt."""
    audio = np.full(16000, 0.5, dtype=np.float32)
    va = voice_assistant_with_summary
    va.listener.record_audio.return_value = audio
    va.listener.transcribe.return_value = "質問"
    va.retriever.search_web.return_value = []
    va.composer.compose_prompt.return_value = "prompt"
    va.brain.generate_response.return_value = "回答"

    with patch.object(va._summarizer, 'summarize', return_value="要約") as mock_sum:
        va.listen_and_respond()
        mock_sum.assert_called_once_with(va.history)


def test_llm_summary_passes_history_text_to_composer(voice_assistant_with_summary):
    """With LLM summary enabled, history_text= kwarg should be passed to compose_prompt."""
    audio = np.full(16000, 0.5, dtype=np.float32)
    va = voice_assistant_with_summary
    va.listener.record_audio.return_value = audio
    va.listener.transcribe.return_value = "質問"
    va.retriever.search_web.return_value = []
    va.composer.compose_prompt.return_value = "prompt"
    va.brain.generate_response.return_value = "回答"

    with patch.object(va._summarizer, 'summarize', return_value="要約テキスト"):
        va.listen_and_respond()

    call_kwargs = va.composer.compose_prompt.call_args[1]
    assert call_kwargs.get("history_text") == "要約テキスト"


def test_no_llm_summary_passes_history_object_to_composer(voice_assistant):
    """Without LLM summary, the ConversationHistory object should be passed positionally."""
    audio = np.full(16000, 0.5, dtype=np.float32)
    voice_assistant.listener.record_audio.return_value = audio
    voice_assistant.listener.transcribe.return_value = "質問"
    voice_assistant.retriever.search_web.return_value = []
    voice_assistant.composer.compose_prompt.return_value = "prompt"
    voice_assistant.brain.generate_response.return_value = "回答"

    voice_assistant.listen_and_respond()

    call_args = voice_assistant.composer.compose_prompt.call_args
    assert call_args[0][2] is voice_assistant.history
