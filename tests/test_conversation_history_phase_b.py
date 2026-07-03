#!/usr/bin/env python3
"""
Phase B tests: LLM-based summarisation for ConversationHistory
"""

import pytest
from unittest.mock import MagicMock, patch
from src.conversation_history import ConversationHistory
from src.exceptions import GenerationError


@pytest.fixture
def history_with_two_turns():
    h = ConversationHistory(max_turns=3)
    h.add("質問1", "回答1")
    h.add("質問2", "回答2")
    return h


@pytest.fixture
def mock_brain():
    brain = MagicMock()
    brain.generate_response.return_value = "これまでの要約です。"
    return brain


# ---------------------------------------------------------------------------
# summarize_with_llm
# ---------------------------------------------------------------------------

def test_summarize_with_llm_returns_summary(history_with_two_turns, mock_brain):
    summary = history_with_two_turns.summarize_with_llm(mock_brain)
    assert summary == "これまでの要約です。"
    mock_brain.generate_response.assert_called_once()


def test_summarize_with_llm_prompt_contains_history(history_with_two_turns, mock_brain):
    history_with_two_turns.summarize_with_llm(mock_brain)
    prompt_used = mock_brain.generate_response.call_args[0][0]
    assert "質問1" in prompt_used
    assert "回答1" in prompt_used


def test_summarize_with_llm_empty_history_returns_empty():
    h = ConversationHistory()
    brain = MagicMock()
    result = h.summarize_with_llm(brain)
    assert result == ""
    brain.generate_response.assert_not_called()


def test_summarize_with_llm_fallback_on_empty_response(history_with_two_turns, mock_brain):
    """When LLM returns empty string, raw history text is returned as fallback"""
    mock_brain.generate_response.return_value = "   "
    fallback = history_with_two_turns.summarize_with_llm(mock_brain)
    assert "質問1" in fallback


def test_summarize_with_llm_strips_whitespace(history_with_two_turns, mock_brain):
    mock_brain.generate_response.return_value = "  要約テキスト  "
    result = history_with_two_turns.summarize_with_llm(mock_brain)
    assert result == "要約テキスト"


# ---------------------------------------------------------------------------
# replace_with_summary
# ---------------------------------------------------------------------------

def test_replace_with_summary_clears_old_turns(history_with_two_turns):
    history_with_two_turns.replace_with_summary("コンパクトな要約")
    turns = history_with_two_turns.get_history()
    assert len(turns) == 1


def test_replace_with_summary_stores_summary_as_answer(history_with_two_turns):
    history_with_two_turns.replace_with_summary("コンパクトな要約")
    turns = history_with_two_turns.get_history()
    assert turns[0]["answer"] == "コンパクトな要約"


def test_replace_with_summary_last_answer_is_summary(history_with_two_turns):
    history_with_two_turns.replace_with_summary("コンパクトな要約")
    assert history_with_two_turns.last_answer() == "コンパクトな要約"


# ---------------------------------------------------------------------------
# Integration: summarise → replace cycle in VoiceAssistant
# ---------------------------------------------------------------------------

def test_voice_assistant_triggers_summarisation_when_history_full():
    """After _SUMMARIZE_AFTER_TURNS turns, history is compacted via LLM"""
    import sys
    from unittest.mock import patch, MagicMock
    sys.modules.setdefault('faster_whisper', MagicMock())
    sys.modules.setdefault('piper', MagicMock())

    from src.main import VoiceAssistant, _SUMMARIZE_AFTER_TURNS
    import numpy as np

    with patch('src.main.Listener'), \
         patch('src.main.Retriever'), \
         patch('src.main.Composer') as MockComposer, \
         patch('src.main.Brain') as MockBrain, \
         patch('src.main.Speaker'):

        assistant = VoiceAssistant()
        MockComposer.return_value.compose_prompt.return_value = "prompt"
        MockBrain.return_value.generate_response.return_value = "回答"

        audio = np.full(16000, 0.5, dtype=np.float32)
        assistant.listener.record_audio.return_value = audio
        assistant.retriever.search_web.return_value = []

        for i in range(_SUMMARIZE_AFTER_TURNS):
            assistant.listener.transcribe.return_value = f"質問{i}"
            assistant.brain.generate_response.return_value = f"回答{i}"
            assistant.listen_and_respond()

        # After filling history, replace_with_summary should have been called
        # meaning history is now condensed to 1 turn
        assert len(assistant.history.get_history()) == 1
        summary_turn = assistant.history.get_history()[0]
        assert summary_turn["question"] == "[要約]"
