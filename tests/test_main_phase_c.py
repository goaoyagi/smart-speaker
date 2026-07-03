#!/usr/bin/env python3
"""
Phase C tests: continuous standby loop for VoiceAssistant
"""

import sys
import pytest
import numpy as np
from unittest.mock import patch, MagicMock, call

sys.modules.setdefault('faster_whisper', MagicMock())
sys.modules.setdefault('piper', MagicMock())

from src.main import VoiceAssistant, EXIT_COMMANDS
from src.exceptions import ListenerError, GenerationError, SpeakerError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_audio(level: float = 0.5, length: int = 16000) -> np.ndarray:
    return np.full(length, level, dtype=np.float32)


def _silent_audio(length: int = 16000) -> np.ndarray:
    return np.zeros(length, dtype=np.float32)


@pytest.fixture
def va():
    """VoiceAssistant with all external components mocked."""
    with patch('src.main.Listener'), \
         patch('src.main.Retriever'), \
         patch('src.main.Composer'), \
         patch('src.main.Brain'), \
         patch('src.main.Speaker'):
        assistant = VoiceAssistant()
        assistant.composer.compose_prompt.return_value = "prompt"
        assistant.brain.generate_response.return_value = "回答"
        assistant.retriever.search_web.return_value = []
        return assistant


# ---------------------------------------------------------------------------
# EXIT_COMMANDS constant
# ---------------------------------------------------------------------------

def test_exit_commands_not_empty():
    assert len(EXIT_COMMANDS) > 0


@pytest.mark.parametrize("cmd", list(EXIT_COMMANDS))
def test_is_exit_command_known(va, cmd):
    assert va._is_exit_command(cmd) is True


def test_is_exit_command_ordinary_text(va):
    assert va._is_exit_command("今日の天気は？") is False


def test_is_exit_command_empty(va):
    assert va._is_exit_command("") is False


# ---------------------------------------------------------------------------
# run_loop – exits on first exit command
# ---------------------------------------------------------------------------

def test_run_loop_exits_on_exit_command(va):
    """Loop should break after one exit command, speaking a farewell."""
    audio = _make_audio()
    va.listener.record_audio.return_value = audio
    va.listener.transcribe.return_value = "終了"

    va.run_loop()

    va.brain.generate_response.assert_not_called()
    # Farewell message must be spoken
    speak_calls = [c[0][0] for c in va.speaker.speak.call_args_list]
    assert any("終了" in s or "またいつでも" in s for s in speak_calls)


def test_run_loop_processes_normal_turn_then_exits(va):
    """One normal question followed by exit command."""
    audio = _make_audio()
    va.listener.record_audio.return_value = audio
    va.listener.transcribe.side_effect = ["今日の天気は？", "終了"]

    va.run_loop()

    va.brain.generate_response.assert_called_once()
    turns = va.history.get_history()
    assert len(turns) == 1
    assert turns[0]["question"] == "今日の天気は？"


def test_run_loop_skips_silence(va):
    """Silent audio (level < 0.03) should not call transcribe; loop continues."""
    silent = _silent_audio()
    normal = _make_audio()
    va.listener.record_audio.side_effect = [silent, silent, normal]
    va.listener.transcribe.return_value = "終了"

    va.run_loop()

    assert va.listener.transcribe.call_count == 1


def test_run_loop_skips_empty_transcription(va):
    """Empty transcription should not call the LLM; loop continues."""
    audio = _make_audio()
    va.listener.record_audio.return_value = audio
    va.listener.transcribe.side_effect = ["   ", "終了"]

    va.run_loop()

    va.brain.generate_response.assert_not_called()


def test_run_loop_continues_after_generation_error(va):
    """GenerationError on one turn should not break the loop."""
    audio = _make_audio()
    va.listener.record_audio.return_value = audio
    va.listener.transcribe.side_effect = ["壊れた質問", "次の質問", "終了"]
    va.brain.generate_response.side_effect = [
        GenerationError("timeout"),
        "正常な回答",
        "正常な回答",
    ]

    va.run_loop()

    assert va.listener.transcribe.call_count == 3


def test_run_loop_repeat_command_replays_last_answer(va):
    """'もう一回言って' inside the loop should replay the last stored answer."""
    audio = _make_audio()
    va.listener.record_audio.return_value = audio
    va.listener.transcribe.side_effect = ["質問", "もう一回言って", "終了"]
    va.brain.generate_response.return_value = "初回の回答"

    va.run_loop()

    speak_calls = [c[0][0] for c in va.speaker.speak.call_args_list]
    assert "初回の回答" in speak_calls


def test_run_loop_fatal_listener_error_propagates(va):
    """A ListenerError from record_audio should propagate out of the loop."""
    va.listener.record_audio.side_effect = ListenerError("mic broken")
    va.listener.transcribe.return_value = "dummy"

    with pytest.raises(ListenerError):
        va.run_loop()


# ---------------------------------------------------------------------------
# _process_turn
# ---------------------------------------------------------------------------

def test_process_turn_stores_history(va):
    va._process_turn("質問テキスト", _make_audio())
    turns = va.history.get_history()
    assert len(turns) == 1
    assert turns[0]["question"] == "質問テキスト"


def test_process_turn_repeat_replays_without_llm(va):
    va.history.add("前の質問", "前の回答")
    va._process_turn("もう一回言って", _make_audio())
    va.brain.generate_response.assert_not_called()
    va.speaker.speak.assert_called_once_with("前の回答")
