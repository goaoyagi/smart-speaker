#!/usr/bin/env python3
"""
Tests for listener module
"""

import subprocess
import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import sys

# Mock external dependencies before import
sys.modules['faster_whisper'] = MagicMock()

from src.listener import Listener
from src.exceptions import ListenerError


@pytest.fixture
def listener():
    """Create Listener instance with mocked WhisperModel"""
    return Listener()


def test_listener_initialization(listener):
    """Test that Listener initializes correctly"""
    assert listener.whisper_model is not None


def test_transcribe(listener, mock_audio_array):
    """Test transcribe method"""
    mock_segment = Mock(text="テスト")
    mock_info = Mock()
    listener.whisper_model.transcribe = Mock(return_value=(iter([mock_segment]), mock_info))

    result = listener.transcribe(mock_audio_array)

    assert result == "テスト"
    listener.whisper_model.transcribe.assert_called_once()


def test_record_audio(listener):
    """Test record_audio method"""
    mock_wav_cm = MagicMock()
    mock_wav_file = MagicMock()
    mock_wav_file.getnframes.return_value = 16000
    mock_wav_file.readframes.return_value = b'\x00' * 32000
    mock_wav_cm.__enter__.return_value = mock_wav_file

    with patch('src.listener.subprocess.run'), \
         patch('src.listener.wave.open', return_value=mock_wav_cm), \
         patch('src.audio_utils.os.path.exists', return_value=True), \
         patch('src.audio_utils.os.unlink'), \
         patch('src.audio_utils.tempfile.mkstemp', return_value=(3, '/tmp/test.wav')), \
         patch('src.audio_utils.os.close'):

        result = listener.record_audio(duration=1)

        assert isinstance(result, np.ndarray)


def test_record_audio_arecord_failure(listener):
    """Test record_audio raises ListenerError when arecord fails"""
    with patch('src.listener.subprocess.run',
               side_effect=subprocess.CalledProcessError(1, 'arecord', stderr=b'no device')), \
         patch('src.audio_utils.os.path.exists', return_value=True), \
         patch('src.audio_utils.os.unlink'), \
         patch('src.audio_utils.tempfile.mkstemp', return_value=(3, '/tmp/test.wav')), \
         patch('src.audio_utils.os.close'):

        with pytest.raises(ListenerError, match="Audio recording failed"):
            listener.record_audio(duration=1)


def _mock_wav():
    mock_wav_cm = MagicMock()
    mock_wav_file = MagicMock()
    mock_wav_file.getnframes.return_value = 16000
    mock_wav_file.readframes.return_value = b'\x00' * 32000
    mock_wav_cm.__enter__.return_value = mock_wav_file
    return mock_wav_cm


def test_start_and_stop_recording(listener):
    """Variable-length recording launches arecord and decodes on stop."""
    mock_proc = MagicMock()
    mock_proc.returncode = -15  # terminated by SIGTERM
    mock_proc.communicate.return_value = (b'', b'')

    with patch('src.listener.subprocess.Popen', return_value=mock_proc) as mock_popen, \
         patch('src.listener.wave.open', return_value=_mock_wav()), \
         patch('src.audio_utils.os.path.exists', return_value=True), \
         patch('src.audio_utils.os.unlink'), \
         patch('src.audio_utils.tempfile.mkstemp', return_value=(3, '/tmp/test.wav')), \
         patch('src.audio_utils.os.close'):

        listener.start_recording()
        mock_popen.assert_called_once()
        assert listener._record_proc is mock_proc

        result = listener.stop_recording()

        mock_proc.terminate.assert_called_once()
        assert isinstance(result, np.ndarray)
        assert listener._record_proc is None


def test_stop_recording_without_start_raises(listener):
    """stop_recording without an active recording is an error."""
    with pytest.raises(ListenerError, match="no active recording"):
        listener.stop_recording()


def test_start_recording_arecord_missing(listener):
    """start_recording raises ListenerError when arecord is absent."""
    with patch('src.listener.subprocess.Popen', side_effect=FileNotFoundError()), \
         patch('src.audio_utils.os.path.exists', return_value=True), \
         patch('src.audio_utils.os.unlink'), \
         patch('src.audio_utils.tempfile.mkstemp', return_value=(3, '/tmp/test.wav')), \
         patch('src.audio_utils.os.close'):

        with pytest.raises(ListenerError, match="arecord command not found"):
            listener.start_recording()
        assert listener._record_proc is None


def test_stop_recording_detects_early_failure(listener):
    """A non-signal, non-zero arecord exit surfaces as ListenerError."""
    mock_proc = MagicMock()
    mock_proc.returncode = 1  # failed before we asked it to stop
    mock_proc.communicate.return_value = (b'', b'no device')

    with patch('src.listener.subprocess.Popen', return_value=mock_proc), \
         patch('src.audio_utils.os.path.exists', return_value=True), \
         patch('src.audio_utils.os.unlink'), \
         patch('src.audio_utils.tempfile.mkstemp', return_value=(3, '/tmp/test.wav')), \
         patch('src.audio_utils.os.close'):

        listener.start_recording()
        with pytest.raises(ListenerError, match="Audio recording failed"):
            listener.stop_recording()
        assert listener._record_proc is None
