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
