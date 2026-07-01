#!/usr/bin/env python3
"""
Tests for speaker module
"""

import subprocess
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys

# Mock external dependencies before import
sys.modules['piper'] = MagicMock()

from src.speaker import Speaker
from src.exceptions import SpeakerError


@pytest.fixture
def speaker():
    """Create Speaker instance with mocked PiperVoice"""
    return Speaker()


def test_speaker_initialization(speaker):
    """Test that Speaker initializes correctly"""
    assert speaker.piper_model is not None
    assert speaker.speaker_device == "plughw:0,0"


def test_speak(speaker):
    """Test speak method"""
    with patch('src.audio_utils.tempfile.mkstemp', return_value=(3, '/tmp/test.wav')), \
         patch('src.audio_utils.os.close'), \
         patch('src.speaker.wave.open'), \
         patch('src.speaker.subprocess.run'), \
         patch('src.audio_utils.os.path.exists', return_value=True), \
         patch('src.audio_utils.os.unlink'):

        speaker.speak("テスト音声")


def test_speak_playback_failure(speaker):
    """Test speak raises SpeakerError when aplay fails"""
    with patch('src.audio_utils.tempfile.mkstemp', return_value=(3, '/tmp/test.wav')), \
         patch('src.audio_utils.os.close'), \
         patch('src.speaker.wave.open'), \
         patch('src.speaker.subprocess.run',
               side_effect=subprocess.CalledProcessError(1, 'aplay', stderr=b'device not found')), \
         patch('src.audio_utils.os.path.exists', return_value=True), \
         patch('src.audio_utils.os.unlink'):

        with pytest.raises(SpeakerError, match="Audio playback failed"):
            speaker.speak("テスト音声")
