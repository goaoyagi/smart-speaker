#!/usr/bin/env python3
"""
Tests for speaker module
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys

# Mock external dependencies before import
sys.modules['piper'] = MagicMock()

from src.speaker import Speaker


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
    with patch('src.speaker.tempfile.NamedTemporaryFile'), \
         patch('src.speaker.wave.open'), \
         patch('src.speaker.subprocess.run'), \
         patch('src.speaker.os.unlink'):
        
        speaker.speak("テスト音声")
        
        # Should not raise any exceptions
