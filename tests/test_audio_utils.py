#!/usr/bin/env python3
"""
Tests for shared audio_utils module
"""

import pytest
import logging
import os
import tempfile
from src.audio_utils import TempWavFile, log_init, log_ready


def test_temp_wav_file_creates_and_deletes(tmp_path):
    """Test that TempWavFile creates a .wav file and deletes it on exit"""
    with TempWavFile() as path:
        assert path.endswith('.wav')
        assert os.path.exists(path)
        saved_path = path
    
    assert not os.path.exists(saved_path)


def test_temp_wav_file_deletes_on_exception(tmp_path):
    """Test that TempWavFile cleans up even if an exception occurs"""
    saved_path = None
    try:
        with TempWavFile() as path:
            saved_path = path
            assert os.path.exists(path)
            raise ValueError("simulated error")
    except ValueError:
        pass
    
    assert not os.path.exists(saved_path)


def test_log_init(caplog):
    """Test log_init logs correct message"""
    with caplog.at_level(logging.INFO, logger="src.audio_utils"):
        log_init("TestModule")
    assert "Initializing TestModule..." in caplog.text


def test_log_ready(caplog):
    """Test log_ready logs correct message"""
    with caplog.at_level(logging.INFO, logger="src.audio_utils"):
        log_ready("TestModule")
    assert "TestModule initialized!" in caplog.text
