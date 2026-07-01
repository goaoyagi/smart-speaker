#!/usr/bin/env python3
"""
Tests for shared audio_utils module
"""

import pytest
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


def test_log_init(capsys):
    """Test log_init prints correct message"""
    log_init("TestModule")
    captured = capsys.readouterr()
    assert "Initializing TestModule..." in captured.out


def test_log_ready(capsys):
    """Test log_ready prints correct message"""
    log_ready("TestModule")
    captured = capsys.readouterr()
    assert "TestModule initialized!" in captured.out
