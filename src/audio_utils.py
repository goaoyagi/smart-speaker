#!/usr/bin/env python3
"""
Shared audio utilities - Eliminates duplicated temp-WAV-file lifecycle
in listener.py and speaker.py.
"""

import os
import tempfile


class TempWavFile:
    """Context manager for a temporary .wav file that auto-deletes on exit.

    Handles cleanup even when exceptions occur, replacing the repeated
    pattern of manual tempfile creation + try/finally/os.unlink.
    """

    def __init__(self):
        self._path = None

    @property
    def path(self):
        return self._path

    def __enter__(self):
        fd, self._path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        return self._path

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._path and os.path.exists(self._path):
            os.unlink(self._path)
        return False


def log_init(component_name):
    """Standard init logging shared by all components."""
    print(f"Initializing {component_name}...")


def log_ready(component_name):
    """Standard ready logging shared by all components."""
    print(f"{component_name} initialized!")
