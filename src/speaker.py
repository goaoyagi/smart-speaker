#!/usr/bin/env python3
"""
Speaker module - Text-to-speech using Piper-Plus
"""

import logging
import subprocess
import tempfile
import wave
import os

from piper import PiperVoice
from exceptions import SpeakerError

logger = logging.getLogger(__name__)

# Configuration
SPEAKER_DEVICE = os.getenv("SPEAKER_DEVICE", "plughw:0,0")
PIPER_MODEL_PATH = os.getenv("PIPER_MODEL_PATH", "./models/tsukuyomi.onnx")
PIPER_CONFIG_PATH = os.getenv("PIPER_CONFIG_PATH", "./models/config.json")


class Speaker:
    def __init__(self):
        print("Initializing Speaker (Piper-Plus)...")
        self.piper_model = PiperVoice.load(
            PIPER_MODEL_PATH,
            config_path=PIPER_CONFIG_PATH
        )
        self.speaker_device = SPEAKER_DEVICE
        print("Speaker initialized!")

    def speak(self, text):
        """Convert text to speech using Piper-Plus"""
        logger.info("Speaking: %s", text)

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_file = f.name

        try:
            with wave.open(temp_file, 'wb') as wav_file:
                self.piper_model.synthesize(
                    text,
                    wav_file,
                    speaker_id=0,
                    length_scale=1.0
                )
        except Exception as e:
            os.unlink(temp_file)
            raise SpeakerError(f"TTS synthesis failed: {e}") from e

        try:
            subprocess.run(
                ['aplay', '-D', self.speaker_device, temp_file],
                capture_output=True,
                check=True,
            )
        except FileNotFoundError as e:
            raise SpeakerError("aplay command not found") from e
        except subprocess.CalledProcessError as e:
            raise SpeakerError(
                f"Audio playback failed (exit {e.returncode}): "
                f"{e.stderr.decode(errors='replace').strip()}"
            ) from e
        finally:
            os.unlink(temp_file)
