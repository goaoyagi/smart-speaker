#!/usr/bin/env python3
"""
Speaker module - Text-to-speech using Piper-Plus
"""

import logging
import subprocess
import wave

from piper import PiperVoice

from .config import SPEAKER_DEVICE, PIPER_MODEL_PATH, PIPER_CONFIG_PATH
from .audio_utils import TempWavFile, log_init, log_ready
from .exceptions import SpeakerError
from .status_led import LedState

logger = logging.getLogger(__name__)


class Speaker:
    def __init__(self, status_led=None):
        log_init("Speaker (Piper-Plus)")
        self._status_led = status_led
        self.piper_model = PiperVoice.load(
            PIPER_MODEL_PATH,
            config_path=PIPER_CONFIG_PATH
        )
        self.speaker_device = SPEAKER_DEVICE
        log_ready("Speaker")

    def speak(self, text):
        """Convert text to speech using Piper-Plus"""
        if self._status_led is not None:
            self._status_led.set_state(LedState.SPEAKING)
        logger.info("Speaking: %s", text)

        with TempWavFile() as temp_file:
            try:
                with wave.open(temp_file, 'wb') as wav_file:
                    self.piper_model.synthesize(
                        text,
                        wav_file,
                        speaker_id=0,
                        length_scale=1.0
                    )
            except Exception as e:
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
