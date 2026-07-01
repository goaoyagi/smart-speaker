#!/usr/bin/env python3
"""
Listener module - Speech recognition using Whisper
"""

import logging
import numpy as np
from faster_whisper import WhisperModel
import subprocess
import wave
import shutil
import os

from .config import MIC_DEVICE, SAMPLE_RATE, CHANNELS, RECORD_SECONDS, WHISPER_MODEL_SIZE, DEBUG_AUDIO_DIR
from .audio_utils import TempWavFile, log_init, log_ready
from .exceptions import ListenerError

logger = logging.getLogger(__name__)


class Listener:
    def __init__(self):
        log_init("Listener (Whisper)")
        self.whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device="cpu",
            compute_type="int8"
        )
        log_ready("Listener")

    def record_audio(self, duration=RECORD_SECONDS):
        """Record audio from USB microphone using arecord"""
        logger.info("Recording for %d seconds...", duration)

        with TempWavFile() as temp_file:
            cmd = [
                'arecord',
                '-f', 'S16_LE',
                '-r', str(SAMPLE_RATE),
                '-c', str(CHANNELS),
                '-D', MIC_DEVICE,
                '-d', str(duration),
                temp_file
            ]

            try:
                subprocess.run(cmd, check=True, capture_output=True)
            except FileNotFoundError as e:
                raise ListenerError("arecord command not found") from e
            except subprocess.CalledProcessError as e:
                raise ListenerError(
                    f"Audio recording failed (exit {e.returncode}): "
                    f"{e.stderr.decode(errors='replace').strip()}"
                ) from e

            try:
                with wave.open(temp_file, 'rb') as wav_file:
                    frames = wav_file.readframes(wav_file.getnframes())
                    audio_array = np.frombuffer(frames, dtype=np.int16)
                    audio_array = audio_array.astype(np.float32) / 32768.0
            except wave.Error as e:
                raise ListenerError(f"Failed to read recorded WAV file: {e}") from e

            max_level = np.max(np.abs(audio_array))
            logger.debug("Audio max level: %.4f", max_level)
            if max_level < 0.01:
                logger.warning("Audio level is very low. Microphone may not be working.")

            if DEBUG_AUDIO_DIR:
                debug_path = os.path.join(DEBUG_AUDIO_DIR, "last_record.wav")
                try:
                    shutil.copy(temp_file, debug_path)
                except OSError as e:
                    logger.warning("Could not save debug audio copy: %s", e)

        return audio_array

    def transcribe(self, audio_array):
        """Transcribe audio to text using Whisper"""
        logger.info("Transcribing...")

        try:
            segments, info = self.whisper_model.transcribe(
                audio_array,
                language="ja",
                beam_size=5
            )
            text = "".join(segment.text for segment in segments)
        except Exception as e:
            raise ListenerError(f"Whisper transcription failed: {e}") from e

        logger.info("Recognized: %s", text)
        return text
