#!/usr/bin/env python3
"""
Listener module - Speech recognition using Whisper
"""

import logging
import numpy as np
from faster_whisper import WhisperModel
import tempfile
import subprocess
import wave
import shutil
import os
from exceptions import ListenerError

logger = logging.getLogger(__name__)

# Configuration
MIC_DEVICE = os.getenv("MIC_DEVICE", "hw:0,0")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
CHANNELS = int(os.getenv("CHANNELS", "1"))
RECORD_SECONDS = int(os.getenv("RECORD_SECONDS", "10"))
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")


class Listener:
    def __init__(self):
        print("Initializing Listener (Whisper)...")
        self.whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device="cpu",
            compute_type="int8"
        )
        print("Listener initialized!")

    def record_audio(self, duration=RECORD_SECONDS):
        """Record audio from USB microphone using arecord"""
        logger.info("Recording for %d seconds...", duration)

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_file = f.name

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
            os.unlink(temp_file)
            raise ListenerError("arecord command not found") from e
        except subprocess.CalledProcessError as e:
            os.unlink(temp_file)
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
            os.unlink(temp_file)
            raise ListenerError(f"Failed to read recorded WAV file: {e}") from e

        max_level = np.max(np.abs(audio_array))
        logger.debug("Audio max level: %.4f", max_level)
        if max_level < 0.01:
            logger.warning("Audio level is very low. Microphone may not be working.")

        # Save a debug copy (non-critical — log and continue on failure)
        debug_dir = os.getenv("DEBUG_AUDIO_DIR", "")
        if debug_dir:
            debug_path = os.path.join(debug_dir, "last_record.wav")
            try:
                shutil.copy(temp_file, debug_path)
            except OSError as e:
                logger.warning("Could not save debug audio copy: %s", e)

        os.unlink(temp_file)
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
