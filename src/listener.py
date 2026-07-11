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

# Seconds to wait for arecord to flush the WAV header after terminate().
_STOP_TIMEOUT = 2.0


class Listener:
    def __init__(self):
        log_init("Listener (Whisper)")
        self.whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device="cpu",
            compute_type="int8"
        )
        self._record_proc = None
        self._record_file = None
        log_ready("Listener")

    def _arecord_cmd(self, output_path, duration=None):
        """Build the arecord command, optionally with a fixed duration."""
        cmd = [
            'arecord',
            '-f', 'S16_LE',
            '-r', str(SAMPLE_RATE),
            '-c', str(CHANNELS),
            '-D', MIC_DEVICE,
        ]
        if duration is not None:
            cmd += ['-d', str(duration)]
        cmd.append(output_path)
        return cmd

    def _decode_wav(self, wav_path):
        """Read a WAV file into a normalised float32 numpy array."""
        try:
            with wave.open(wav_path, 'rb') as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
                audio_array = np.frombuffer(frames, dtype=np.int16)
                audio_array = audio_array.astype(np.float32) / 32768.0
        except wave.Error as e:
            raise ListenerError(f"Failed to read recorded WAV file: {e}") from e

        max_level = np.max(np.abs(audio_array)) if audio_array.size else 0.0
        logger.debug("Audio max level: %.4f", max_level)
        if max_level < 0.01:
            logger.warning("Audio level is very low. Microphone may not be working.")

        if DEBUG_AUDIO_DIR:
            debug_path = os.path.join(DEBUG_AUDIO_DIR, "last_record.wav")
            try:
                shutil.copy(wav_path, debug_path)
            except OSError as e:
                logger.warning("Could not save debug audio copy: %s", e)

        return audio_array

    def record_audio(self, duration=RECORD_SECONDS):
        """Record audio from USB microphone using arecord (fixed duration)."""
        logger.info("Recording for %d seconds...", duration)

        with TempWavFile() as temp_file:
            cmd = self._arecord_cmd(temp_file, duration=duration)

            try:
                subprocess.run(cmd, check=True, capture_output=True)
            except FileNotFoundError as e:
                raise ListenerError("arecord command not found") from e
            except subprocess.CalledProcessError as e:
                raise ListenerError(
                    f"Audio recording failed (exit {e.returncode}): "
                    f"{e.stderr.decode(errors='replace').strip()}"
                ) from e

            return self._decode_wav(temp_file)

    def start_recording(self):
        """Begin variable-length recording for push-to-talk.

        Launches arecord as a background process that records until
        stop_recording() is called.
        """
        if self._record_proc is not None:
            logger.warning("Recording already in progress; restarting.")
            self._terminate_recording()

        self._record_file = TempWavFile()
        temp_path = self._record_file.__enter__()
        cmd = self._arecord_cmd(temp_path)

        logger.info("Recording started (push-to-talk)...")
        try:
            self._record_proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        except FileNotFoundError as e:
            self._cleanup_record_file()
            raise ListenerError("arecord command not found") from e

    def stop_recording(self):
        """Stop the active recording and return the captured audio."""
        if self._record_proc is None:
            raise ListenerError("stop_recording called with no active recording")

        logger.info("Recording stopped (push-to-talk).")
        proc = self._record_proc
        temp_path = self._record_file.path

        proc.terminate()
        try:
            _, stderr = proc.communicate(timeout=_STOP_TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()
            _, stderr = proc.communicate()

        self._record_proc = None

        # arecord may exit non-zero when terminated by signal; that is expected.
        if proc.returncode is not None and proc.returncode > 0:
            self._cleanup_record_file()
            message = stderr.decode(errors='replace').strip() if stderr else ""
            raise ListenerError(
                f"Audio recording failed (exit {proc.returncode}): {message}"
            )

        try:
            return self._decode_wav(temp_path)
        finally:
            self._cleanup_record_file()

    def _terminate_recording(self):
        """Best-effort teardown of a running recording process."""
        if self._record_proc is not None:
            try:
                self._record_proc.terminate()
                self._record_proc.communicate(timeout=_STOP_TIMEOUT)
            except (subprocess.TimeoutExpired, OSError):
                self._record_proc.kill()
                self._record_proc.communicate()
            self._record_proc = None
        self._cleanup_record_file()

    def _cleanup_record_file(self):
        if self._record_file is not None:
            self._record_file.__exit__(None, None, None)
            self._record_file = None

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
