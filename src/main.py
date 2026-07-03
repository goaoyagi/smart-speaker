#!/usr/bin/env python3
"""
Main orchestrator - Coordinates all components for voice assistant
"""

import logging
import numpy as np
from .listener import Listener
from .retriever import Retriever
from .composer import Composer
from .brain import Brain
from .speaker import Speaker
from .conversation_history import ConversationHistory
from .exceptions import (
    ListenerError,
    SearchError,
    GenerationError,
    SpeakerError,
)

logger = logging.getLogger(__name__)

EXIT_COMMANDS = {"終了", "停止", "終わり", "止めて", "バイバイ", "おやすみ"}


class VoiceAssistant:
    def __init__(self):
        print("Initializing Voice Assistant...")

        # Initialize all components
        self.listener = Listener()
        self.retriever = Retriever()
        self.composer = Composer()
        self.brain = Brain()
        self.speaker = Speaker()
        self.history = ConversationHistory(max_turns=5)

        print("Voice Assistant initialized!")

    def listen_and_respond(self):
        """Single-turn conversation: record → transcribe → process → speak."""
        logger.info("Voice Assistant Ready — Speak now (will record for 10 seconds)...")

        # --- Record & validate audio ---
        try:
            audio_array = self.listener.record_audio()
        except ListenerError as e:
            logger.error("Recording failed: %s", e)
            raise

        max_level = np.max(np.abs(audio_array))
        if max_level < 0.03:
            logger.info("No speech detected (audio level too low: %.4f).", max_level)
            self._safe_speak("音声が検出されませんでした。")
            return

        # --- Transcribe ---
        try:
            text = self.listener.transcribe(audio_array)
        except ListenerError as e:
            logger.error("Transcription failed: %s", e)
            raise

        if not text.strip():
            logger.info("No speech detected.")
            return

        try:
            self._process_turn(text, audio_array)
        except GenerationError as e:
            logger.error("AI generation failed: %s", e)
            self._safe_speak("申し訳ありませんが、回答を生成できませんでした。")
            raise
        except SpeakerError as e:
            logger.error("Speech output failed: %s", e)
            raise

    def _safe_speak(self, text):
        """Attempt to speak; log and continue if it fails."""
        try:
            self.speaker.speak(text)
        except SpeakerError as e:
            logger.warning("Could not speak error message: %s", e)

    def _is_exit_command(self, text: str) -> bool:
        """Return True when the user spoke a termination keyword."""
        return text.strip() in EXIT_COMMANDS

    def run_loop(self):
        """Continuous standby loop — listens repeatedly until an exit command.

        Each iteration calls ``listen_and_respond``.  The loop terminates when:
        - The user speaks an exit command (「終了」「停止」など).
        - A ``KeyboardInterrupt`` (Ctrl-C) is received.
        - A fatal ``ListenerError`` or ``GenerationError`` propagates.

        ``SearchError`` and ``SpeakerError`` are handled inside
        ``listen_and_respond`` and do not break the loop.
        """
        logger.info("Entering continuous standby loop. Say '終了' to exit.")
        self._safe_speak("スマートスピーカーを起動しました。ご用件をどうぞ。")

        while True:
            try:
                # Record a short sample just to get the transcribed text for
                # exit-command detection before delegating to the full pipeline.
                audio_array = self.listener.record_audio()
            except ListenerError as e:
                logger.error("Recording failed: %s", e)
                raise

            max_level = np.max(np.abs(audio_array))
            if max_level < 0.03:
                logger.info("Silence detected, continuing loop.")
                continue

            try:
                text = self.listener.transcribe(audio_array)
            except ListenerError as e:
                logger.error("Transcription failed: %s", e)
                raise

            if not text.strip():
                continue

            if self._is_exit_command(text):
                logger.info("Exit command detected ('%s'). Stopping loop.", text.strip())
                self._safe_speak("スマートスピーカーを終了します。またいつでも話しかけてください。")
                break

            # Delegate the full pipeline (search → generate → speak) to the
            # existing single-turn method, which also handles history.
            try:
                self._process_turn(text, audio_array)
            except (GenerationError, SpeakerError) as e:
                logger.error("Turn error (continuing loop): %s", e)

    def _process_turn(self, text: str, audio_array):
        """Execute one full pipeline turn for pre-transcribed text.

        Extracted so that run_loop can reuse the post-transcription pipeline
        without re-recording audio.
        """
        # --- Repeat command ---
        if ConversationHistory.is_repeat_command(text):
            last = self.history.last_answer()
            if last:
                logger.info("Repeat command detected — replaying last answer.")
                self._safe_speak(last)
            else:
                self._safe_speak("まだ会話の履歴がありません。")
            return

        # --- Web search ---
        try:
            search_results = self.retriever.search_web(text)
        except SearchError as e:
            logger.warning("Search failed, proceeding without context: %s", e)
            search_results = []

        # --- Compose & generate ---
        history_text = self.history.format_for_prompt()
        prompt = self.composer.compose_prompt(text, search_results, history_text)

        response = self.brain.generate_response(prompt)
        self.history.add(text, response)

        # --- Speak ---
        self.speaker.speak(response)

    def cleanup(self):
        """Clean up resources"""
        pass


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        assistant = VoiceAssistant()
    except Exception as e:
        logger.critical("Failed to initialize Voice Assistant: %s", e, exc_info=True)
        raise SystemExit(1) from e

    try:
        assistant.run_loop()
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except (ListenerError, GenerationError) as e:
        logger.error("Fatal pipeline error: %s", e)
        raise SystemExit(1) from e
    finally:
        assistant.cleanup()


if __name__ == "__main__":
    main()
