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
from .config import EXIT_WORDS
from .exceptions import (
    ListenerError,
    SearchError,
    GenerationError,
    SpeakerError,
)

logger = logging.getLogger(__name__)


class VoiceAssistant:
    def __init__(self):
        print("Initializing Voice Assistant...")

        # Initialize all components
        self.listener = Listener()
        self.retriever = Retriever()
        self.composer = Composer()
        self.brain = Brain()
        self.speaker = Speaker()
        self.history = ConversationHistory()

        print("Voice Assistant initialized!")

    def listen_and_respond(self):
        """Main conversation loop"""
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
            return True

        # --- Transcribe ---
        try:
            text = self.listener.transcribe(audio_array)
        except ListenerError as e:
            logger.error("Transcription failed: %s", e)
            raise

        if not text.strip():
            logger.info("No speech detected.")
            return True

        # --- Check for exit command (Phase C) ---
        if any(word in text for word in EXIT_WORDS):
            logger.info("Exit command detected: %s", text)
            self._safe_speak("会話を終了します。またいつでも話しかけてください。")
            return False

        # --- Web search (non-fatal: degrade to no-context prompt) ---
        try:
            search_results = self.retriever.search_web(text)
        except SearchError as e:
            logger.warning("Search failed, proceeding without context: %s", e)
            search_results = []

        # --- Compose & generate ---
        prompt = self.composer.compose_prompt(text, search_results, self.history)

        try:
            response = self.brain.generate_response(prompt)
        except GenerationError as e:
            logger.error("AI generation failed: %s", e)
            self._safe_speak("申し訳ありませんが、回答を生成できませんでした。")
            raise

        # --- Store turn in history ---
        self.history.add_turn(text, response)

        # --- Speak ---
        try:
            self.speaker.speak(response)
        except SpeakerError as e:
            logger.error("Speech output failed: %s", e)
            raise

        return True

    def run_loop(self):
        """Continuously listen and respond until an exit word is spoken.

        Calls :meth:`listen_and_respond` in a loop.  The loop terminates when:

        * the user speaks one of the :data:`~src.config.EXIT_WORDS`, causing
          ``listen_and_respond`` to return ``False``.
        * a :class:`KeyboardInterrupt` is raised (e.g. Ctrl-C).

        Non-fatal pipeline errors (:class:`~src.exceptions.SearchError`,
        :class:`~src.exceptions.SpeakerError`) are logged and the loop
        continues.  Fatal errors (:class:`~src.exceptions.ListenerError`,
        :class:`~src.exceptions.GenerationError`) propagate to the caller.
        """
        logger.info("Starting continuous loop. Say '終わり' to stop.")
        self._safe_speak(
            "会話を開始します。終わりたいときは「終わり」と言ってください。"
        )

        while True:
            try:
                should_continue = self.listen_and_respond()
                if should_continue is False:
                    break
            except KeyboardInterrupt:
                logger.info("Loop interrupted by user.")
                break
            except (SearchError, SpeakerError) as e:
                logger.warning("Non-fatal error in loop, continuing: %s", e)

    def _safe_speak(self, text):
        """Attempt to speak; log and continue if it fails."""
        try:
            self.speaker.speak(text)
        except SpeakerError as e:
            logger.warning("Could not speak error message: %s", e)

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
    except (ListenerError, SearchError, GenerationError, SpeakerError) as e:
        logger.error("Pipeline error: %s", e)
        raise SystemExit(1) from e
    finally:
        assistant.cleanup()


if __name__ == "__main__":
    main()
