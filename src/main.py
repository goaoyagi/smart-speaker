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
from .status_led import StatusLED, LedState
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
        self.status_led = StatusLED()

        print("Voice Assistant initialized!")

    def listen_and_respond(self):
        """Main conversation loop"""
        logger.info("Voice Assistant Ready — Speak now (will record for 10 seconds)...")

        try:
            # --- Record & validate audio ---
            self.status_led.set_state(LedState.LISTENING)
            try:
                audio_array = self.listener.record_audio()
            except ListenerError as e:
                logger.error("Recording failed: %s", e)
                raise

            max_level = np.max(np.abs(audio_array))
            if max_level < 0.03:
                logger.info("No speech detected (audio level too low: %.4f).", max_level)
                self._safe_speak("音声が検出されませんでした。")
                self.status_led.set_state(LedState.IDLE)
                return

            # --- Transcribe ---
            try:
                text = self.listener.transcribe(audio_array)
            except ListenerError as e:
                logger.error("Transcription failed: %s", e)
                raise

            if not text.strip():
                logger.info("No speech detected.")
                self.status_led.set_state(LedState.IDLE)
                return

            # --- Web search (non-fatal: degrade to no-context prompt) ---
            self.status_led.set_state(LedState.SEARCHING)
            try:
                search_results = self.retriever.search_web(text)
            except SearchError as e:
                logger.warning("Search failed, proceeding without context: %s", e)
                search_results = []

            # --- Compose & generate ---
            self.status_led.set_state(LedState.THINKING)
            prompt = self.composer.compose_prompt(text, search_results)

            try:
                response = self.brain.generate_response(prompt)
            except GenerationError as e:
                logger.error("AI generation failed: %s", e)
                self._safe_speak("申し訳ありませんが、回答を生成できませんでした。")
                raise

            # --- Speak ---
            self.status_led.set_state(LedState.SPEAKING)
            try:
                self.speaker.speak(response)
            except SpeakerError as e:
                logger.error("Speech output failed: %s", e)
                raise

            self.status_led.set_state(LedState.IDLE)
        except Exception:
            self.status_led.set_state(LedState.ERROR)
            raise

    def _safe_speak(self, text):
        """Attempt to speak; log and continue if it fails."""
        try:
            self.speaker.speak(text)
        except SpeakerError as e:
            logger.warning("Could not speak error message: %s", e)

    def cleanup(self):
        """Clean up resources"""
        self.status_led.close()


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
        assistant.listen_and_respond()
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except (ListenerError, SearchError, GenerationError, SpeakerError) as e:
        logger.error("Pipeline error: %s", e)
        raise SystemExit(1) from e
    finally:
        assistant.cleanup()


if __name__ == "__main__":
    main()
