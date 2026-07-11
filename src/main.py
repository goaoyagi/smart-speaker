#!/usr/bin/env python3
"""
Main orchestrator - Coordinates all components for voice assistant
"""

import logging
import time
import numpy as np
from .listener import Listener
from .retriever import Retriever
from .composer import Composer
from .brain import Brain
from .speaker import Speaker
from .status_led import StatusLED, LedState
from .push_to_talk import PushToTalkButton
from .config import PTT_MIN_RECORD_SECONDS, PTT_MAX_RECORD_SECONDS
from .exceptions import (
    VoiceAssistantError,
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
        self.button = PushToTalkButton()

        print("Voice Assistant initialized!")

    def run(self):
        """Run the assistant using push-to-talk when a button is available,
        otherwise fall back to a single fixed-duration recording."""
        if self.button.available:
            self.run_push_to_talk()
        else:
            self.listen_and_respond()

    def listen_and_respond(self):
        """Single conversation turn using fixed-duration recording."""
        logger.info("Voice Assistant Ready — Speak now (will record for 10 seconds)...")

        try:
            self.status_led.set_state(LedState.LISTENING)
            try:
                audio_array = self.listener.record_audio()
            except ListenerError as e:
                logger.error("Recording failed: %s", e)
                raise

            self._handle_audio(audio_array)
        except Exception:
            self.status_led.set_state(LedState.ERROR)
            raise

    def run_push_to_talk(self):
        """Event-driven loop: record while the button is held, then respond."""
        logger.info("Push-to-talk ready — hold the button and speak...")

        try:
            while True:
                self.status_led.set_state(LedState.IDLE)
                self.button.wait_for_press()
                try:
                    self._push_to_talk_turn()
                except VoiceAssistantError as e:
                    logger.error("Conversation turn failed: %s", e)
                    self.status_led.set_state(LedState.ERROR)
        except KeyboardInterrupt:
            logger.info("Interrupted by user.")

    def _push_to_talk_turn(self):
        """Record while the button is held, then run the response pipeline."""
        self.status_led.set_state(LedState.LISTENING)
        self.listener.start_recording()
        pressed_at = time.monotonic()

        if not self.button.wait_for_release(timeout=PTT_MAX_RECORD_SECONDS):
            logger.info("Maximum recording time reached; stopping.")

        held_for = time.monotonic() - pressed_at
        audio_array = self.listener.stop_recording()

        if held_for < PTT_MIN_RECORD_SECONDS:
            logger.info("Button held too briefly (%.2fs); ignoring.", held_for)
            self.status_led.set_state(LedState.IDLE)
            return

        self._handle_audio(audio_array)

    def _handle_audio(self, audio_array):
        """Validate, transcribe, search, generate and speak for one turn."""
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

    def _safe_speak(self, text):
        """Attempt to speak; log and continue if it fails."""
        try:
            self.speaker.speak(text)
        except SpeakerError as e:
            logger.warning("Could not speak error message: %s", e)

    def cleanup(self):
        """Clean up resources"""
        self.status_led.close()
        self.button.close()


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
        assistant.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except (ListenerError, SearchError, GenerationError, SpeakerError) as e:
        logger.error("Pipeline error: %s", e)
        raise SystemExit(1) from e
    finally:
        assistant.cleanup()


if __name__ == "__main__":
    main()
