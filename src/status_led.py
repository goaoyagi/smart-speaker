#!/usr/bin/env python3
"""
Status LED module - GPIO visual feedback for assistant state.
"""

import logging
from enum import Enum

from .audio_utils import log_init, log_ready
from .config import STATUS_LED_ENABLED, STATUS_LED_PIN

logger = logging.getLogger(__name__)


class LedState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    SEARCHING = "searching"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR = "error"


class StatusLED:
    def __init__(self):
        log_init("Status LED")
        self._led = None
        self._state = LedState.IDLE
        self._enabled = STATUS_LED_ENABLED

        if not self._enabled:
            log_ready("Status LED")
            return

        try:
            import gpiozero
            self._led = gpiozero.LED(STATUS_LED_PIN)
        except Exception as e:
            logger.warning("Status LED unavailable: %s", e)
            self._led = None
            self._enabled = False
        log_ready("Status LED")

    def set_state(self, state: LedState):
        self._state = state

        if not self._enabled or self._led is None:
            return

        try:
            if state == LedState.IDLE:
                self._led.off()
            elif state == LedState.LISTENING:
                self._led.on()
            elif state == LedState.SEARCHING:
                self._led.blink(0.15, 0.15)
            elif state == LedState.THINKING:
                self._led.blink(0.5, 0.5)
            elif state == LedState.SPEAKING:
                self._led.on()
            elif state == LedState.ERROR:
                self._led.blink(0.08, 0.08)
        except Exception as e:
            logger.warning("Status LED update failed: %s", e)
            self._enabled = False
            self._led = None

    def close(self):
        if self._led is None:
            return

        try:
            self._led.close()
        except Exception as e:
            logger.warning("Status LED close failed: %s", e)
        finally:
            self._led = None
