#!/usr/bin/env python3
"""
Push-to-talk module - GPIO button trigger for hold-to-record.

Wraps a ``gpiozero.Button`` so the assistant can record only while the
button is held down. Mirrors ``status_led.py``: on non-Pi environments or
when the GPIO pin is unavailable, the button is automatically disabled and
callers can fall back to fixed-duration recording.
"""

import logging

from .audio_utils import log_init, log_ready
from .config import PUSH_TO_TALK_ENABLED, PTT_BUTTON_PIN, PTT_BOUNCE_TIME

logger = logging.getLogger(__name__)


class PushToTalkButton:
    def __init__(self):
        log_init("Push-to-talk button")
        self._button = None
        self._enabled = PUSH_TO_TALK_ENABLED

        if not self._enabled:
            log_ready("Push-to-talk button")
            return

        try:
            import gpiozero
            self._button = gpiozero.Button(PTT_BUTTON_PIN, bounce_time=PTT_BOUNCE_TIME)
        except Exception as e:
            logger.warning("Push-to-talk button unavailable: %s", e)
            self._button = None
            self._enabled = False
        log_ready("Push-to-talk button")

    @property
    def available(self):
        """True when a usable GPIO button is wired up."""
        return self._enabled and self._button is not None

    def wait_for_press(self, timeout=None):
        """Block until the button is pressed. Returns True if pressed."""
        if not self.available:
            return False
        return self._button.wait_for_press(timeout)

    def wait_for_release(self, timeout=None):
        """Block until the button is released. Returns True if released."""
        if not self.available:
            return True
        return self._button.wait_for_release(timeout)

    def close(self):
        if self._button is None:
            return
        try:
            self._button.close()
        except Exception as e:
            logger.warning("Push-to-talk button close failed: %s", e)
        finally:
            self._button = None
