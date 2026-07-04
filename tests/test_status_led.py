#!/usr/bin/env python3
"""
Tests for status LED module.
"""

import sys
from unittest.mock import MagicMock, patch

sys.modules['gpiozero'] = MagicMock()

from src.status_led import StatusLED, LedState  # noqa: E402


def _prepare_gpiozero(mock_led=None):
    sys.modules['gpiozero'].reset_mock()
    if mock_led is None:
        mock_led = MagicMock()
    sys.modules['gpiozero'].LED.return_value = mock_led
    sys.modules['gpiozero'].LED.side_effect = None
    return mock_led


def test_status_led_state_mapping():
    mock_led = _prepare_gpiozero()

    with patch('src.status_led.STATUS_LED_ENABLED', True), \
         patch('src.status_led.STATUS_LED_PIN', 17):
        status_led = StatusLED()

        status_led.set_state(LedState.IDLE)
        mock_led.off.assert_called_once_with()
        mock_led.reset_mock()

        status_led.set_state(LedState.LISTENING)
        mock_led.on.assert_called_once_with()
        mock_led.reset_mock()

        status_led.set_state(LedState.SEARCHING)
        mock_led.blink.assert_called_once_with(0.15, 0.15)
        mock_led.reset_mock()

        status_led.set_state(LedState.THINKING)
        mock_led.blink.assert_called_once_with(0.5, 0.5)
        mock_led.reset_mock()

        status_led.set_state(LedState.SPEAKING)
        mock_led.on.assert_called_once_with()
        mock_led.reset_mock()

        status_led.set_state(LedState.ERROR)
        mock_led.blink.assert_called_once_with(0.08, 0.08)

        status_led.close()
        mock_led.close.assert_called_once_with()


def test_status_led_disabled_is_noop():
    _prepare_gpiozero()
    with patch('src.status_led.STATUS_LED_ENABLED', False), \
         patch('src.status_led.STATUS_LED_PIN', 17):
        status_led = StatusLED()

        sys.modules['gpiozero'].LED.assert_not_called()
        status_led.set_state(LedState.SEARCHING)
        assert status_led._state == LedState.SEARCHING
        status_led.close()


def test_status_led_construction_failure_disables_led():
    _prepare_gpiozero()
    sys.modules['gpiozero'].LED.side_effect = RuntimeError("no pin factory")

    with patch('src.status_led.STATUS_LED_ENABLED', True), \
         patch('src.status_led.STATUS_LED_PIN', 17):
        status_led = StatusLED()

        assert status_led._led is None
        assert status_led._enabled is False
        status_led.set_state(LedState.ERROR)
        status_led.close()


def test_status_led_close_is_safe_when_disabled():
    _prepare_gpiozero()
    with patch('src.status_led.STATUS_LED_ENABLED', False), \
         patch('src.status_led.STATUS_LED_PIN', 17):
        status_led = StatusLED()
        status_led.close()
