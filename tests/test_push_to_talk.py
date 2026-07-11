#!/usr/bin/env python3
"""
Tests for push-to-talk button module.
"""

import sys
from unittest.mock import MagicMock, patch

sys.modules['gpiozero'] = MagicMock()

from src.push_to_talk import PushToTalkButton  # noqa: E402


def _prepare_gpiozero(mock_button=None):
    sys.modules['gpiozero'].reset_mock()
    if mock_button is None:
        mock_button = MagicMock()
    sys.modules['gpiozero'].Button.return_value = mock_button
    sys.modules['gpiozero'].Button.side_effect = None
    return mock_button


def test_button_available_and_events():
    mock_button = _prepare_gpiozero()
    mock_button.wait_for_press.return_value = True
    mock_button.wait_for_release.return_value = True

    with patch('src.push_to_talk.PUSH_TO_TALK_ENABLED', True), \
         patch('src.push_to_talk.PTT_BUTTON_PIN', 17), \
         patch('src.push_to_talk.PTT_BOUNCE_TIME', 0.05):
        button = PushToTalkButton()

        assert button.available is True
        sys.modules['gpiozero'].Button.assert_called_once_with(17, bounce_time=0.05)

        assert button.wait_for_press() is True
        assert button.wait_for_release(timeout=5) is True
        mock_button.wait_for_press.assert_called_once()
        mock_button.wait_for_release.assert_called_once_with(5)

        button.close()
        mock_button.close.assert_called_once_with()


def test_button_disabled_is_noop():
    _prepare_gpiozero()
    with patch('src.push_to_talk.PUSH_TO_TALK_ENABLED', False), \
         patch('src.push_to_talk.PTT_BUTTON_PIN', 17):
        button = PushToTalkButton()

        sys.modules['gpiozero'].Button.assert_not_called()
        assert button.available is False
        assert button.wait_for_press() is False
        assert button.wait_for_release() is True
        button.close()


def test_button_construction_failure_disables():
    _prepare_gpiozero()
    sys.modules['gpiozero'].Button.side_effect = RuntimeError("no pin factory")

    with patch('src.push_to_talk.PUSH_TO_TALK_ENABLED', True), \
         patch('src.push_to_talk.PTT_BUTTON_PIN', 17):
        button = PushToTalkButton()

        assert button.available is False
        assert button.wait_for_press() is False
        button.close()


def test_button_close_is_safe_when_disabled():
    _prepare_gpiozero()
    with patch('src.push_to_talk.PUSH_TO_TALK_ENABLED', False):
        button = PushToTalkButton()
        button.close()
