#!/usr/bin/env python3
"""
Pytest configuration - Common fixtures and mocks
"""

import pytest
import numpy as np


@pytest.fixture
def mock_audio_array():
    """Mock audio array for testing"""
    return np.random.uniform(-0.5, 0.5, 16000).astype(np.float32)


@pytest.fixture
def mock_search_results():
    """Mock search results for testing"""
    return [
        {
            'title': 'テスト結果1',
            'content': 'これはテストコンテンツです。',
            'url': 'http://example.com/1'
        },
        {
            'title': 'テスト結果2',
            'content': '別のテストコンテンツ。',
            'url': 'http://example.com/2'
        }
    ]


@pytest.fixture
def mock_transcribed_text():
    """Mock transcribed text for testing"""
    return "今日の天気はどうですか"
