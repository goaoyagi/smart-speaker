#!/usr/bin/env python3
"""
Tests for brain module
"""

import pytest
from unittest.mock import Mock, patch
from src.brain import Brain


@pytest.fixture
def brain():
    """Create Brain instance"""
    return Brain()


def test_brain_initialization(brain):
    """Test that Brain initializes correctly"""
    assert brain.ollama_api_url == "http://localhost:11434/api/generate"
    assert brain.ollama_model == "qwen2.5:3b"


def test_generate_response_success(brain):
    """Test successful AI response generation"""
    mock_response = Mock()
    mock_response.json.return_value = {'response': 'テスト回答'}
    mock_response.raise_for_status = Mock()
    
    with patch('src.brain.requests.post', return_value=mock_response):
        result = brain.generate_response("テストプロンプト")
    
    assert result == 'テスト回答'


def test_generate_response_error(brain):
    """Test AI response generation with error"""
    with patch('src.brain.requests.post', side_effect=Exception("API error")):
        result = brain.generate_response("テストプロンプト")
    
    assert "申し訳ありません" in result
