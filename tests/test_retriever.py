#!/usr/bin/env python3
"""
Tests for retriever module
"""

import pytest
from unittest.mock import Mock, patch
from src.retriever import Retriever


@pytest.fixture
def retriever():
    """Create Retriever instance"""
    return Retriever()


def test_retriever_initialization(retriever):
    """Test that Retriever initializes correctly"""
    assert retriever.searxng_url == "http://localhost:8080"


def test_search_web_success(retriever):
    """Test successful web search"""
    mock_response = Mock()
    mock_response.json.return_value = {
        'results': [
            {'title': 'Test', 'content': 'Test content', 'url': 'http://test.com'}
        ]
    }
    mock_response.raise_for_status = Mock()
    
    with patch('src.retriever.requests.get', return_value=mock_response):
        results = retriever.search_web("test query")
    
    assert len(results) == 1
    assert results[0]['title'] == 'Test'


def test_search_web_error(retriever):
    """Test web search with error"""
    with patch('src.retriever.requests.get', side_effect=Exception("Network error")):
        results = retriever.search_web("test query")
    
    assert results == []
