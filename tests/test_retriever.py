#!/usr/bin/env python3
"""
Tests for retriever module
"""

import pytest
import requests
from unittest.mock import Mock, patch
from src.retriever import Retriever
from src.exceptions import SearchError


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

    with patch('src.http_client.requests.get', return_value=mock_response):
        results = retriever.search_web("test query")

    assert len(results) == 1
    assert results[0]['title'] == 'Test'


def test_search_web_connection_error(retriever):
    """Test web search raises SearchError on connection failure"""
    with patch('src.http_client.requests.get',
               side_effect=requests.exceptions.ConnectionError("Connection refused")):
        with pytest.raises(SearchError, match="Cannot connect to SearXNG"):
            retriever.search_web("test query")


def test_search_web_timeout(retriever):
    """Test web search raises SearchError on timeout"""
    with patch('src.http_client.requests.get',
               side_effect=requests.exceptions.Timeout("timed out")):
        with pytest.raises(SearchError, match="timed out"):
            retriever.search_web("test query")
