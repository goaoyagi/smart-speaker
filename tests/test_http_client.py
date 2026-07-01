#!/usr/bin/env python3
"""
Tests for shared http_client module
"""

import pytest
import requests
from unittest.mock import Mock, patch
from src.http_client import http_get_json, http_post_json
from src.exceptions import SearchError, GenerationError


def test_http_get_json_success():
    """Test successful GET request"""
    mock_response = Mock()
    mock_response.json.return_value = {'key': 'value'}
    mock_response.raise_for_status = Mock()

    with patch('src.http_client.requests.get', return_value=mock_response):
        result = http_get_json("http://example.com/api", SearchError, "TestService")

    assert result == {'key': 'value'}


def test_http_get_json_connection_error():
    """Test GET request raises error_class on connection failure"""
    with patch('src.http_client.requests.get',
               side_effect=requests.exceptions.ConnectionError("refused")):
        with pytest.raises(SearchError, match="Cannot connect to TestService"):
            http_get_json("http://example.com/api", SearchError, "TestService")


def test_http_get_json_timeout():
    """Test GET request raises error_class on timeout"""
    with patch('src.http_client.requests.get',
               side_effect=requests.exceptions.Timeout("timed out")):
        with pytest.raises(SearchError, match="timed out"):
            http_get_json("http://example.com/api", SearchError, "TestService")


def test_http_post_json_success():
    """Test successful POST request"""
    mock_response = Mock()
    mock_response.json.return_value = {'response': 'ok'}
    mock_response.raise_for_status = Mock()

    with patch('src.http_client.requests.post', return_value=mock_response):
        result = http_post_json(
            "http://example.com/api", GenerationError, "TestService",
            json_body={'q': 'test'}
        )

    assert result == {'response': 'ok'}


def test_http_post_json_connection_error():
    """Test POST request raises error_class on connection failure"""
    with patch('src.http_client.requests.post',
               side_effect=requests.exceptions.ConnectionError("refused")):
        with pytest.raises(GenerationError, match="Cannot connect to TestService"):
            http_post_json(
                "http://example.com/api", GenerationError, "TestService",
                json_body={'q': 'test'}
            )


def test_http_post_json_timeout():
    """Test POST request raises error_class on timeout"""
    with patch('src.http_client.requests.post',
               side_effect=requests.exceptions.Timeout("timed out")):
        with pytest.raises(GenerationError, match="timed out"):
            http_post_json(
                "http://example.com/api", GenerationError, "TestService",
                json_body={'q': 'test'}
            )
