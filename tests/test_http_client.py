#!/usr/bin/env python3
"""
Tests for shared http_client module
"""

import pytest
import requests
from unittest.mock import Mock, patch
from src.http_client import http_get_json, http_post_json, http_get_text
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


def test_http_get_text_success():
    """Test successful GET text request"""
    mock_response = Mock()
    mock_response.text = "<html><body>Hello</body></html>"
    mock_response.raise_for_status = Mock()

    with patch('src.http_client.requests.get', return_value=mock_response):
        result = http_get_text("http://example.com", SearchError, "TestService")

    assert result == "<html><body>Hello</body></html>"


def test_http_get_text_connection_error():
    """Test GET text raises error_class on connection failure"""
    with patch('src.http_client.requests.get',
               side_effect=requests.exceptions.ConnectionError("refused")):
        with pytest.raises(SearchError, match="Cannot connect to TestService"):
            http_get_text("http://example.com", SearchError, "TestService")


def test_http_get_text_timeout():
    """Test GET text raises error_class on timeout"""
    with patch('src.http_client.requests.get',
               side_effect=requests.exceptions.Timeout("timed out")):
        with pytest.raises(SearchError, match="timed out"):
            http_get_text("http://example.com", SearchError, "TestService")


def test_http_get_text_http_error():
    """Test GET text raises error_class on HTTP error"""
    mock_response = Mock()
    mock_response.status_code = 404
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "404 Not Found"
    )

    with patch('src.http_client.requests.get', return_value=mock_response):
        with pytest.raises(SearchError, match="HTTP 404"):
            http_get_text("http://example.com", SearchError, "TestService")
