#!/usr/bin/env python3
"""
Shared HTTP client - Eliminates duplicated request/error-handling logic
in retriever.py and brain.py.

Each caller passes its own exception class so the generic HTTP logic
still raises domain-specific errors (SearchError, GenerationError).
"""

import requests


def http_get_json(url, error_class, service_name, params=None, headers=None, timeout=10):
    """Perform a GET request and return parsed JSON.

    Raises error_class (wrapping the underlying requests exception) on failure.
    """
    try:
        response = requests.get(url, params=params, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError as e:
        raise error_class(f"Cannot connect to {service_name} at {url}: {e}") from e
    except requests.exceptions.Timeout as e:
        raise error_class(f"{service_name} request timed out: {e}") from e
    except requests.exceptions.HTTPError as e:
        raise error_class(
            f"{service_name} returned an error (HTTP {response.status_code}): {e}"
        ) from e
    except (ValueError, requests.exceptions.JSONDecodeError) as e:
        raise error_class(f"Invalid JSON response from {service_name}: {e}") from e


def http_post_json(url, error_class, service_name, json_body=None, timeout=30):
    """Perform a POST request and return parsed JSON.

    Raises error_class (wrapping the underlying requests exception) on failure.
    """
    try:
        response = requests.post(url, json=json_body, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError as e:
        raise error_class(f"Cannot connect to {service_name} at {url}: {e}") from e
    except requests.exceptions.Timeout as e:
        raise error_class(f"{service_name} request timed out after {timeout}s: {e}") from e
    except requests.exceptions.HTTPError as e:
        raise error_class(
            f"{service_name} returned an error (HTTP {response.status_code}): {e}"
        ) from e
    except (ValueError, requests.exceptions.JSONDecodeError) as e:
        raise error_class(f"Invalid JSON response from {service_name}: {e}") from e
