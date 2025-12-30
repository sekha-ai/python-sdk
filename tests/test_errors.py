"""Tests for custom exceptions"""

import pytest
from sekha.errors import (
    SekhaError,
    SekhaAPIError,
    SekhaNotFoundError,
    SekhaAuthError,
    SekhaConnectionError,
    SekhaValidationError,
    SekhaRateLimitError,
)


class TestSekhaAPIError:
    def test_api_error_creation(self):
        """Test API error with status code"""
        error = SekhaAPIError(
            message="Bad Request",
            status_code=400,
            response='{"detail": "Invalid data"}'
        )
        assert error.status_code == 400
        assert "400" in str(error)
        assert "Bad Request" in str(error)

    def test_api_error_inheritance(self):
        """Test that API error inherits from base SekhaError"""
        error = SekhaAPIError("Test", 500, "")
        assert isinstance(error, SekhaError)


class TestSekhaNotFoundError:
    def test_not_found_error(self):
        """Test not found error"""
        error = SekhaNotFoundError("Conversation not found")
        assert "not found" in str(error)
        assert isinstance(error, SekhaError)


class TestSekhaAuthError:
    def test_auth_error(self):
        """Test authentication error"""
        error = SekhaAuthError("Invalid API key")
        assert "Invalid API key" in str(error)
        assert isinstance(error, SekhaError)


class TestSekhaConnectionError:
    def test_connection_error(self):
        """Test connection error"""
        error = SekhaConnectionError("Cannot connect to server")
        assert "connect" in str(error).lower()
        assert isinstance(error, SekhaError)


class TestSekhaValidationError:
    def test_validation_error(self):
        """Test validation error"""
        error = SekhaValidationError("Invalid input data")
        assert "Invalid" in str(error)
        assert isinstance(error, SekhaError)


class TestSekhaRateLimitError:
    def test_rate_limit_error(self):
        """Test rate limit error"""
        error = SekhaRateLimitError("Rate limit exceeded")
        assert "limit" in str(error).lower()
        assert isinstance(error, SekhaError)