"""Tests for utility functions"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import patch

from sekha.utils import (
    RateLimiter,
    ExponentialBackoff,
    validate_api_key,
    validate_base_url,
    parse_iso_datetime,
    format_bytes,
)


class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_rate_limiter_acquire(self):
        """Test basic rate limiting functionality"""
        limiter = RateLimiter(max_requests=2, window_seconds=1.0)

        # First two requests should be immediate
        start = asyncio.get_event_loop().time()
        await limiter.acquire()
        await limiter.acquire()
        elapsed = asyncio.get_event_loop().time() - start

        assert elapsed < 0.1  # Should be nearly instant

    @pytest.mark.asyncio
    async def test_rate_limiter_blocks_when_at_capacity(self):
        """Test that rate limiter blocks when at max capacity"""
        limiter = RateLimiter(max_requests=1, window_seconds=0.5)

        await limiter.acquire()  # First request
        start = asyncio.get_event_loop().time()
        await limiter.acquire()  # Second request should wait
        elapsed = asyncio.get_event_loop().time() - start

        assert elapsed >= 0.4  # Should wait ~0.5 seconds


class TestExponentialBackoff:
    @pytest.mark.asyncio
    async def test_backoff_increases_delay(self):
        """Test that backoff delay increases exponentially"""
        backoff = ExponentialBackoff(base_delay=0.1, max_delay=1.0, factor=2.0)

        # First wait
        start = asyncio.get_event_loop().time()
        await backoff.wait()
        elapsed1 = asyncio.get_event_loop().time() - start

        # Second wait (should be ~0.2s)
        start = asyncio.get_event_loop().time()
        await backoff.wait()
        elapsed2 = asyncio.get_event_loop().time() - start

        assert elapsed2 > elapsed1 * 1.5  # Should be significantly longer

    @pytest.mark.asyncio
    async def test_backoff_respects_max_delay(self):
        """Test that backoff respects max delay cap"""
        backoff = ExponentialBackoff(base_delay=1.0, max_delay=2.0, factor=2.0)

        await backoff.wait()  # 1s
        await backoff.wait()  # Should be 2s (capped)

        start = asyncio.get_event_loop().time()
        await backoff.wait()  # Should still be 2s
        elapsed = asyncio.get_event_loop().time() - start

        assert 1.7 <= elapsed <= 2.3  # Should be ~2s

    def test_backoff_reset(self):
        """Test resetting backoff counter"""
        backoff = ExponentialBackoff()
        backoff.attempt = 5

        backoff.reset()
        assert backoff.attempt == 0


class TestValidateApiKey:
    def test_valid_api_key(self):
        """Test valid API key format"""
        key = "sk-sekha-" + "a" * 32
        assert validate_api_key(key) is True

    def test_invalid_api_key_too_short(self):
        """Test API key that's too short"""
        with pytest.raises(ValueError, match="too short"):
            validate_api_key("sk-sekha-short")

    def test_invalid_api_key_wrong_prefix(self):
        """Test API key without correct prefix"""
        with pytest.raises(ValueError, match="must start with"):
            validate_api_key("wrong-prefix-" + "a" * 32)

    def test_empty_api_key(self):
        """Test empty API key"""
        with pytest.raises(ValueError, match="API key is required"):
            validate_api_key("")


class TestParseIsoDatetime:
    def test_parse_iso_with_z(self):
        """Test parsing ISO datetime with Z suffix"""
        dt = parse_iso_datetime("2025-12-30T10:30:00Z")
        assert dt.year == 2025
        assert dt.month == 12
        assert dt.day == 30

    def test_parse_iso_with_timezone(self):
        """Test parsing ISO datetime with timezone offset"""
        dt = parse_iso_datetime("2025-12-30T10:30:00+00:00")
        assert dt.tzinfo is not None

    def test_parse_without_microseconds(self):
        """Test parsing datetime without microseconds"""
        dt = parse_iso_datetime("2025-12-30 10:30:00")
        assert dt.hour == 10
        assert dt.minute == 30


class TestFormatBytes:
    def test_format_bytes_bytes(self):
        """Test formatting bytes"""
        assert format_bytes(512) == "512.0 B"

    def test_format_bytes_kilobytes(self):
        """Test formatting kilobytes"""
        assert "KB" in format_bytes(1024)

    def test_format_bytes_megabytes(self):
        """Test formatting megabytes"""
        result = format_bytes(1024 * 1024)
        assert "MB" in result

    def test_format_bytes_gigabytes(self):
        """Test formatting gigabytes"""
        result = format_bytes(1024 * 1024 * 1024)
        assert "GB" in result

    def test_format_bytes_terabytes(self):
        """Test formatting terabytes"""
        result = format_bytes(1024 * 1024 * 1024 * 1024)
        assert "TB" in result

class TestRateLimiterEdgeCases:
    """Test RateLimiter edge cases"""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_concurrent_access(self):
        """Test rate limiter handles concurrent requests correctly"""
        limiter = RateLimiter(max_requests=2, window_seconds=0.5)
        
        # Acquire 2 tokens concurrently
        await asyncio.gather(
            limiter.acquire(),
            limiter.acquire()
        )
        
        # Third should have to wait
        start = asyncio.get_event_loop().time()
        await limiter.acquire()
        elapsed = asyncio.get_event_loop().time() - start
        
        assert elapsed >= 0.4  # Should wait for window reset

    @pytest.mark.asyncio
    async def test_rate_limiter_zero_requests(self):
        """Test rate limiter with zero max_requests"""
        limiter = RateLimiter(max_requests=0, window_seconds=1.0)
        
        # Should always have to wait
        start = asyncio.get_event_loop().time()
        await limiter.acquire()
        elapsed = asyncio.get_event_loop().time() - start
        
        assert elapsed >= 0.9  # Should wait full window

class TestExponentialBackoffEdgeCases:
    """Test ExponentialBackoff edge cases"""
    
    @pytest.mark.asyncio
    async def test_backoff_max_delay_reached(self):
        """Test that backoff caps at max_delay"""
        backoff = ExponentialBackoff(base_delay=1.0, max_delay=2.0, factor=2.0)
        
        # First wait: 1s
        start = asyncio.get_event_loop().time()
        await backoff.wait()
        elapsed1 = asyncio.get_event_loop().time() - start
        
        # Second wait: should be 2s (capped)
        start = asyncio.get_event_loop().time()
        await backoff.wait()
        elapsed2 = asyncio.get_event_loop().time() - start
        
        # Third wait: should still be 2s
        start = asyncio.get_event_loop().time()
        await backoff.wait()
        elapsed3 = asyncio.get_event_loop().time() - start
        
        assert 1.8 <= elapsed2 <= 2.3
        assert 1.8 <= elapsed3 <= 2.3
        assert abs(elapsed2 - elapsed3) < 0.2  # Should be similar

    def test_backoff_reset_multiple_times(self):
        """Test multiple resets work correctly"""
        backoff = ExponentialBackoff()
        
        backoff.attempt = 5
        backoff.reset()
        assert backoff.attempt == 0
        
        backoff.attempt = 10
        backoff.reset()
        assert backoff.attempt == 0

class TestValidateApiKeyEdgeCases:
    """Test API key validation edge cases"""
    
    def test_validate_api_key_non_string(self):
        """Test non-string API key"""
        with pytest.raises(ValueError, match="must be a string"):
            validate_api_key(12345)
    
    def test_validate_base_url_non_string(self):
        """Test non-string base URL"""
        with pytest.raises(ValueError, match="must be a string"):
            validate_base_url(12345)
    
    def test_validate_base_url_no_scheme(self):
        """Test URL without http/https"""
        with pytest.raises(ValueError, match="Invalid base_url.*starting with http"):
            validate_base_url("ftp://example.com")
        
        with pytest.raises(ValueError, match="Invalid base_url.*starting with http"):
            validate_base_url("example.com")
    
    def test_validate_base_url_with_spaces(self):
        """Test URL with spaces"""
        with pytest.raises(ValueError, match="Invalid base_url"):
            validate_base_url("http://example.com/path with spaces")

class TestParseIsoDatetimeEdgeCases:
    """Test datetime parsing edge cases"""
    
    def test_parse_iso_datetime_invalid_format(self):
        """Test parsing invalid datetime string"""
        with pytest.raises(ValueError):
            parse_iso_datetime("not-a-date")
    
    def test_parse_iso_datetime_with_microseconds(self):
        """Test datetime with microseconds"""
        dt = parse_iso_datetime("2025-12-30T10:30:00.123456Z")
        assert dt.microsecond == 123456
    
    def test_parse_iso_datetime_utc_offset(self):
        """Test datetime with UTC offset"""
        dt = parse_iso_datetime("2025-12-30T10:30:00+05:30")
        assert dt.utcoffset() is not None

class TestFormatBytesEdgeCases:
    """Test format_bytes edge cases"""
    
    def test_format_bytes_zero(self):
        """Test zero bytes"""
        assert format_bytes(0) == "0.0 B"
    
    def test_format_bytes_negative(self):
        """Test negative bytes (edge case)"""
        assert format_bytes(-1) == "-1.0 B"
    
    def test_format_bytes_exact_units(self):
        """Test exact unit boundaries"""
        assert "1023.0 B" in format_bytes(1023)  # Actual value
        assert "1.0 KB" in format_bytes(1024)    # Correct expectation
        assert "1.0 MB" in format_bytes(1024 * 1024)
        assert "1.0 GB" in format_bytes(1024 * 1024 * 1024)
        assert "1.0 TB" in format_bytes(1024 * 1024 * 1024 * 1024)
    
    def test_format_bytes_decimal_values(self):
        """Test non-integer byte values"""
        result = format_bytes(1536)  # 1.5 KB
        assert "1.5" in result
        assert "KB" in result        