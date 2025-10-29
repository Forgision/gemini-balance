"""
Unit tests for API key redaction functionality
"""
# Note: pytest is a development dependency and needs to be installed separately.
import logging
from unittest.mock import patch, MagicMock

import pytest
from app.utils.helpers import redact_key_for_logging
from app.log.logger import AccessLogFormatter


def test_valid_long_key_redaction():
    """Test redaction of valid long API keys"""
    # Test Google/Gemini API key
    gemini_key = "AIzaSyDhKGfJ8xYzQwErTyUiOpLkMnBvCxDfGhI"
    result = redact_key_for_logging(gemini_key)
    expected = "AIzaSy...xDfGhI"
    assert result == expected

    # Test OpenAI API key
    openai_key = "sk-1234567890abcdef1234567890abcdef1234567890abcdef"
    result = redact_key_for_logging(openai_key)
    expected = "sk-123...abcdef"
    assert result == expected


def test_short_key_handling():
    """Test handling of short keys"""
    short_key = "short"
    result = redact_key_for_logging(short_key)
    assert result == "[SHORT_KEY]"

    # Test exactly 12 characters (boundary case)
    boundary_key = "123456789012"
    result = redact_key_for_logging(boundary_key)
    assert result == "[SHORT_KEY]"


def test_empty_and_none_keys():
    """Test handling of empty and None keys"""
    # Test empty string
    result = redact_key_for_logging("")
    assert result == "[INVALID_KEY]"

    # Test None
    result = redact_key_for_logging(None)
    assert result == "[INVALID_KEY]"


@pytest.mark.parametrize("invalid_input", [
    123,
    ["key"],
    {"key": "value"}
])
def test_invalid_input_types(invalid_input):
    """Test handling of invalid input types"""
    result = redact_key_for_logging(invalid_input)
    assert result == "[INVALID_KEY]"


def test_boundary_cases():
    """Test boundary cases for key length"""
    # Test 13 characters (just above the threshold)
    key_13 = "1234567890123"
    result = redact_key_for_logging(key_13)
    expected = "123456...890123"
    assert result == expected

    # Test very long key
    long_key = "a" * 100
    result = redact_key_for_logging(long_key)
    expected = "aaaaaa...aaaaaa"
    assert result == expected


@pytest.fixture
def formatter():
    """Fixture for AccessLogFormatter"""
    return AccessLogFormatter()


def test_gemini_key_redaction_in_url(formatter):
    """Test redaction of Gemini API keys in URLs"""
    log_message = 'POST /verify-key/AIzaSyDhKGfJ8xYzQwErTyUiOpLkMnBvCxDfGhI HTTP/1.1" 200'
    result = formatter._redact_api_keys_in_message(log_message)
    assert "AIzaSy...xDfGhI" in result
    assert "AIzaSyDhKGfJ8xYzQwErTyUiOpLkMnBvCxDfGhI" not in result


def test_openai_key_redaction_in_url(formatter):
    """Test redaction of OpenAI API keys in URLs"""
    log_message = 'GET /api/models?key=sk-1234567890abcdef1234567890abcdef1234567890abcdef HTTP/1.1" 200'
    result = formatter._redact_api_keys_in_message(log_message)
    assert "sk-123...abcdef" in result
    assert "sk-1234567890abcdef1234567890abcdef1234567890abcdef" not in result


def test_multiple_keys_in_message(formatter):
    """Test redaction of multiple API keys in a single message"""
    log_message = "Request with keys: AIzaSyDhKGfJ8xYzQwErTyUiOpLkMnBvCxDfGhI and sk-1234567890abcdef1234567890abcdef1234567890abcdef"
    result = formatter._redact_api_keys_in_message(log_message)
    assert "AIzaSy...xDfGhI" in result
    assert "sk-123...abcdef" in result
    assert "AIzaSyDhKGfJ8xYzQwErTyUiOpLkMnBvCxDfGhI" not in result
    assert "sk-1234567890abcdef1234567890abcdef1234567890abcdef" not in result


def test_no_keys_in_message(formatter):
    """Test that messages without API keys are unchanged"""
    log_message = 'GET /api/health HTTP/1.1" 200'
    result = formatter._redact_api_keys_in_message(log_message)
    assert result == log_message


def test_partial_key_patterns_not_redacted(formatter):
    """Test that partial key patterns are not redacted"""
    log_message = "Message with partial patterns: AIza sk- incomplete"
    result = formatter._redact_api_keys_in_message(log_message)
    assert result == log_message


def test_error_handling_in_redaction(formatter):
    """Test error handling in the redaction process"""
    original_patterns = formatter.compiled_patterns
    mock_pattern = MagicMock()
    mock_pattern.sub.side_effect = Exception("Regex error")
    formatter.compiled_patterns = [mock_pattern]

    try:
        log_message = 'POST /verify-key/AIzaSyDhKGfJ8xYzQwErTyUiOpLkMnBvCxDfGhI HTTP/1.1" 200'
        result = formatter._redact_api_keys_in_message(log_message)
        assert result == "[LOG_REDACTION_ERROR]"
    finally:
        formatter.compiled_patterns = original_patterns


def test_format_method(formatter):
    """Test the format method of AccessLogFormatter"""
    record = MagicMock(spec=logging.LogRecord)
    record.getMessage.return_value = 'POST /verify-key/AIzaSyDhKGfJ8xYzQwErTyUiOpLkMnBvCxDfGhI HTTP/1.1" 200'

    with patch('logging.Formatter.format', return_value='2025-01-01 12:00:00 | INFO | POST /verify-key/AIzaSyDhKGfJ8xYzQwErTyUiOpLkMnBvCxDfGhI HTTP/1.1" 200'):
        result = formatter.format(record)
        assert "AIzaSy...xDfGhI" in result
        assert "AIzaSyDhKGfJ8xYzQwErTyUiOpLkMnBvCxDfGhI" not in result


def test_regex_patterns_compilation():
    """Test that regex patterns are properly compiled"""
    formatter = AccessLogFormatter()
    assert len(formatter.compiled_patterns) == 2
    assert all(hasattr(pattern, 'sub') for pattern in formatter.compiled_patterns)


@pytest.mark.parametrize("test_key", [
    "sk-1234567890abcdef1234567890abcdef1234567890abcdef",
    "sk-proj-1234567890abcdef1234567890abcdef1234567890abcdef",
    "sk-1234567890abcdef_1234567890abcdef-1234567890abcdef",
    "sk-12345678901234567890",
])
def test_flexible_openai_pattern(formatter, test_key):
    """Test the flexible OpenAI pattern matches various formats"""
    log_message = f"Request with key: {test_key}"
    result = formatter._redact_api_keys_in_message(log_message)
    assert test_key not in result
    assert "sk-" in result
