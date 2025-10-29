import pytest
from unittest.mock import patch, MagicMock
import logging
from app.log.logger import (
    get_main_logger,
    get_database_logger,
    get_config_logger,
    setup_access_logging,
    get_request_logger,
    Logger,
    AccessLogFormatter,
    ColoredFormatter,
    redact_key_for_logging,
)

def test_get_loggers():
    """Test the logger factory functions."""
    assert get_main_logger().name == "main"
    assert get_database_logger().name == "database"
    assert get_config_logger().name == "config"
    assert setup_access_logging().name == "uvicorn.access"
    assert get_request_logger().name == "request"

@patch("logging.Logger.setLevel")
def test_logger_update_log_levels(mock_set_level):
    """Test the Logger.update_log_levels method."""
    Logger.update_log_levels("DEBUG")
    assert mock_set_level.call_count > 0 # Check that it's called at least once

def test_get_logger():
    """Test the get_logger method."""
    assert Logger.get_logger("main") is not None
    assert Logger.get_logger("non_existent_logger") is None

@patch("app.log.logger.redact_key_for_logging")
def test_access_log_formatter(mock_redact):
    """Test the AccessLogFormatter."""
    mock_redact.return_value = "AIzaSy..."
    formatter = AccessLogFormatter()
    key = "AIza" + "a" * 35
    record = logging.LogRecord('test', logging.INFO, 'test', 1, f'some message with {key}', None, None)
    formatter.format(record)
    mock_redact.assert_called_once_with(key)

def test_colored_formatter():
    """Test the ColoredFormatter."""
    formatter = ColoredFormatter()
    record = logging.LogRecord('test', logging.INFO, 'test.py', 1, 'formatted message', None, None)
    formatter.format(record)
    assert "\033[32mINFO\033[0m" in record.levelname
    assert "[test.py:1]" in record.fileloc

def test_redact_key_for_logging():
    """Test the redact_key_for_logging function."""
    assert redact_key_for_logging("1234567890123") == "123456...890123"
    assert redact_key_for_logging("short") == "sho...ort"
    assert redact_key_for_logging("") == ""
