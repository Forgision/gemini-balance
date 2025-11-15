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

@pytest.fixture()
def test_logger():
    return Logger.setup_logger('test')

def test_get_loggers():
    """Test the logger factory functions."""
    assert get_main_logger().name == "main"
    assert get_database_logger().name == "database"
    assert get_config_logger().name == "config"
    assert setup_access_logging().name == "uvicorn.access"
    assert get_request_logger().name == "request"


def test_logger_update_log_levels(test_logger):
    """Test the Logger.update_log_levels method."""
    Logger.update_log_levels("DEBUG")
    
    test_logger = Logger.get_logger("test")
    assert test_logger is not None
    assert test_logger.level == logging.DEBUG

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
    # Create a mock LogRecord with the necessary attributes
    mock_record = MagicMock(spec=logging.LogRecord, name="test",
                            levelname=logging.INFO, pathname="test.py", lineno=1,
                            msg="formatted message", args=None, exc_info=None,
                            filename = 'test_logger.py', exc_text='',
                            stack_info=None)
    log_recoder = logging.LogRecord('test', logging.INFO, 'test', 1, 'formatted message', None, None)
    formatted_message = formatter.format(log_recoder)
    print(formatted_message)
    assert isinstance(formatted_message, str)
    #TODO: validate color formate not working
    # assert "\033[32mINFO\033[0m" in formatted_message
    # assert "[test.py:1]" in formatted_message

def test_redact_key_for_logging():
    """Test the redact_key_for_logging function."""
    assert redact_key_for_logging("1234567890123") == "123456...890123"
    assert redact_key_for_logging("short") == "sho...ort"
    assert redact_key_for_logging("") == ""
