import logging
import platform
import re
import sys
from typing import Dict, Optional

# ANSI escape sequence color codes
COLORS = {
    "DEBUG": "\033[34m",  # Blue
    "INFO": "\033[32m",  # Green
    "WARNING": "\033[33m",  # Yellow
    "ERROR": "\033[31m",  # Red
    "CRITICAL": "\033[1;31m",  # Red bold
}


# Enable ANSI support on Windows systems
if platform.system() == "Windows":
    import ctypes

    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)


class ColoredFormatter(logging.Formatter):
    """
    Custom log formatter, adds color support
    """

    def format(self, record):
        # Get the color code for the corresponding level
        color = COLORS.get(record.levelname, "")
        # Add color code and reset code
        record.levelname = f"{color}{record.levelname}\033[0m"
        # Create a fixed-width string containing the file name and line number
        record.fileloc = f"[{record.filename}:{record.lineno}]"
        return super().format(record)


class AccessLogFormatter(logging.Formatter):
    """
    Custom access log formatter that redacts API keys in URLs
    """

    # API key patterns to match in URLs
    API_KEY_PATTERNS = [
        r"\bAIza[0-9A-Za-z_-]{35}",  # Google API keys (like Gemini)
        r"\bsk-[0-9A-Za-z_-]{20,}",  # OpenAI and general sk- prefixed keys
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Compile regex patterns for better performance
        self.compiled_patterns = [
            re.compile(pattern) for pattern in self.API_KEY_PATTERNS
        ]

    def format(self, record):
        # Format the record normally first
        formatted_msg = super().format(record)

        # Redact API keys in the formatted message
        return self._redact_api_keys_in_message(formatted_msg)

    def _redact_api_keys_in_message(self, message: str) -> str:
        """
        Replace API keys in log message with redacted versions
        """
        try:
            for pattern in self.compiled_patterns:

                def replace_key(match):
                    key = match.group(0)
                    return redact_key_for_logging(key)

                message = pattern.sub(replace_key, message)

            return message
        except Exception as e:
            # Log the error but don't expose the original message in case it contains keys
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"Error redacting API keys in access log: {e}")
            return "[LOG_REDACTION_ERROR]"


def redact_key_for_logging(key: str) -> str:
    """
    Redacts API key for secure logging by showing only first and last 6 characters.

    Args:
        key: API key to redact

    Returns:
        str: Redacted key in format "first6...last6" or descriptive placeholder for edge cases
    """
    if not key:
        return key

    if len(key) <= 12:
        return f"{key[:3]}...{key[-3:]}"
    else:
        return f"{key[:6]}...{key[-6:]}"


# Log format - use fileloc and set a fixed width (e.g., 30)
FORMATTER = ColoredFormatter(
    "%(asctime)s | %(levelname)-17s | %(fileloc)-30s | %(message)s"
)

# Log level mapping
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


class Logger:
    def __init__(self):
        pass

    _loggers: Dict[str, logging.Logger] = {}

    @staticmethod
    def setup_logger(name: str) -> logging.Logger:
        """
        Set up and get a logger
        :param name: logger name
        :return: logger instance
        """
        # Import the settings object
        from app.config.config import settings

        # Get the log level from the global configuration
        log_level_str = settings.LOG_LEVEL.lower()
        level = LOG_LEVELS.get(log_level_str, logging.INFO)

        if name in Logger._loggers:
            # If the logger already exists, check and update its level (if necessary)
            existing_logger = Logger._loggers[name]
            if existing_logger.level != level:
                existing_logger.setLevel(level)
            return existing_logger

        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = False

        # Add console output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(FORMATTER)
        logger.addHandler(console_handler)

        Logger._loggers[name] = logger
        return logger

    @staticmethod
    def get_logger(name: str) -> Optional[logging.Logger]:
        """
        Get an existing logger
        :param name: logger name
        :return: logger instance or None
        """
        return Logger._loggers.get(name)

    @staticmethod
    def update_log_levels(log_level: str):
        """
        Update the log level of all created loggers based on the current global configuration.
        """
        log_level_str = log_level.lower()
        new_level = LOG_LEVELS.get(log_level_str, logging.INFO)

        updated_count = 0
        for logger_name, logger_instance in Logger._loggers.items():
            if logger_instance.level != new_level:
                logger_instance.setLevel(new_level)
                # Optional: log level change log, but be careful to avoid generating too much log inside the log module
                # print(f"Updated log level for logger '{logger_name}' to {log_level_str.upper()}")
                updated_count += 1


# Predefined loggers
def get_openai_logger():
    return Logger.setup_logger("openai")


def get_gemini_logger():
    return Logger.setup_logger("gemini")


def get_chat_logger():
    return Logger.setup_logger("chat")


def get_model_logger():
    return Logger.setup_logger("model")


def get_security_logger():
    return Logger.setup_logger("security")


def get_key_manager_logger():
    return Logger.setup_logger("key_manager")


def get_main_logger():
    return Logger.setup_logger("main")


def get_embeddings_logger():
    return Logger.setup_logger("embeddings")


def get_request_logger():
    return Logger.setup_logger("request")


def get_retry_logger():
    return Logger.setup_logger("retry")


def get_image_create_logger():
    return Logger.setup_logger("image_create")


def get_exceptions_logger():
    return Logger.setup_logger("exceptions")


def get_application_logger():
    return Logger.setup_logger("application")


def get_initialization_logger():
    return Logger.setup_logger("initialization")


def get_middleware_logger():
    return Logger.setup_logger("middleware")


def get_routes_logger():
    return Logger.setup_logger("routes")


def get_config_routes_logger():
    return Logger.setup_logger("config_routes")


def get_config_logger():
    return Logger.setup_logger("config")


def get_database_logger():
    return Logger.setup_logger("database")


def get_log_routes_logger():
    return Logger.setup_logger("log_routes")


def get_stats_logger():
    return Logger.setup_logger("stats")


def get_update_logger():
    return Logger.setup_logger("update_service")


def get_scheduler_routes():
    return Logger.setup_logger("scheduler_routes")


def get_message_converter_logger():
    return Logger.setup_logger("message_converter")


def get_api_client_logger():
    return Logger.setup_logger("api_client")


def get_openai_compatible_logger():
    return Logger.setup_logger("openai_compatible")


def get_error_log_logger():
    return Logger.setup_logger("error_log")


def get_request_log_logger():
    return Logger.setup_logger("request_log")


def get_files_logger():
    return Logger.setup_logger("files")


def get_vertex_express_logger():
    return Logger.setup_logger("vertex_express")


def get_gemini_embedding_logger():
    return Logger.setup_logger("gemini_embedding")


def setup_access_logging():
    """
    Configure uvicorn access logging with API key redaction

    This function sets up a custom access log formatter that automatically
    redacts API keys in HTTP access logs. It works by:

    1. Intercepting uvicorn's access log messages
    2. Using regex patterns to find API keys in URLs
    3. Replacing them with redacted versions (first6...last6)

    Supported API key formats:
    - Google/Gemini API keys: AIza[35 chars]
    - OpenAI API keys: sk-[48 chars]
    - General sk- prefixed keys: sk-[20+ chars]

    Usage:
    - Automatically called in main.py when running with uvicorn
    - For production deployment with gunicorn, ensure this is called in startup
    """
    # Get the uvicorn access logger
    access_logger = logging.getLogger("uvicorn.access")

    # Remove existing handlers to avoid duplicate logs
    for handler in access_logger.handlers[:]:
        access_logger.removeHandler(handler)

    # Create new handler with our custom formatter that includes timestamp and log level
    handler = logging.StreamHandler(sys.stdout)
    access_formatter = AccessLogFormatter("%(asctime)s | %(levelname)-8s | %(message)s")
    handler.setFormatter(access_formatter)

    # Add the handler to uvicorn access logger
    access_logger.addHandler(handler)
    access_logger.setLevel(logging.INFO)
    access_logger.propagate = False

    return access_logger
