import pytest
from unittest.mock import MagicMock, AsyncMock

from app.service.key.key_manager import KeyManager

@pytest.fixture
def mock_key_manager():
    """Fixture to create a mock KeyManager. Defaults to function scope for test isolation."""
    mock = MagicMock(spec=KeyManager)
    mock.get_random_valid_key = AsyncMock(return_value="test_api_key")
    mock.get_next_working_key = AsyncMock(return_value="test_api_key_for_model")
    mock.get_paid_key = AsyncMock(return_value="test_paid_api_key")
    return mock


@pytest.fixture
def mock_error_log_service():
    """Fixture to create a mock error_log_service module. Defaults to function scope for test isolation."""
    mock = AsyncMock()
    mock.process_get_error_logs.return_value = {"logs": [], "total": 0}
    mock.process_get_error_log_details.return_value = {}
    mock.process_find_error_log_by_info.return_value = {}
    mock.process_delete_error_logs_by_ids.return_value = 1
    mock.process_delete_all_error_logs.return_value = None
    mock.process_delete_error_log_by_id.return_value = True
    return mock


@pytest.fixture
def mock_chat_service():
    """Fixture to create a mock chat service. Defaults to function scope for test isolation."""
    mock = MagicMock()
    mock.generate_content = AsyncMock(return_value={"candidates": [{"content": {"parts": [{"text": "Hello, world!"}]}}]})
    mock.count_tokens = AsyncMock(return_value={"totalTokens": 123})
    return mock


@pytest.fixture
def mock_embedding_service():
    """Fixture to create a mock embedding service. Defaults to function scope for test isolation."""
    mock = MagicMock()
    mock.embed_content = AsyncMock(return_value={"embedding": {"values": [0.1, 0.2, 0.3]}})
    return mock
