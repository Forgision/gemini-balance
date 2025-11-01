import pytest
from unittest.mock import MagicMock, AsyncMock

from app.service.key.key_manager import KeyManager


@pytest.fixture(scope="session")
def mock_key_manager():
    """Fixture to create a mock KeyManager."""
    mock = MagicMock(spec=KeyManager)
    mock.get_random_valid_key = AsyncMock(return_value="test_api_key")
    mock.get_next_working_key = AsyncMock(return_value="test_api_key_for_model")
    mock.get_paid_key = AsyncMock(return_value="test_paid_api_key")
    return mock


@pytest.fixture(scope="session")
def mock_error_log_service():
    """Fixture to create a mock error_log_service module."""
    mock = AsyncMock()
    mock.process_get_error_logs.return_value = {"logs": [], "total": 0}
    mock.process_get_error_log_details.return_value = {}
    mock.process_find_error_log_by_info.return_value = {}
    mock.process_delete_error_logs_by_ids.return_value = 1
    mock.process_delete_all_error_logs.return_value = None
    mock.process_delete_error_log_by_id.return_value = True
    return mock


@pytest.fixture(scope="session")
def mock_chat_service():
    """Fixture to create a mock chat service."""
    mock = MagicMock()
    mock.generate_content = AsyncMock(return_value={"candidates": [{"content": {"parts": [{"text": "Hello, world!"}]}}]})
    mock.count_tokens = AsyncMock(return_value={"totalTokens": 123})
    return mock


@pytest.fixture(scope="session")
def mock_embedding_service():
    """Fixture to create a mock embedding service."""
    mock = MagicMock()
    mock.embed_content = AsyncMock(return_value={"embedding": {"values": [0.1, 0.2, 0.3]}})
    return mock


@pytest.fixture(autouse=True)
def reset_mock_key_manager(mock_key_manager):
    """
    Automatically reset the mock's call counts and return values before each test.
    This prevents state from leaking between tests.
    """
    mock_key_manager.get_random_valid_key.reset_mock()
    mock_key_manager.get_random_valid_key.return_value = "test_api_key"
    mock_key_manager.get_next_working_key.reset_mock()
    mock_key_manager.get_next_working_key.return_value = "test_api_key_for_model"
    mock_key_manager.get_paid_key.reset_mock()
    mock_key_manager.get_paid_key.return_value = "test_paid_api_key"


@pytest.fixture(autouse=True)
def reset_mock_chat_service(mock_chat_service):
    """Automatically reset the mock chat service before each test."""
    mock_chat_service.reset_mock()
    mock_chat_service.generate_content.return_value = {"candidates": [{"content": {"parts": [{"text": "Hello, world!"}]}}]}
    mock_chat_service.count_tokens.return_value = {"totalTokens": 123}


@pytest.fixture(autouse=True)
def reset_mock_embedding_service(mock_embedding_service):
    """Automatically reset the mock embedding service before each test."""
    mock_embedding_service.reset_mock()
    mock_embedding_service.embed_content.return_value = {"embedding": {"values": [0.1, 0.2, 0.3]}}
