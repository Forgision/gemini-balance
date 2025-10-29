from dotenv import load_dotenv
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock

# Load test environment variables before any application code is imported
load_dotenv(dotenv_path="tests/.env.test")

from app.core.application import create_app
from app.service.key.key_manager import KeyManager
from app.router import (
    gemini_routes,
    openai_routes,
    vertex_express_routes,
    openai_compatible_routes,
    key_routes,
)

@pytest.fixture
def mock_key_manager():
    """Fixture to create a mock KeyManager."""
    mock = MagicMock(spec=KeyManager)
    mock.get_random_valid_key = AsyncMock(return_value="test_api_key")
    mock.get_next_working_key = AsyncMock(return_value="test_api_key_for_model")
    return mock

@pytest.fixture
def client(mock_key_manager):
    """Fixture to create a TestClient with dependencies overridden."""
    app = create_app()

    async def override_get_key_manager():
        return mock_key_manager

    app.dependency_overrides[gemini_routes.get_key_manager] = override_get_key_manager
    app.dependency_overrides[openai_routes.get_key_manager] = override_get_key_manager
    app.dependency_overrides[
        vertex_express_routes.get_key_manager
    ] = override_get_key_manager
    app.dependency_overrides[
        openai_compatible_routes.get_key_manager
    ] = override_get_key_manager
    app.dependency_overrides[key_routes.get_key_manager] = override_get_key_manager

    with TestClient(app) as test_client:
        yield test_client
