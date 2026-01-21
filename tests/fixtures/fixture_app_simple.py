from unittest.mock import AsyncMock
import pytest
from fastapi.testclient import TestClient
from app.core.application import create_app
from app.dependencies import get_key_manager
from app.config.config import settings

from tests.fixtures.fixture_consts import TEST_AUTH_TOKEN

@pytest.fixture(scope="session")
def test_app(setup_test_db, mock_external_apis, mock_env_import):
    """
    Create a test app instance.
    """

    # Mock KeyManager dependency
    async def mock_get_key_manager():
        mock_km = AsyncMock()
        mock_km.is_ready = True
        mock_km.get_key.return_value = "mock_api_key"
        mock_km.get_random_valid_key.return_value = "mock_api_key"
        mock_km.handle_api_failure.return_value = "mock_api_key"
        mock_km.update_usage.return_value = True
        mock_km.reset_key_failure_count.return_value = True
        mock_km.get_state.return_value = {"models": {}, "summary": {}}
        mock_km.get_keys_by_status.return_value = {"valid_keys": {}, "invalid_keys": {}}
        return mock_km

    # Set auth token in settings for verification
    settings.AUTH_TOKEN = TEST_AUTH_TOKEN
    settings.ALLOWED_TOKENS = [TEST_AUTH_TOKEN]
    print(f"DEBUG CONFTEST: id(settings)={id(settings)}")
    print(f"DEBUG CONFTEST: settings.AUTH_TOKEN set to '{settings.AUTH_TOKEN}'")

    app = create_app()
    app.dependency_overrides[get_key_manager] = mock_get_key_manager

    print(
        f"DEBUG CONFTEST: after create_app, settings.AUTH_TOKEN='{settings.AUTH_TOKEN}'"
    )
    return app


@pytest.fixture(scope="session")
def test_client(test_app):
    """
    Create a test client.
    """
    with TestClient(test_app) as client:
        yield client
