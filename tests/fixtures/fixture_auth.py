from unittest.mock import AsyncMock, patch
import pytest

from tests.fixtures.fixture_consts import TEST_AUTH_TOKEN

@pytest.fixture(scope="session")
def auth_token():
    return TEST_AUTH_TOKEN


@pytest.fixture(scope="session")
def goog_api_key_header(auth_token):
    return {"x-goog-api-key": auth_token}


@pytest.fixture(scope="session", autouse=True)
def mock_env_import():
    """
    Prevent importing env vars to DB during tests.
    """
    with patch("app.core.application.sync_initial_settings", new=AsyncMock()) as mock:
        yield mock
