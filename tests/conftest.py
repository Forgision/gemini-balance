import os
import threading
import time
from pathlib import Path
import sys
import pytest

import uvicorn
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock
from unittest.mock import patch

from app.core.application import create_app
from app.dependencies import get_error_log_service
from app.service.key.key_manager import KeyManager
from app.router import (
    gemini_routes,
    openai_routes,
    vertex_express_routes,
    openai_compatible_routes,
    key_routes,
)
from app.main import app
from app.config.config import settings

# Add the project root to the Python path to ensure app modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load test environment variables before importing any application modules
# This is crucial to ensure the app is configured for testing
load_dotenv(dotenv_path=PROJECT_ROOT / ".env.test", override=True)

# Now, it's safe to import the application instance

# Define a fixed port and host for the test server
TEST_SERVER_PORT = 8002
TEST_SERVER_HOST = "127.0.0.1"


class TestServer(uvicorn.Server):
    """
    Custom uvicorn server to run in a background thread.
    Disables signal handlers to allow it to run in a non-main thread.
    """

    def install_signal_handlers(self):
        pass

    def run_in_thread(self):
        """Starts the server in a daemon thread."""
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        # Wait for the server to start up
        while not self.started:
            time.sleep(1e-3)

    def stop(self):
        """Stops the server and waits for the thread to terminate."""
        self.should_exit = True
        self.thread.join()


@pytest.fixture(scope="session")
def live_server_url():
    """
    Session-scoped fixture to start and stop a live FastAPI server.
    """

    config = uvicorn.Config(
        app, host=TEST_SERVER_HOST, port=TEST_SERVER_PORT, log_level="info"
    )
    server = TestServer(config=config)
    server.run_in_thread()

    base_url = f"http://{TEST_SERVER_HOST}:{TEST_SERVER_PORT}"
    yield base_url

    server.stop()

    # Cleanup: remove the test database file after the test session
    db_file = Path(settings.SQLITE_DATABASE)
    if db_file.exists():
        try:
            os.remove(db_file)
        except OSError as e:
            print(f"Error removing test database file {db_file}: {e}")


@pytest.fixture(scope="session")
def base_url(live_server_url):
    """
    Provides the live server's base URL to the pytest-base-url plugin,
    so it can be used automatically by page objects.
    """
    return live_server_url


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
def test_app(mock_key_manager, mock_error_log_service):
    app = create_app()

    async def override_get_key_manager():
        return mock_key_manager

    app.dependency_overrides[gemini_routes.get_key_manager] = (
        override_get_key_manager
    )
    app.dependency_overrides[openai_routes.get_key_manager] = (
        override_get_key_manager
    )
    app.dependency_overrides[vertex_express_routes.get_key_manager] = (
        override_get_key_manager
    )
    app.dependency_overrides[openai_compatible_routes.get_key_manager] = (
        override_get_key_manager
    )
    app.dependency_overrides[key_routes.get_key_manager] = override_get_key_manager

    async def override_get_error_log_service_dep():
        return mock_error_log_service
    app.dependency_overrides[get_error_log_service] = override_get_error_log_service_dep

    async def override_security():
        return "test_token"
        app.dependency_overrides[gemini_routes.security_service.verify_key_or_goog_api_key] = override_security
    app.dependency_overrides[
        openai_compatible_routes.security_service.verify_authorization
    ] = override_security

    yield app


@pytest.fixture(autouse=True)
def mock_verify_auth_token(request, mocker):
    if "no_mock_auth" in request.keywords:
        return

    mocker.patch("app.core.security.verify_auth_token", return_value=True)
    mocker.patch("app.middleware.middleware.verify_auth_token", return_value=True)
    mocker.patch("app.router.config_routes.verify_auth_token", return_value=True)
    mocker.patch("app.router.error_log_routes.verify_auth_token", return_value=True)
    mocker.patch("app.router.key_routes.verify_auth_token", return_value=True)
    mocker.patch("app.router.routes.verify_auth_token", return_value=True)
    mocker.patch("app.router.scheduler_routes.verify_auth_token", return_value=True)
    mocker.patch("app.router.stats_routes.verify_auth_token", return_value=True)


@pytest.fixture(scope="session")
def client(test_app):
    """Fixture to create a TestClient with dependencies overridden."""
    with TestClient(test_app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def reset_mock_key_manager(mock_key_manager):
    """
    Automatically reset the mock's call counts before each test,
    without clearing the configured return_values.
    """
    mock_key_manager.get_random_valid_key.await_count = 0
    mock_key_manager.get_random_valid_key.call_count = 0
    mock_key_manager.get_next_working_key.await_count = 0
    mock_key_manager.get_next_working_key.call_count = 0
    mock_key_manager.get_paid_key.await_count = 0
    mock_key_manager.get_paid_key.call_count = 0



# @pytest.fixture()
# def mock_verify_auth_token():
#     """Fixture to patch verify_auth_token.
#     By default, it allows authorization. Can be overridden for unauthorized tests.
#     """
#     with patch("app.core.security.verify_auth_token") as mock:
#         mock.return_value = True  # Default to authorized
#         yield mock


