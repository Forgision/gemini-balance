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
        app,
        host=TEST_SERVER_HOST,
        port=TEST_SERVER_PORT,
        log_level="info"
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


@pytest.fixture(autouse=True)
def mock_verify_auth_token():
    """Fixture to patch verify_auth_token to always return True."""
    with patch("app.core.security.verify_auth_token", return_value=True) as mock:
        yield mock

