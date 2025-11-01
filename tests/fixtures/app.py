import pytest
import uvicorn
import threading
import time

from fastapi.testclient import TestClient

from app.core.application import create_app
from app.dependencies import get_error_log_service
from app.router import (
    gemini_routes,
    openai_routes,
    vertex_express_routes,
    openai_compatible_routes,
    key_routes,
)

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
def live_server_url(test_app, db_engine):
    """
    Session-scoped fixture to start and stop a live FastAPI server.
    Depends on db_engine to ensure the database is set up.
    """

    config = uvicorn.Config(
        test_app, host=TEST_SERVER_HOST, port=TEST_SERVER_PORT, log_level="info"
    )
    server = TestServer(config=config)
    server.run_in_thread()

    base_url = f"http://{TEST_SERVER_HOST}:{TEST_SERVER_PORT}"
    yield base_url

    server.stop()


@pytest.fixture(scope="session")
def base_url(live_server_url):
    """
    Provides the live server's base URL to the pytest-base-url plugin,
    so it can be used automatically by page objects.
    """
    return live_server_url


@pytest.fixture(scope="session")
def test_app(
    mock_key_manager,
    mock_error_log_service,
    db_engine,
    mock_chat_service,
    mock_embedding_service,
):
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

    async def override_get_error_log_service_dep():
        return mock_error_log_service

    app.dependency_overrides[
        get_error_log_service
    ] = override_get_error_log_service_dep

    async def override_get_chat_service():
        return mock_chat_service

    app.dependency_overrides[gemini_routes.get_chat_service] = override_get_chat_service

    async def override_get_embedding_service():
        return mock_embedding_service

    app.dependency_overrides[
        gemini_routes.get_embedding_service
    ] = override_get_embedding_service

    async def mock_security_dependency():
        pass

    app.dependency_overrides[
        gemini_routes.security_service.verify_key_or_goog_api_key
    ] = mock_security_dependency

    app.dependency_overrides[
        openai_compatible_routes.security_service.verify_authorization
    ] = mock_security_dependency

    yield app


@pytest.fixture(scope="session")
def client(test_app):
    """Fixture to create a TestClient with dependencies overridden."""
    with TestClient(test_app) as test_client:
        yield test_client
