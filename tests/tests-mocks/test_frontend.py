# This file for testing frontend pages

from app.config.config import settings
# from playwright.sync_api import Page, expect
import platform
import pytest
from pydoll.browser import Chrome
from pydoll.browser.options import ChromiumOptions
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
    claude_routes,
)

TEST_SERVER_PORT = 8002
TEST_SERVER_HOST = "127.0.0.1"


class LiveServer(uvicorn.Server):
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


@pytest.fixture(scope="function")
def live_server_url(test_app):
    """
    Function-scoped fixture to start and stop a live FastAPI server.
    Only created when explicitly requested by tests.
    Note: This fixture does NOT set up a database.
    For tests that need a database, use route_test_app from tests/routes/conftest.py
    """
    import random
    # Use a random port to avoid conflicts
    port = random.randint(8100, 8999)
    
    config = uvicorn.Config(
        test_app, host=TEST_SERVER_HOST, port=port, log_level="error"
    )
    server = LiveServer(config=config)
    server.run_in_thread()

    base_url = f"http://{TEST_SERVER_HOST}:{port}"
    yield base_url

    server.stop()


@pytest.fixture(scope="session")
def base_url():
    """
    Provides a default base URL for the pytest-base-url plugin.
    Tests that need a live server should use live_server_url directly.
    """
    return "http://localhost:8000"


@pytest.fixture(scope="function")
def test_app(
    mock_key_manager,
    mock_error_log_service,
    mock_chat_service,
    mock_embedding_service,
):
    """
    Function-scoped fixture to create a test app for non-route tests.
    Note: Route tests should use route_test_app from tests/routes/conftest.py
    This fixture does NOT set up a database - use route_test_app if you need DB.
    """
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

    async def override_claude_proxy_service():
        return mock_chat_service

    app.dependency_overrides[claude_routes.ClaudeProxyService] = override_claude_proxy_service
    app.dependency_overrides[claude_routes.security_service.verify_auth_token] = mock_security_dependency

    yield app

    # Clean up dependency overrides after each test
    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
def client(test_app):
    """Fixture to create a TestClient with dependencies overridden."""
    with TestClient(test_app) as test_client:
        yield test_client


@pytest.mark.asyncio
async def test_front_end(live_server_url):
    """
    Tests the authentication flow by:
    1. Navigating to the login page.
    2. Entering the test token.
    3. Clicking the login button.
    4. Verifying redirection to the keys status page.
    """
    # base_url = "http://localhost:8000"
    base_url = live_server_url
    AUTH_TOKEN = settings.AUTH_TOKEN
    
    options = ChromiumOptions()
    if platform.system() == "Linux":
        options.binary_location = "/usr/local/bin/wrapped-chromium"
    options.headless = False
    
    async with Chrome(options=options) as browser:
        tab = await browser.start()
        res = await tab.go_to(f"{base_url}/")
        # tab.
        token_input = await tab.find(id='auth-token')
        
        assert token_input is not None
        assert token_input.is_enabled
        assert token_input.is_visible()
        
        await token_input.type_text(AUTH_TOKEN)
        login_button = await tab.find(text='Login')
        await login_button.click()
        # page.find(text='Login').click()
        print(res)
        #TODO: implement rest of playright to pydoll

    # expect(page.locator("#auth-token")).to_be_visible()
    # # Enter the test token
    # page.locator("#auth-token").fill(settings.AUTH_TOKEN)

    # # Click the login button
    # page.get_by_text("Login").click()

    # # Verify redirection to the keys status page
    # expect(page).to_have_url(f"{base_url}/keys")
    # expect(page.get_by_text("Gemini Balance - Monitoring Panel")).to_be_visible()
    
    
    # page.locator('xpath=//html/body/div[1]/div/div[2]/a[1]').click()
    # expect(page).to_have_url(f"{base_url}/config")
    # #TODO: Api keys on page match with valid settings
    
    # page.locator('xpath=//html/body/div[1]/div/div[1]/a[3]').click()
    # expect(page).to_have_url(f"{base_url}/logs")

