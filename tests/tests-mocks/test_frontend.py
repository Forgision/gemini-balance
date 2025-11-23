# This file for testing frontend pages


# from playwright.sync_api import Page, expect
import platform
import pytest
from pydoll.browser import Chrome
from pydoll.browser.options import ChromiumOptions
from pydoll.exceptions import ElementNotFound
import uvicorn
import threading
import time
import asyncio


from app.config.config import settings
from app.core.application import create_app

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
def live_server_url():
    """
    Function-scoped fixture to start and stop a live FastAPI server.
    Only created when explicitly requested by tests.
    Note: This fixture does NOT set up a database.
    For tests that need a database, use route_test_app from tests/routes/conftest.py
    """
    import random

    # Use a random port to avoid conflicts
    port = random.randint(8100, 8999)

    app = create_app()

    config = uvicorn.Config(app, host=TEST_SERVER_HOST, port=port, log_level="error")
    server = LiveServer(config=config)
    server.run_in_thread()

    base_url = f"http://{TEST_SERVER_HOST}:{port}"
    yield base_url

    server.stop()


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
        token_input = await tab.find(id="auth-token")

        assert token_input is not None
        assert token_input.is_enabled
        assert token_input.is_visible()

        await token_input.type_text(AUTH_TOKEN)
        login_button = await tab.find(text="Login")
        await login_button.click()
        # page.find(text='Login').click()

        # await tab.wait_for_url(f"{base_url}/keys")
        start = time.time()
        while True:
            try:
                await tab.find(tag_name="a", class_name="nav-link", href="/config")
                break
            except ElementNotFound:
                if time.time() - start > 10:
                    raise TimeoutError("Element not found")
                await asyncio.sleep(1)

        await asyncio.sleep(5)
        print(res)
        # TODO: implement rest of playright to pydoll

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
