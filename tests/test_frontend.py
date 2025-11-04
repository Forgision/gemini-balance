# This file for testing frontend pages

from app.config.config import settings
# from playwright.sync_api import Page, expect
import os
import asyncio
import pytest
from pydoll.browser import Chrome
from pydoll.browser.options import ChromiumOptions
from pydoll.constants import Key


# @pytest.fixture(scope="session")
# async def page():
#     options = ChromiumOptions()
#     options.binary_location = "/usr/local/bin/wrapped-chromium"
#     options.headless = False
    
#     async with Chrome(options=options) as browser:
#         tab = await browser.start()
#         yield tab
#         browser.close()



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

