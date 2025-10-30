# This file for testing frontend pages

from app.config.config import settings
from playwright.sync_api import Page, expect

def test_front_end(page: Page, base_url: str):
    """
    Tests the authentication flow by:
    1. Navigating to the login page.
    2. Entering the test token.
    3. Clicking the login button.
    4. Verifying redirection to the keys status page.
    """
    # Navigate to the login page
    page.goto(f"{base_url}/")

    expect(page.locator("#auth-token")).to_be_visible()
    # Enter the test token
    page.locator("#auth-token").fill(settings.AUTH_TOKEN)

    # Click the login button
    page.get_by_text("Login").click()

    # Verify redirection to the keys status page
    expect(page).to_have_url(f"{base_url}/keys")
    expect(page.get_by_text("Gemini Balance - Monitoring Panel")).to_be_visible()
    
    
    page.locator('xpath=//html/body/div[1]/div/div[2]/a[1]').click()
    expect(page).to_have_url(f"{base_url}/config")
    #TODO: Api keys on page match with valid settings
    
    page.locator('xpath=//html/body/div[1]/div/div[1]/a[3]').click()
    expect(page).to_have_url(f"{base_url}/logs")

