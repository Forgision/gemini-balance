from playwright.sync_api import Page, expect

def test_auth_flow(page: Page, base_url: str):
    """
    Tests the authentication flow by:
    1. Navigating to the login page.
    2. Entering the test token.
    3. Clicking the login button.
    4. Verifying redirection to the keys status page.
    """
    # Navigate to the login page
    page.goto(f"{base_url}/")

    # Enter the test token
    page.locator("#auth-token").fill("test-token")

    # Click the login button
    page.get_by_text("Login").click()

    # Verify redirection to the keys status page
    expect(page).to_have_url(f"{base_url}/keys")
    expect(page.get_by_text("Gemini Balance - Monitoring Panel")).to_be_visible()
