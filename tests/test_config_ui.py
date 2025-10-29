from playwright.sync_api import Page, expect

def test_config_page_loads(page: Page, base_url: str):
    """
    Tests that the /config page loads correctly.
    """
    # Navigate to the login page and log in
    page.goto(f"{base_url}/")
    page.locator("#auth-token").fill("test-token")
    page.get_by_text("Login").click()

    # Navigate to the /config page
    page.goto(f"{base_url}/config")

    # Verify that the main heading is visible
    expect(page.get_by_text("Gemini Balance - Configuration Edit")).to_be_visible()