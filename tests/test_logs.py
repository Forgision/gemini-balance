from playwright.sync_api import Page, expect

def test_logs_page_loads(page: Page, base_url: str):
    """
    Tests that the /logs page loads correctly.
    """
    # Navigate to the login page and log in
    page.goto(f"{base_url}/")
    page.locator("#auth-token").fill("test-token")
    page.get_by_text("Login").click()

    # Navigate to the /logs page
    page.goto(f"{base_url}/logs")

    # Verify that the main heading is visible
    expect(page.get_by_text("Gemini Balance - Error Logs")).to_be_visible()
