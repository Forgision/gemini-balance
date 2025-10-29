from playwright.sync_api import Page, expect

def test_keys_page_renders_keys(page: Page, base_url: str):
    """
    Tests that the /keys page loads, expands the valid keys section,
    and correctly renders the number of keys specified in the .env.test file.
    """
    # Navigate to the login page and log in
    page.goto(f"{base_url}/")
    page.locator("#auth-token").fill("test-token")
    page.get_by_text("Login").click()

    # Wait for the /keys page to load
    expect(page).to_have_url(f"{base_url}/keys")
    expect(page.get_by_text("Gemini Balance - Monitoring Panel")).to_be_visible()

    # Expand the "Valid Keys List" section
    # Use a specific locator targeting the onclick attribute to avoid ambiguity
    valid_keys_header = page.locator(".stats-card-header[onclick*=\"'validKeys'\"]")
    valid_keys_header.click()

    # Wait for the "Loading..." message to disappear.
    expect(page.locator("#validKeys li", has_text="Loading...")).to_be_hidden(timeout=10000)

    # Assert that exactly one key item is rendered in the list.
    # This corresponds to the single key in the .env.test file.
    expect(page.locator("#validKeys li")).to_have_count(1)

    # Additionally, check for some text within the rendered key to ensure it's not an empty item
    first_key_item = page.locator("#validKeys li").first
    expect(first_key_item).to_contain_text("Failures:")
    expect(first_key_item.get_by_text("Verify")).to_be_visible()
