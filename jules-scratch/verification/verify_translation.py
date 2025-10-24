
from playwright.sync_api import sync_playwright
import time

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        time.sleep(5) # Add a 5-second delay
        page.goto("http://localhost:8000/")
        page.fill("input[name='auth_token']", "sk-123456")
        page.click("button[type='submit']")
        page.screenshot(path="jules-scratch/verification/auth_attempt.png")
        page.wait_for_url("http://localhost:8000/keys")
        page.goto("http://localhost:8000/config")
        page.screenshot(path="jules-scratch/verification/verification.png")
        browser.close()

if __name__ == "__main__":
    run()
