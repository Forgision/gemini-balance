from playwright.sync_api import sync_playwright

def run(playwright):
    browser = playwright.chromium.launch()
    page = browser.new_page()
    page.goto("http://127.0.0.1:8000/config")
    page.screenshot(path="jules-scratch/verification/config_page.png")
    browser.close()

with sync_playwright() as playwright:
    run(playwright)
