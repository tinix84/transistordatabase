"""Playwright configuration for browser testing."""
from playwright.sync_api import Playwright


def pytest_configure(config):
    """Configure Playwright for pytest."""
    config.addinivalue_line("markers", "e2e: mark test as end-to-end browser test")
    config.addinivalue_line("markers", "slow: mark test as slow running")


# Playwright fixtures will be automatically provided by pytest-playwright
# Default configuration is sufficient for most cases:
# - Browser: chromium
# - Headless: True (in CI), False (when running locally with --headed)
# - Base URL: can be set via --base-url flag

# Example usage in tests:
# def test_example(page):
#     page.goto("http://localhost:8000")
#     page.click("text=Click me")
#     assert page.inner_text("h1") == "Welcome"
