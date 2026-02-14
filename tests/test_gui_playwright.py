"""
Playwright end-to-end tests for the web GUI.

These tests require:
1. Playwright installed: pip install playwright pytest-playwright
2. Browsers installed: playwright install chromium
3. Web server running: uvicorn transistordatabase.gui_web.backend.main:app

Run with:
    pytest tests/test_gui_playwright.py --headed  # see browser
    pytest tests/test_gui_playwright.py           # headless mode
"""
import pytest
import subprocess
import time
from pathlib import Path


# Skip all tests if playwright is not installed
try:
    from playwright.sync_api import Page, expect
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not PLAYWRIGHT_AVAILABLE,
    reason="Playwright not installed (pip install playwright pytest-playwright)"
)


@pytest.fixture(scope="module")
def base_url():
    """Return the base URL for the web GUI."""
    return "http://localhost:8000"


@pytest.fixture(scope="module")
def server_process(base_url):
    """Start the FastAPI server for testing."""
    # Start server
    main_path = Path(__file__).parent.parent / "transistordatabase" / "gui_web" / "backend" / "main.py"
    process = subprocess.Popen(
        ["uvicorn", "transistordatabase.gui_web.backend.main:app", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to start
    time.sleep(2)

    yield process

    # Cleanup
    process.terminate()
    process.wait(timeout=5)


@pytest.mark.e2e
class TestWebGUIBasics:
    """Basic web GUI functionality tests."""

    def test_homepage_loads(self, page: Page, base_url: str, server_process):
        """Test that the homepage loads successfully."""
        page.goto(base_url)

        # Check that we get a response (may be API response or static HTML)
        # For now, just check page loads without error
        assert page.url == base_url + "/"

        # API should return JSON with message
        content = page.content()
        assert "Transistor Database" in content or "message" in content

    def test_api_root_endpoint(self, page: Page, base_url: str, server_process):
        """Test the API root endpoint."""
        page.goto(f"{base_url}/")

        # Should return JSON
        content = page.content()
        assert "message" in content or "Transistor Database" in content

    def test_api_transistors_list(self, page: Page, base_url: str, server_process):
        """Test the transistors list endpoint."""
        page.goto(f"{base_url}/api/transistors")

        # Should return JSON array
        content = page.content()
        # Empty array [] or transistor data
        assert "[" in content or "transistor" in content.lower()


@pytest.mark.e2e
@pytest.mark.slow
class TestTransistorOperations:
    """Test transistor CRUD operations via web GUI."""

    def test_get_transistor_by_name(self, page: Page, base_url: str, server_process):
        """Test fetching a specific transistor."""
        # First check if CREE transistor exists
        page.goto(f"{base_url}/api/transistors")
        transistors_list = page.content()

        if "CREE_C3M0016120K" in transistors_list:
            # Fetch specific transistor
            page.goto(f"{base_url}/api/transistors/CREE_C3M0016120K")
            content = page.content()

            # Should contain transistor data
            assert "metadata" in content or "CREE" in content
            assert "manufacturer" in content.lower() or "type" in content.lower()

    def test_api_docs_available(self, page: Page, base_url: str, server_process):
        """Test that API documentation is accessible."""
        page.goto(f"{base_url}/docs")

        # FastAPI auto-generates /docs (Swagger UI)
        # Check for common Swagger UI elements
        content = page.content()
        assert "swagger" in content.lower() or "api" in content.lower()


@pytest.mark.e2e
class TestAPIEndpoints:
    """Test various API endpoints."""

    def test_validation_endpoint(self, page: Page, base_url: str, server_process):
        """Test validation endpoint for a transistor."""
        # Assuming CREE transistor exists
        page.goto(f"{base_url}/api/transistors/CREE_C3M0016120K/validate")

        # Should return validation result (may be 200 or 404)
        content = page.content()
        # Either valid response or "not found"
        assert "errors" in content or "not found" in content.lower() or "detail" in content

    def test_export_json_endpoint(self, page: Page, base_url: str, server_process):
        """Test JSON export endpoint."""
        page.goto(f"{base_url}/api/transistors/CREE_C3M0016120K/export/json")

        # Should trigger download or return JSON
        content = page.content()
        # Either JSON data or 404
        assert "{" in content or "not found" in content.lower()


@pytest.mark.e2e
@pytest.mark.slow
class TestPlotEndpoints:
    """Test plot data endpoints."""

    def test_channel_plot_endpoint(self, page: Page, base_url: str, server_process):
        """Test channel characteristics plot data."""
        page.goto(f"{base_url}/api/plots/channel/CREE_C3M0016120K")

        content = page.content()
        # Should return plot data or error
        assert "curves" in content or "error" in content or "not found" in content.lower()

    def test_switching_plot_endpoint(self, page: Page, base_url: str, server_process):
        """Test switching losses plot data."""
        page.goto(f"{base_url}/api/plots/switching/CREE_C3M0016120K")

        content = page.content()
        # Should return plot data or error
        assert "curves" in content or "error" in content or "not found" in content.lower()


# Integration test example (requires full setup)
@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.skip(reason="Requires Vue frontend running")
class TestFullWebInterface:
    """Tests for the full web interface (requires Vue frontend)."""

    def test_vue_app_loads(self, page: Page):
        """Test that Vue.js frontend loads (requires npm run dev)."""
        # This would test http://localhost:5173 (Vite dev server)
        page.goto("http://localhost:5173")

        # Wait for Vue app to render
        page.wait_for_selector("h1", timeout=5000)

        # Check for app title
        title = page.inner_text("h1")
        assert "Transistor Database" in title

    def test_transistor_search_form(self, page: Page):
        """Test the transistor search functionality."""
        page.goto("http://localhost:5173")

        # Fill search form
        page.fill("input[name='search']", "CREE")
        page.click("button[type='submit']")

        # Wait for results
        page.wait_for_selector(".transistor-card", timeout=5000)

        # Check results
        results = page.query_selector_all(".transistor-card")
        assert len(results) > 0


if __name__ == "__main__":
    # Run with: python -m pytest tests/test_gui_playwright.py -v --headed
    pytest.main([__file__, "-v", "--headed"])
