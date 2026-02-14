# Playwright Testing Guide

End-to-end browser testing for the Transistor Database web GUI using Playwright.

## Installation

### 1. Install Playwright and pytest plugin

```bash
pip install playwright pytest-playwright
```

### 2. Install browsers

```bash
playwright install chromium
# Or install all browsers:
playwright install
```

For system-wide installation on Ubuntu/Debian:
```bash
sudo apt install python3-playwright
playwright install
```

## Running Tests

### Basic Usage

```bash
# Run all Playwright tests (headless mode)
pytest tests/test_gui_playwright.py -v

# Run with visible browser (helpful for debugging)
pytest tests/test_gui_playwright.py -v --headed

# Run with slow motion (easier to see what's happening)
pytest tests/test_gui_playwright.py -v --headed --slowmo 500

# Run only e2e marked tests
pytest tests/test_gui_playwright.py -v -m e2e

# Skip slow tests
pytest tests/test_gui_playwright.py -v -m "not slow"
```

### Advanced Options

```bash
# Run in different browsers
pytest tests/test_gui_playwright.py --browser chromium
pytest tests/test_gui_playwright.py --browser firefox
pytest tests/test_gui_playwright.py --browser webkit

# Run with custom base URL
pytest tests/test_gui_playwright.py --base-url http://localhost:3000

# Generate screenshots on failure
pytest tests/test_gui_playwright.py --screenshot on

# Generate video recordings
pytest tests/test_gui_playwright.py --video on

# Run in parallel (requires pytest-xdist)
pytest tests/test_gui_playwright.py -n 4
```

## Test Structure

### Test Classes

1. **TestWebGUIBasics**: Basic functionality (homepage, API root)
2. **TestTransistorOperations**: CRUD operations via API
3. **TestAPIEndpoints**: Validation, export endpoints
4. **TestPlotEndpoints**: Plot data endpoints
5. **TestFullWebInterface**: Vue.js frontend tests (requires frontend running)

### Test Markers

- `@pytest.mark.e2e`: End-to-end browser tests
- `@pytest.mark.slow`: Slow-running tests
- `@pytest.mark.skip`: Skipped tests (e.g., require frontend setup)

## Starting the Web Server

The tests automatically start the FastAPI backend. For manual testing:

```bash
# Start backend (from project root)
uvicorn transistordatabase.gui_web.backend.main:app --reload

# Or with specific port
uvicorn transistordatabase.gui_web.backend.main:app --port 8000
```

For the Vue.js frontend (optional):
```bash
cd transistordatabase/gui_web/frontend
npm install
npm run dev
```

## Writing New Tests

### Basic Test Structure

```python
@pytest.mark.e2e
def test_my_feature(page: Page, base_url: str, server_process):
    """Test description."""
    # Navigate to page
    page.goto(f"{base_url}/my-endpoint")

    # Interact with elements
    page.click("text=Button")
    page.fill("input[name='field']", "value")

    # Assertions
    assert page.inner_text("h1") == "Expected Text"
    expect(page.locator(".result")).to_be_visible()
```

### Common Patterns

```python
# Navigate
page.goto("http://localhost:8000")

# Click elements
page.click("button#submit")
page.click("text=Click me")

# Fill forms
page.fill("input[name='email']", "test@example.com")
page.select_option("select#country", "US")

# Wait for elements
page.wait_for_selector(".result", timeout=5000)
page.wait_for_load_state("networkidle")

# Get text
text = page.inner_text("h1")
content = page.content()

# Assertions (new Playwright API)
from playwright.sync_api import expect
expect(page.locator("h1")).to_have_text("Welcome")
expect(page.locator(".error")).to_be_visible()
```

## Debugging

### Visual Debugging

```bash
# Run with browser visible and slow motion
pytest tests/test_gui_playwright.py -v --headed --slowmo 1000

# Open Playwright Inspector (step through test)
PWDEBUG=1 pytest tests/test_gui_playwright.py
```

### Capture Artifacts

```bash
# Save screenshots
pytest tests/test_gui_playwright.py --screenshot only-on-failure

# Save videos
pytest tests/test_gui_playwright.py --video retain-on-failure

# Save traces (detailed debugging info)
pytest tests/test_gui_playwright.py --tracing on
```

Artifacts are saved to `test-results/` directory by default.

## CI/CD Integration

### GitHub Actions Example

```yaml
name: E2E Tests
on: [push, pull_request]

jobs:
  e2e:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          playwright install --with-deps chromium

      - name: Run E2E tests
        run: |
          pytest tests/test_gui_playwright.py -v --screenshot on --video on

      - name: Upload artifacts
        if: failure()
        uses: actions/upload-artifact@v3
        with:
          name: test-results
          path: test-results/
```

## Troubleshooting

### Browser Not Found

```bash
# Reinstall browsers
playwright install chromium
```

### Port Already in Use

If the server fails to start on port 8000, another process may be using it:

```bash
# Find process using port 8000
lsof -i :8000
# Or
netstat -tuln | grep 8000

# Kill the process
kill -9 <PID>
```

### Tests Timing Out

Increase timeout in tests:
```python
page.wait_for_selector(".result", timeout=10000)  # 10 seconds
```

Or set global timeout in `playwright.config.py`:
```python
def pytest_configure(config):
    config.option.timeout = 30000  # 30 seconds
```

### Tests Failing in Headless Mode

Some tests may behave differently in headless vs. headed mode. Run with `--headed` to debug:
```bash
pytest tests/test_gui_playwright.py::test_name -v --headed
```

## Best Practices

1. **Use data-testid attributes**: Add `data-testid` to HTML elements for stable selectors
   ```html
   <button data-testid="submit-button">Submit</button>
   ```
   ```python
   page.click("[data-testid='submit-button']")
   ```

2. **Wait for network idle**: After navigation or form submission
   ```python
   page.goto(url)
   page.wait_for_load_state("networkidle")
   ```

3. **Isolate tests**: Each test should be independent and not rely on state from other tests

4. **Use fixtures**: Share common setup (base_url, server_process) via fixtures

5. **Mark slow tests**: Use `@pytest.mark.slow` for tests that take >5 seconds

## Resources

- [Playwright Python Docs](https://playwright.dev/python/)
- [pytest-playwright Plugin](https://github.com/microsoft/playwright-pytest)
- [Playwright Best Practices](https://playwright.dev/python/docs/best-practices)
- [Selector Strategies](https://playwright.dev/python/docs/selectors)

## Examples

See `tests/test_gui_playwright.py` for complete examples of:
- API endpoint testing
- Form interaction
- Navigation and assertions
- Plot data validation
- Full Vue.js integration tests
