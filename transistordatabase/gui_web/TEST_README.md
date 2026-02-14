# Frontend Test Suite Documentation

Comprehensive test suite for the Transistor Database Vue.js frontend application.

## Test Structure

```
src/tests/
├── setup.js              # Global test setup
├── fixtures/             # Mock data
│   └── transistors.js    # Mock transistor data
├── mocks/                # API mocks
│   └── api.js            # Axios mock adapter
├── unit/                 # Unit tests
│   ├── api.test.js       # API service tests
│   ├── App.test.js       # App component tests
│   └── SearchDatabase.test.js  # Search component tests
├── integration/          # Integration tests
└── e2e/                  # End-to-end tests
    ├── app.spec.js       # Full app E2E tests
    └── search.spec.js    # Search functionality E2E tests
```

## Test Categories

### 1. Unit Tests (Vitest + Vue Test Utils)

Fast, isolated tests for individual components and services.

**Coverage:**
- ✅ API Service (`api.test.js`) - 47 tests
  - All CRUD operations
  - Error handling
  - Network errors
  - Validation
  - Export/Import functionality

- ✅ App Component (`App.test.js`) - 30 tests
  - Component mounting
  - Data loading
  - Navigation
  - Theme toggle
  - Transistor operations
  - Inter-component communication

- ✅ SearchDatabase Component (`SearchDatabase.test.js`) - 35 tests
  - Filtering (name, type, manufacturer, voltage range)
  - Multiple filters
  - Filter reset
  - Pagination
  - Sorting
  - View modes
  - Event emissions

**Total Unit Tests: 112 tests**

### 2. Integration Tests

Tests for component interactions and data flow between components.

**To be added:**
- Component communication
- State management
- Route navigation
- Form submissions

### 3. End-to-End Tests (Playwright)

Full user workflow tests in real browser environments.

**Coverage:**
- ✅ App E2E (`app.spec.js`) - 25 tests
  - Application loading
  - Theme toggle
  - Navigation
  - Data loading
  - Responsive design
  - Error handling
  - Accessibility
  - Performance

- ✅ Search E2E (`search.spec.js`) - 15 tests
  - Search interface
  - Filtering functionality
  - Results display
  - Mobile usability

**Total E2E Tests: 40 tests**

## Running Tests

### Install Dependencies

```bash
cd transistordatabase/gui_web
npm install
```

### Run All Unit Tests

```bash
npm test
```

### Run Tests in Watch Mode

```bash
npm test -- --watch
```

### Run Tests with UI

```bash
npm run test:ui
```

### Generate Coverage Report

```bash
npm run test:coverage
```

Open `coverage/index.html` to view detailed coverage report.

### Run E2E Tests

```bash
# Install Playwright browsers (first time only)
npx playwright install

# Run E2E tests
npm run test:e2e

# Run E2E tests with UI
npm run test:e2e:ui

# Run specific browser
npx playwright test --project=chromium

# Run with headed browser (see the browser)
npx playwright test --headed

# Debug mode
npx playwright test --debug
```

## Test Coverage Goals

| Category | Target | Current |
|----------|--------|---------|
| Lines | 80% | TBD |
| Functions | 80% | TBD |
| Branches | 75% | TBD |
| Statements | 80% | TBD |

Run `npm run test:coverage` to see current coverage.

## Writing New Tests

### Unit Test Example

```javascript
import { describe, it, expect, beforeEach } from 'vitest'
import { mount } from '@vue/test-utils'
import MyComponent from '@/components/MyComponent.vue'

describe('MyComponent', () => {
  let wrapper

  beforeEach(() => {
    wrapper = mount(MyComponent, {
      props: {
        myProp: 'value'
      }
    })
  })

  it('should render correctly', () => {
    expect(wrapper.exists()).toBe(true)
  })

  it('should emit event on button click', async () => {
    await wrapper.find('button').trigger('click')
    expect(wrapper.emitted('my-event')).toBeTruthy()
  })
})
```

### E2E Test Example

```javascript
import { test, expect } from '@playwright/test'

test('should perform user action', async ({ page }) => {
  await page.goto('/')

  await page.locator('button', { hasText: 'Click Me' }).click()

  await expect(page.locator('.result')).toContainText('Success')
})
```

## Mocking

### Mock API Responses

```javascript
import { createMockApi } from '../mocks/api'
import axios from 'axios'

const mock = createMockApi(axios)
// API calls will now return mock data
```

### Mock Components

```javascript
vi.mock('@/components/MyComponent.vue', () => ({
  default: { name: 'MyComponent', template: '<div>Mocked</div>' }
}))
```

## Fixtures

Pre-defined mock data for testing:

```javascript
import { mockTransistors, mockTransistor1 } from '../fixtures/transistors'

// Use in tests
expect(wrapper.props('transistors')).toEqual(mockTransistors)
```

## Debugging Tests

### Debug Unit Tests

```bash
# Run specific test file
npm test -- api.test.js

# Run specific test
npm test -- -t "should fetch all transistors"

# Show console output
npm test -- --reporter=verbose
```

### Debug E2E Tests

```bash
# Run with headed browser
npx playwright test --headed

# Debug mode (step through)
npx playwright test --debug

# Specific test
npx playwright test app.spec.js:10

# Generate trace
npx playwright test --trace on
npx playwright show-trace trace.zip
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Frontend Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'

      - name: Install dependencies
        run: |
          cd transistordatabase/gui_web
          npm install

      - name: Run unit tests
        run: npm test -- --run

      - name: Generate coverage
        run: npm run test:coverage

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage/lcov.info

      - name: Install Playwright
        run: npx playwright install --with-deps

      - name: Run E2E tests
        run: npm run test:e2e

      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: playwright-report
          path: playwright-report/
```

## Common Issues

### Issue: Tests Timeout

**Solution:**
- Increase timeout in test file:
  ```javascript
  test('my test', { timeout: 60000 }, async ({ page }) => {
    // test code
  })
  ```
- Or in config:
  ```javascript
  // vitest.config.js
  export default defineConfig({
    test: {
      testTimeout: 30000
    }
  })
  ```

### Issue: Mock Not Working

**Solution:**
- Ensure mock is defined before importing component
- Use `vi.clearAllMocks()` in `beforeEach`
- Check mock path matches import path exactly

### Issue: Async State Not Updated

**Solution:**
```javascript
import { flushPromises } from '@vue/test-utils'

await flushPromises()  // Wait for all promises
await wrapper.vm.$nextTick()  // Wait for DOM update
```

### Issue: Playwright Browser Not Found

**Solution:**
```bash
npx playwright install chromium
# or install all browsers
npx playwright install
```

## Performance Optimization

### Fast Test Runs

1. **Run tests in parallel:**
   ```bash
   npm test -- --threads
   ```

2. **Use test sharding in CI:**
   ```bash
   npm test -- --shard=1/4
   npm test -- --shard=2/4
   # etc.
   ```

3. **Skip slow tests during development:**
   ```javascript
   test.skip('slow test', async ({ page }) => {
     // ...
   })
   ```

## Best Practices

1. **Test User Behavior, Not Implementation**
   - ✅ Good: `await page.click('button:has-text("Submit")')`
   - ❌ Bad: Testing internal state directly

2. **Use Data Test IDs for Stability**
   ```html
   <button data-testid="submit-button">Submit</button>
   ```
   ```javascript
   await page.locator('[data-testid="submit-button"]').click()
   ```

3. **Keep Tests Independent**
   - Each test should set up its own state
   - Don't rely on test execution order
   - Clean up after each test

4. **Use Descriptive Test Names**
   - ✅ Good: `should display error when API fails`
   - ❌ Bad: `test error`

5. **Mock External Dependencies**
   - API calls
   - Third-party libraries
   - System time/dates

## Test Metrics

### Current Status

**Unit Tests:**
- API Service: 47 tests
- App Component: 30 tests
- SearchDatabase: 35 tests
- **Total: 112 unit tests**

**E2E Tests:**
- App workflows: 25 tests
- Search functionality: 15 tests
- **Total: 40 E2E tests**

**Grand Total: 152 tests**

### Speed Benchmarks

- Unit Tests: ~2-5 seconds
- E2E Tests: ~30-60 seconds
- Full Suite: ~1-2 minutes

## Resources

- [Vitest Documentation](https://vitest.dev/)
- [Vue Test Utils](https://test-utils.vuejs.org/)
- [Playwright Documentation](https://playwright.dev/)
- [Testing Best Practices](https://kentcdodds.com/blog/common-mistakes-with-react-testing-library)

## Contributing

When adding new features:

1. Write unit tests for new components/functions
2. Add E2E tests for new user workflows
3. Ensure coverage stays above 80%
4. Run full test suite before submitting PR

```bash
# Before submitting PR:
npm run test:run          # Unit tests
npm run test:coverage     # Check coverage
npm run test:e2e          # E2E tests
```
