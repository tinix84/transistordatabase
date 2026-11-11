import { randomUUID } from 'node:crypto'
import { defineConfig, devices } from '@playwright/test'

// Start the compiled application separately on port 8002.
// Each invocation preserves earlier test artifacts in a different directory.
export default defineConfig({
  testDir: './src/tests/e2e',
  outputDir: `/tmp/tdb-parity-browser-${randomUUID()}`,
  fullyParallel: false,
  workers: 1,
  timeout: 60_000,
  expect: { timeout: 30_000 },
  reporter: 'line',
  use: {
    baseURL: 'http://localhost:8002',
    trace: 'off',
    screenshot: 'only-on-failure',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
})
