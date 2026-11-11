import { defineConfig, devices } from '@playwright/test'
export default defineConfig({
  testDir: './e2e', outputDir: '/tmp/tdb-consolidation-browser-57876578-f75f-4604-8bbb-b94717415892', fullyParallel: false,
  workers: 1, retries: 0, timeout: 60000,
  expect: { timeout: 30000 }, reporter: 'line',
  use: { baseURL: 'http://127.0.0.1:8003', trace: 'off', screenshot: 'only-on-failure' },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
})
