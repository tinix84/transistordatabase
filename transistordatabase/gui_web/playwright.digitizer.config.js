import { defineConfig } from '@playwright/test'

// Use the compiled app and an already running Python service.
export default defineConfig({
  testDir: './src/tests/e2e',
  testMatch: 'digitizer-workbench.spec.js',
  timeout: 120000,
  workers: 1,
  reporter: 'line',
  use: { browserName: 'chromium', headless: true },
})
