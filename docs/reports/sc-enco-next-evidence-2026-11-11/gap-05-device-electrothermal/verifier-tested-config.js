import { defineConfig, devices } from '@playwright/test'
const root = process.env.TDB_NATIVE_ACCEPTANCE_OUTPUT
if (!root) throw new Error('Set an absent unique native-device acceptance output path')
export default defineConfig({
  testDir: './src/tests/e2e', testMatch: ['device-electrothermal-acceptance.spec.js', 'thermal-contacts-acceptance.spec.js'],
  grep: /native device acceptance|thermal contact acceptance actual controls/,
  outputDir: `${root}/test-results`, fullyParallel: false, workers: 1, retries: 0,
  timeout: 60_000, expect: { timeout: 15_000 },
  reporter: [['line'], ['json', { outputFile: `${root}/results.json` }]],
  use: { baseURL: 'http://127.0.0.1:8004', trace: 'off', screenshot: 'only-on-failure' },
  projects: [{ name: 'desktop', use: { ...devices['Desktop Chrome'] } },
    { name: 'mobile-390', use: { browserName: 'chromium', viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true } }],
})
