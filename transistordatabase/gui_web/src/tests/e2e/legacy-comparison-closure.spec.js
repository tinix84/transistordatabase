import { test, expect } from '@playwright/test'
import fs from 'node:fs/promises'

// Production app/HTTP/calculations; only the independent algebraic repository is in memory.
// Start tests/legacy_comparison_browser_server.py on port8031 before running this spec.
const origin = 'http://localhost:8031'
async function load(page) {
  await page.goto(`${origin}/index.html?view=compare-advanced`)
  await expect(page.getByRole('heading', { name: 'Advanced transistor comparison', exact: true })).toBeVisible()
  await page.getByTestId('comparison-device-1').selectOption('bounded-a')
  await page.getByTestId('comparison-device-3').selectOption('bounded-b')
}
async function estimate(page) {
  await page.getByTestId('comparison-estimate').check()
  for (const [key, value] of Object.entries({ t_j: 25, i_channel: 10, f_sw: 100000, duty: .25, output_power_w: 1000 })) await page.getByTestId(`comparison-${key}`).fill(String(value))
  for (const group of await page.locator('fieldset.config-panel').all()) {
    await group.getByLabel('Supply voltage (V)', { exact: true }).fill('400')
    await group.getByLabel('Gate on voltage (V)', { exact: true }).fill('15')
    await group.locator('select').selectOption({ label: 'Known' })
    await group.getByLabel('Gate off voltage (V)', { exact: true }).fill('-3')
    await group.getByLabel('Gate resistance on (Ω)', { exact: true }).fill('4')
    await group.getByLabel('Gate resistance off (Ω)', { exact: true }).fill('7')
  }
}
async function run(page) {
  const waiting = page.waitForResponse(response => response.url().endsWith('/api/comparison/advanced') && response.request().method() === 'POST')
  await page.getByTestId('run-comparison').click()
  const response = await waiting; expect(response.status()).toBe(200)
  return response.json()
}
test('compiled comparison preserves observations and computes305W/605W from explicit PWM inputs', async ({ page }) => {
  const errors = []; page.on('pageerror', error => errors.push(error.message))
  await load(page)
  const observed = await run(page)
  expect(observed.loss_breakdown.rows.map(row => row.total)).toEqual([null, null])
  expect(observed.channel_characteristics.curves.map(curve => curve.conditions)).toEqual([{ t_j: 25, v_g: 15 }, { t_j: 25, v_g: 15 }])
  await estimate(page); const calculated = await run(page)
  expect(calculated.loss_breakdown.rows.map(row => row.total)).toEqual([305, 305])
  expect(calculated.efficiency.rows[0].efficiency_percent).toBeCloseTo(1000 / 1305 * 100, 10)
  await page.getByRole('button', { name: 'Loss breakdown', exact: true }).click()
  await expect(page.getByTestId('comparison-results')).toContainText('305.0000')
  const download = page.waitForEvent('download'); await page.getByRole('button', { name: 'Download evidence JSON' }).click()
  const exported = JSON.parse(await fs.readFile(await (await download).path(), 'utf8'))
  expect(exported).toEqual(calculated)
  await page.getByTestId('comparison-f_sw').fill('200000')
  await expect(page.getByTestId('comparison-results')).toHaveCount(0)
  const doubled = await run(page); expect(doubled.loss_breakdown.rows[0].total).toBe(605)
  await page.getByTestId('comparison-results').screenshot({ path: test.info().outputPath('comparison-computed-desktop.png') })
  expect(errors).toEqual([])
})

test('compiled390px comparison retains per-device mismatch diagnostics and readable tables', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 }); await load(page); await estimate(page)
  await page.getByTestId('comparison-supply-3').fill('401')
  const data = await run(page)
  expect(data.loss_breakdown.rows[0].total).toBe(305); expect(data.loss_breakdown.rows[1].total).toBeNull()
  expect(data.efficiency.rows[1].efficiency_percent).toBeNull()
  await page.getByRole('button', { name: 'Loss breakdown', exact: true }).click()
  const result = page.getByTestId('comparison-results')
  await expect(result).toContainText('exact requested'); await expect(result).toContainText('Unavailable')
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390)
  expect(await result.locator('table').evaluate(node => node.getBoundingClientRect().width)).toBeGreaterThanOrEqual(650)
  expect(await result.locator('tbody td').first().evaluate(node => node.getBoundingClientRect().width)).toBeGreaterThanOrEqual(110)
  expect(await result.locator('.comparison-table-scroll').evaluate(node => node.clientWidth)).toBeLessThan(390)
  await result.screenshot({ path: test.info().outputPath('comparison-missing-mobile.png') })
  await page.getByRole('button', { name: 'Efficiency', exact: true }).click(); await expect(result).toContainText('Excludes magnetics')
  await result.screenshot({ path: test.info().outputPath('comparison-efficiency-mobile.png') })
})

test('compiled comparison suppresses a completed live response after inputs change', async ({ page }) => {
  await load(page); await estimate(page)
  let release; const held = new Promise(resolve => { release = resolve }); let arrived
  const started = new Promise(resolve => { arrived = resolve })
  await page.route('**/api/comparison/advanced', async route => { const response = await route.fetch(); arrived(); await held; await route.fulfill({ response }) })
  await page.getByTestId('run-comparison').click(); await started
  await page.getByTestId('comparison-f_sw').fill('200000'); release()
  await expect(page.getByRole('status').filter({ hasText: 'Inputs changed' })).toBeVisible()
  await expect(page.getByTestId('run-comparison')).toBeEnabled()
  await expect(page.getByTestId('comparison-results')).toHaveCount(0)
})

test('compiled8002 stored-device comparison keeps observations without invented losses', async ({ page }) => {
  const records = await (await page.request.get('http://localhost:8002/api/transistors')).json()
  const names = records.slice(0, 2).map(record => record.metadata.name)
  expect(names).toHaveLength(2)
  await page.goto('http://localhost:8002/index.html?view=compare-advanced')
  await page.getByTestId('comparison-device-1').selectOption(names[0])
  await page.getByTestId('comparison-device-2').selectOption(names[1])
  const data = await run(page)
  expect(data.loss_breakdown.rows.map(row => row.total)).toEqual([null, null])
  expect(data.efficiency.rows.map(row => row.efficiency_percent)).toEqual([null, null])
  await page.getByRole('button', { name: 'Loss breakdown', exact: true }).click()
  await expect(page.getByTestId('comparison-results')).toContainText('Explicit static-duty estimate needs')
  await page.getByTestId('comparison-results').screenshot({ path: test.info().outputPath('comparison-live-stored-devices.png') })
})
