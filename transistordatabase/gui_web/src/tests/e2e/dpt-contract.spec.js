import { readFile } from 'node:fs/promises'
import { test, expect } from '@playwright/test'

async function openDpt(page, url = '/') {
  await page.goto(url)
  await page.getByRole('button', { name: 'More', exact: false }).click()
  await page.getByRole('button', { name: 'DPT Manager', exact: true }).click()
  const selector = page.getByLabel('DPT transistor label')
  await expect(selector.locator('option')).not.toHaveCount(1)
  const value = await selector.locator('option').nth(1).getAttribute('value')
  await selector.selectOption(value)
  await page.getByRole('button', { name: 'Generate Netlist', exact: false }).click()
  await expect(page.getByRole('button', { name: 'Generate LTspice Netlist', exact: true })).toBeDisabled()
  await page.getByLabel('Use the uncalibrated generic NMOS template').check()
}

test('real DPT endpoint renders and downloads the disclosed two-pulse circuit', async ({ page }) => {
  const errors = []
  page.on('pageerror', error => errors.push(error.message))
  await openDpt(page)
  await page.getByRole('button', { name: 'Generate LTspice Netlist', exact: true }).click()
  const netlist = page.getByLabel('Generated DPT netlist')
  await expect(netlist).toContainText('D_freewheel sw bus GENERIC_FREEWHEEL')
  await expect(page.locator('.netlist-display')).toContainText('not a vendor device model')
  await expect(page.locator('.netlist-display')).toContainText('5.000000 µs')
  const downloadPromise = page.waitForEvent('download')
  await page.getByRole('button', { name: 'Download .cir file', exact: false }).click()
  const download = await downloadPromise
  expect(download.suggestedFilename()).toMatch(/^dpt_.+\.cir$/)
  const saved = await readFile(await download.path(), 'utf8')
  expect(saved).toBe(await netlist.textContent())
  const pwl = saved.match(/PWL\(([^)]+)\)/)[1].split(/\s+/).map(Number)
  const points = Array.from({ length: pwl.length / 2 }, (_, i) => pwl.slice(i * 2, i * 2 + 2))
  expect(points).toHaveLength(10)
  expect(points[3][0] - points[2][0]).toBeCloseTo(5e-6, 12)
  expect(points[5][0] - points[4][0]).toBeCloseTo(5e-6, 12)
  expect(points[7][0] - points[6][0]).toBeCloseTo(2e-6, 12)
  await page.screenshot({ path: test.info().outputPath('dpt-generic-netlist.png'), fullPage: true })
  await page.getByLabel('R_g Turn-Off:', { exact: true }).fill('42')
  await expect(netlist).toHaveCount(0)
  await page.getByRole('button', { name: 'Generate LTspice Netlist', exact: true }).click()
  await expect(netlist).toContainText('R_off off_path drive 42.0')
  await page.getByLabel('Use the uncalibrated generic NMOS template').uncheck()
  await expect(netlist).toHaveCount(0)
  expect(errors).toEqual([])
})

test('DPT rejects invalid resistance and discards an in-flight result after input changes', async ({ page }) => {
  await openDpt(page)
  await page.getByLabel('R_g Turn-Off:', { exact: true }).fill('0')
  await page.getByRole('button', { name: 'Generate LTspice Netlist', exact: true }).click()
  await expect(page.getByRole('alert')).toContainText('r_g_off must be positive')
  await expect(page.getByLabel('Generated DPT netlist')).toHaveCount(0)
  await page.getByLabel('R_g Turn-Off:', { exact: true }).fill('10')
  let release
  const paused = new Promise(resolve => { release = resolve })
  let fetched
  const fetchedPromise = new Promise(resolve => { fetched = resolve })
  await page.route('**/api/dpt/generate_netlist?*', async route => {
    const response = await route.fetch()
    fetched()
    await paused
    await route.fulfill({ response })
  })
  await page.getByRole('button', { name: 'Generate LTspice Netlist', exact: true }).click()
  await fetchedPromise
  await page.getByLabel('Pulse 2 Duration:', { exact: true }).fill('0.000003')
  release()
  await expect(page.getByRole('button', { name: 'Generate LTspice Netlist', exact: true })).toBeEnabled()
  await expect(page.getByLabel('Generated DPT netlist')).toHaveCount(0)
})


test('compiled DPT template remains usable on a narrow screen', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await openDpt(page, 'http://localhost:8002/index.html')
  await page.getByRole('button', { name: 'Generate LTspice Netlist', exact: true }).click()
  await expect(page.getByLabel('Generated DPT netlist')).toContainText('PWL(')
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(391)
  await page.screenshot({ path: test.info().outputPath('dpt-mobile.png'), fullPage: true })
})
