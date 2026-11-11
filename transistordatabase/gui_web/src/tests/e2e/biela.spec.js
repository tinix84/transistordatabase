import { test, expect } from '@playwright/test'

async function openBiela(page, url = '/') {
  await page.goto(url)
  await page.getByRole('button', { name: 'More', exact: false }).click()
  await page.getByRole('button', { name: 'Analytical Models', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Christen–Biela half-bridge estimate', exact: true })).toBeVisible()
  await page.getByLabel('Use the corrected analytical half-bridge estimate').check()
}

test('actual Biela SI contract shows independent fixture, partition chart, conditions and export', async ({ page }) => {
  const errors = []
  page.on('pageerror', e => errors.push(e.message))
  await openBiela(page)
  await page.getByLabel('Include a bounded current sweep').check()
  await page.getByRole('button', { name: 'Calculate half-bridge estimate', exact: true }).click()
  const result = page.getByLabel('Biela estimate results')
  await expect(result).toContainText('67.707784 µJ')
  await expect(result).toContainText('111.05118 µJ')
  await expect(result).toContainText('178.75897 µJ')
  await expect(result).toContainText('Independent constant-gm algebra')
  await expect(result.locator('canvas')).toHaveCount(2)
  await expect(page.getByLabel('Biela sweep evidence')).toContainText('unsupported')
  await expect(page.getByLabel('Biela sweep evidence')).toContainText('supported')
  const event = page.waitForEvent('download')
  await page.getByRole('button', { name: 'Download reproducible estimate JSON' }).click()
  const download = await event
  const stream = await download.createReadStream()
  const chunks = []
  for await (const chunk of stream) chunks.push(chunk)
  const exported = JSON.parse(Buffer.concat(chunks).toString())
  expect(exported.parameter_inputs_si.q_oss).toBe(200e-9)
  expect(exported.parameter_inputs_si.parameter_t_j).toBe(25)
  expect(exported.sweep.points[0].status).toBe('unsupported')
  expect(exported.results.e_total).toBeCloseTo(178.7589668316306e-6, 12)
  await page.screenshot({ path: test.info().outputPath('biela-actual-contract.png'), fullPage: true })
  await page.locator('[name=t_j]').fill('100')
  await expect(result).toHaveCount(0)
  await page.getByRole('button', { name: 'Calculate half-bridge estimate', exact: true }).click()
  await expect(page.getByRole('alert')).toContainText('Parameter temperature must equal')
  expect(errors).toEqual([])
})

test('recovery partitions are explicit and no-root estimate reports an actionable domain error', async ({ page }) => {
  await openBiela(page)
  await page.locator('[name=recovery_mode]').selectOption('charge_control')
  await page.getByRole('button', { name: 'Calculate half-bridge estimate', exact: true }).click()
  await expect(page.getByLabel('Biela estimate results')).toContainText('71.056924 µJ')
  await expect(page.getByLabel('Biela estimate results')).toContainText('recovery_storage')
  await page.locator('[name=l_s]').fill('1e-7')
  await expect(page.getByLabel('Biela estimate results')).toHaveCount(0)
  await page.getByRole('button', { name: 'Calculate half-bridge estimate', exact: true }).click()
  await expect(page.getByRole('alert')).toContainText('no converged estimate established')
})

test('compiled mobile workflow keeps labeled phase bars and controls within the viewport', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await openBiela(page, 'http://localhost:8002/index.html')
  await page.getByRole('button', { name: 'Calculate half-bridge estimate', exact: true }).click()
  await expect(page.getByLabel('Biela estimate results')).toContainText('178.75897 µJ')
  await expect(page.getByLabel('Switching phase energy bars')).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
  await page.screenshot({ path: test.info().outputPath('biela-mobile-compiled.png'), fullPage: true })
})
