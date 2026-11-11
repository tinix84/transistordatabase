import { test, expect } from '@playwright/test'
import fs from 'node:fs/promises'

async function calculate(page) {
  const waiting = page.waitForResponse(response => response.url().includes('/api/thermal/analyze') && response.request().method() === 'POST')
  await page.getByRole('button', { name: 'Calculate thermal result', exact: true }).click()
  const response = await waiting
  expect(response.status()).toBe(200)
  return response.json()
}

test('real steady paths, source export and stale invalidation', async ({ page }) => {
  const errors = []; page.on('pageerror', error => errors.push(error.message))
  await page.goto('/index.html?view=thermal')
  await expect(page.getByRole('heading', { name: 'Thermal paths and fixed-case response' })).toBeVisible()
  const data = await calculate(page)
  expect(data.result.forward_temperature_c).toBe(37)
  expect(data.result.reverse_temperature_c).toBe(27)
  const waiting = page.waitForEvent('download')
  await page.getByRole('button', { name: 'Export thermal request + result JSON' }).click()
  const exported = JSON.parse(await fs.readFile(await (await waiting).path(), 'utf8'))
  expect(exported.request).toEqual(data.request)
  expect(exported.result).toEqual(data.result)
  await page.getByLabel('Forward heat', { exact: true }).fill('11')
  await expect(page.getByLabel('Thermal result', { exact: true })).toHaveCount(0)
  expect(errors).toEqual([])
})

test('real cubic Foster response and conserved event heat CSV', async ({ page }) => {
  const errors = []; page.on('pageerror', error => errors.push(error.message))
  await page.goto('/index.html?view=thermal')
  await page.getByLabel('Thermal operation', { exact: true }).selectOption('transient')
  const baseline = await calculate(page)
  expect(baseline.result.final_temperature_c).toBeCloseTo(43.15502669168744, 10)
  await page.getByLabel('Event heat convention', { exact: true }).selectOption('impulse')
  await page.getByLabel('Thermal events', { exact: true }).fill('0.037,0.1,forward')
  const impulse = await calculate(page)
  expect(impulse.result.final_temperature_c - baseline.result.final_temperature_c).toBeCloseTo(.16345014985612696, 12)
  const result = page.getByLabel('Thermal result', { exact: true })
  await expect(result.locator('canvas')).toHaveCount(1)
  const waiting = page.waitForEvent('download'); await page.getByRole('button', { name: 'Export thermal evidence CSV' }).click()
  const csv = await fs.readFile(await (await waiting).path(), 'utf8')
  expect(csv).toContain('pre-event'); expect(csv).toContain('post-event')
  await page.waitForTimeout(1200)
  await result.screenshot({ path: test.info().outputPath('thermal-transient.png') })
  expect(errors).toEqual([])
})

test('compiled mobile estimate selection and bounded electrothermal status', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('http://localhost:8002/index.html?view=thermal')
  await page.getByLabel('Thermal operation', { exact: true }).selectOption('electrothermal')
  await page.getByRole('button', { name: 'Calculate thermal result', exact: true }).click()
  await expect(page.getByRole('alert')).toContainText('explicitly')
  await page.getByLabel('Select branch 1 temperature estimate', { exact: true }).check()
  const data = await calculate(page)
  expect(data.result.status).toBe('converged')
  // The default common .2 plus branch .5 gives .7 K/W in this browser fixture.
  expect(data.result.accepted_solution.temperature_c[0]).toBeCloseTo(25 + 7/(1-.028), 7)
  await expect(page.getByLabel('Thermal result', { exact: true })).toContainText('converged')
  await expect(page.getByLabel('Thermal result', { exact: true }).locator('.branch-results')).toContainText('Junction (°C)')
  await page.waitForTimeout(1200)
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
  await page.screenshot({ path: test.info().outputPath('thermal-mobile.png'), fullPage: true })
})
