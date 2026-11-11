import { test, expect } from '@playwright/test'
import fs from 'node:fs/promises'
import { inspectRenderedChart } from '../fixtures/chartEvidence.js'

async function fixture(page, url = '/index.html?view=waveform') {
  await page.goto(url)
  await expect(page.getByRole('heading', { name: 'Waveform workbench', exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'Load independent switching fixture' }).click()
  for (const name of ['Positive-current conduction', 'Positive-current turn-on', 'Positive-current turn-off']) {
    await page.getByRole('group', { name, exact: true }).getByLabel('I select this named estimate and its assumptions').check()
  }
}

test('real pre-edge waveform losses, linked views and reproducible downloads', async ({ page }) => {
  const errors = []; page.on('pageerror', error => errors.push(error.message))
  await fixture(page)
  const response = page.waitForResponse(response => response.url().includes('/api/waveform/analyze') && response.request().method() === 'POST')
  await page.getByRole('button', { name: 'Calculate waveform losses', exact: true }).click()
  const data = await (await response).json()
  expect(data.events.map(event => [event.sample_index, event.current_a, event.energy_j])).toEqual([[1, 2, .001], [3, 3, .002]])
  const result = page.getByTestId('waveform-results')
  await expect(result).toContainText('250.000')
  await expect(result).toContainText('500.000')
  await expect(result.locator('canvas')).toHaveCount(6)
  await page.getByTestId('range-start').fill('0.000001')
  await page.getByTestId('range-end').fill('0.000003')
  // Changing the range during initial animation must keep point pixels aligned
  // with their SI values after that original animation would have completed.
  await page.waitForTimeout(1200)
  const drawnCurrent = (await result.locator('.curve-plotter').filter({
    has: page.getByRole('heading', { name: 'Signed current', exact: true }),
  }).evaluate(inspectRenderedChart)).datasets[0].points.filter(point => point.input.x >= 1e-6 && point.input.x <= 3e-6)
  for (const point of drawnCurrent) expect(point.value).toBeCloseTo(point.input.y, 8)
  await expect(result).toContainText('0.00000400000 s interval')
  await expect(result.getByTitle('Toggle Zoom')).toHaveCount(0)
  const download = page.waitForEvent('download')
  await page.getByRole('button', { name: 'Download request + result JSON' }).click()
  const saved = await download
  const exported = JSON.parse(await fs.readFile(await saved.path(), 'utf8'))
  expect(exported.response.events).toEqual(data.events)
  expect(exported.request.waveform.current_a).toEqual([0, 2, 9, 3, 0])
  for (const [button, filename, content] of [['Download event CSV', 'waveform-events.csv', 'pre-edge'], ['Download interval CSV', 'waveform-intervals.csv', 'power_coefficients_w']]) {
    const waiting = page.waitForEvent('download'); await page.getByRole('button', { name: button }).click()
    const file = await waiting; expect(file.suggestedFilename()).toBe(filename)
    expect(await fs.readFile(await file.path(), 'utf8')).toContain(content)
  }
  await result.screenshot({ path: test.info().outputPath('waveform-linked-results.png') })
  await page.getByTestId('waveform-csv').fill('time_s,current_a\n0,-10\n1,10')
  await expect(result).toHaveCount(0)
  expect(errors).toEqual([])
})

test('real explicit masks preserve two final blanks and FFT uses a separate frequency view', async ({ page }) => {
  await fixture(page)
  await page.getByTestId('waveform-csv').fill('time_s,current_a,turn_on,turn_off\n0,0,0,0\n0.000001,2,1,0\n0.000002,9,0,0\n0.000003,3,0,1\n0.000004,0,,')
  await page.getByLabel('Event policy', { exact: true }).selectOption('masks')
  await page.getByLabel('Compute uniform-sample FFT (reject irregular sampling)').check()
  const response = page.waitForResponse(response => response.url().includes('/api/waveform/analyze') && response.request().method() === 'POST')
  await page.getByRole('button', { name: 'Calculate waveform losses', exact: true }).click()
  expect((await response).status()).toBe(200)
  const result = page.getByTestId('waveform-results')
  await expect(result).toContainText('250.000')
  await expect(result).toContainText('Parseval RMS')
  await expect(result).toContainText('independent frequency axis')
  await expect(result.locator('canvas')).toHaveCount(5)
  await expect(page.getByRole('alert')).toHaveCount(0)
})

test('compiled mobile waveform view rejects unselected estimates and fits the viewport', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('http://localhost:8002/index.html?view=waveform')
  await page.getByRole('button', { name: 'Calculate waveform losses', exact: true }).click()
  await expect(page.getByRole('alert')).toContainText('explicitly')
  await page.getByRole('group', { name: 'Positive-current conduction', exact: true }).getByLabel('I select this named estimate and its assumptions').check()
  await page.getByRole('button', { name: 'Calculate waveform losses', exact: true }).click()
  await expect(page.getByTestId('waveform-results')).toContainText('partial-explicit-omissions')
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
  await page.getByTestId('waveform-results').screenshot({ path: test.info().outputPath('waveform-mobile-results.png') })
})
