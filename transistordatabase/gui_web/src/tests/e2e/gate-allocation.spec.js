import { expect, test } from '@playwright/test'
import { readFile } from 'node:fs/promises'
import { inspectRenderedChart } from '../fixtures/chartEvidence.js'

const field = (page, name) => page.getByTestId(`allocation-${name}`)
async function fixture(page) {
  await page.goto('/index.html?view=waveform')
  await page.getByTestId('waveform-csv').fill('time_s,current_a,gate_v\n5,-3,0\n6,4,1\n8,-5,1\n10,0,0')
  await page.getByTestId('gate-allocation').locator('summary').first().click()
  await field(page, 'source').fill('=Independent signed allocation fixture')
  await field(page, 'selected').check()
}
async function calculate(page) {
  const pending = page.waitForResponse(response => response.url().includes('/api/waveform/gate-allocation'))
  await field(page, 'calculate').click()
  const response = await pending
  expect(response.status()).toBe(200)
  const raw = await response.text()
  await expect(field(page, 'results')).toBeVisible()
  return { raw, data: JSON.parse(raw) }
}
async function download(page, name) {
  const waiting = page.waitForEvent('download')
  await field(page, name).click()
  return readFile(await (await waiting).path(), 'utf8')
}
for (const width of [1440, 390]) {
  test(`real signed gate allocations and portable evidence at ${width}px`, async ({ page }, info) => {
    const errors = []; page.on('pageerror', error => errors.push(error.message))
    await page.setViewportSize({ width, height: 844 }); await fixture(page)
    const { raw, data } = await calculate(page)
    expect(data.result.samples.map(row => row.gate_on_current_a)).toEqual([0, 4, -5, 0])
    expect(data.result.samples.map(row => row.gate_off_current_a)).toEqual([-3, 0, 0, 0])
    for (const row of data.result.samples) expect(row.gate_on_current_a + row.gate_off_current_a).toBe(row.current_a)
    expect(data.inputs.conditions).toEqual({ junction_temperature_c: null, gate_on_voltage_v: null })
    await expect(field(page, 'results')).toContainText('No state or interpolation is defined between samples')
    await expect(field(page, 'results').locator('canvas')).toHaveCount(2)
    await page.waitForTimeout(1200)
    const chart = await field(page, 'results').locator('.curve-plotter').first().evaluate(inspectRenderedChart)
    expect(chart.datasets[1].points.map(point => point.input.y)).toEqual([0, 4, -5, 0])
    for (const plot of await field(page, 'results').locator('.curve-plotter').all()) {
      const rendered = await plot.evaluate(inspectRenderedChart)
      expect(rendered.datasets.every(dataset => dataset.showLine === false)).toBe(true)
      for (const dataset of rendered.datasets) for (const point of dataset.points) expect(point.value).toBeCloseTo(point.input.y, 8)
    }
    expect(await download(page, 'json')).toBe(raw)
    const csv = await download(page, 'csv')
    expect(csv).toContain('gate_on_current_a'); expect(csv).toContain('"\'=Independent signed allocation fixture"')
    expect(csv).toContain(data.evidence.input_sha256); expect(csv).toContain('No state or interpolation')
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(width + 1)
    await field(page, 'results').screenshot({ path: info.outputPath(`gate-allocation-${width}.png`) })
    await page.getByTestId('waveform-csv').fill('time_s,current_a,gate_v\n0,1,0\n1,2,1')
    await expect(field(page, 'results')).toHaveCount(0); await expect(field(page, 'json')).toHaveCount(0)
    expect(errors).toEqual([])
  })
}
test('threshold equality, missing gates, strict decimals and stale replies retain the declared scope', async ({ page }) => {
  await fixture(page)
  await page.getByTestId('waveform-csv').fill('time_s,current_a,gate_v\n0,-3,-5\n1,4,.5\n2,-5,2')
  await field(page, 'calculate').click(); await expect(page.getByTestId('gate-allocation').getByRole('alert')).toContainText('0/1')
  await page.getByTestId('waveform-csv').fill('time_s,current_a,gate_v\n0,-3,0\n1,4,.99999999999999999')
  await field(page, 'calculate').click(); await expect(page.getByTestId('gate-allocation').getByRole('alert')).toContainText('exact numeric 0/1')
  await page.getByTestId('waveform-csv').fill('time_s,current_a,gate_v\n0,-3,-5\n1,4,.5\n2,-5,2')
  await field(page, 'mode').selectOption('threshold'); await field(page, 'threshold').fill('.5')
  const { data } = await calculate(page)
  expect(data.result.samples.map(row => row.gate_on)).toEqual([false, false, true])
  await field(page, 'threshold').fill('-1e-999'); await field(page, 'calculate').click()
  await expect(page.getByTestId('gate-allocation').getByRole('alert')).toContainText('too small')
  await field(page, 'threshold').fill('.5')
  const temperature = page.getByTestId('gate-allocation').getByLabel('Junction temperature (°C), blank = unknown')
  await temperature.fill('1e-999'); await field(page, 'calculate').click()
  await expect(page.getByTestId('gate-allocation').getByRole('alert')).toContainText('too small'); await temperature.fill('')
  await page.getByTestId('waveform-csv').fill('time_s,current_a\n0,1\n1,2')
  await field(page, 'calculate').click(); await expect(page.getByTestId('gate-allocation').getByRole('alert')).toContainText('requires a gate_v')
  await page.getByTestId('waveform-csv').fill('time_s,current_a,gate_v\n0,1e-999,0\n1,2,1')
  await field(page, 'calculate').click(); await expect(page.getByTestId('gate-allocation').getByRole('alert')).toContainText('too small')
  await page.getByTestId('waveform-csv').fill('time_s,current_a,gate_v\n0,1,0\n1,2,1')
  let release
  const gate = new Promise(resolve => { release = resolve })
  await page.route('**/api/waveform/gate-allocation', async route => { const response = await route.fetch(); await gate; await route.fulfill({ response }) })
  await field(page, 'calculate').click(); await field(page, 'source').fill('Changed while pending'); release()
  await expect(field(page, 'calculate')).toBeEnabled(); await expect(field(page, 'results')).toHaveCount(0)
})
