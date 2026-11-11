import { test, expect } from '@playwright/test'
import fs from 'node:fs/promises'
import { inspectRenderedChart } from '../fixtures/chartEvidence.js'

const omitted = { omit: true, omission_reason: 'Independent FFT fixture has no device loss models' }
const waveform = { time_s: [0, 1, 2, 3], current_a: [2, 2, 2, -2], gate_v: [0, 0, 1, 1],
  device_voltage_v: [400, 400, 2, 2], bus_voltage_v: 400 }
const spectrum = { window: 'rectangular', detrend: 'none', relative_threshold: .5 }
const events = { mode: 'omitted', omission_reason: 'FFT fixture has no switching events' }
const switching = { turn_on: omitted, turn_off: omitted }
const point = { id: 'phasor', source: 'Independent [2,2,2,-2] A samples', weight: 1, waveform, spectrum, events }
const payload = {
  configurations: [{ id: 'fixture', source: 'Explicit omitted loss fixture', conduction: omitted, switching }],
  operating_points: [point,
    { ...point, id: 'not-requested', weight: 0, spectrum: null },
    { ...point, id: 'nonuniform', waveform: { ...waveform, time_s: [0, 1, 2, 4] } },
    { ...point, id: 'zero', waveform: { time_s: [0, 1, 2, 3], current_a: [0, 0, 0, 0] } }],
  metric: { name: 'conduction_power_w', unit: 'W', direction: 'minimize' }, grouped_series_exports: true,
}
async function calculate(page, route, button) {
  const waiting = page.waitForResponse(value => value.url().includes(route) && value.request().method() === 'POST')
  await page.getByRole('button', { name: button, exact: true }).click()
  const response = await waiting; expect(response.status()).toBe(200); return response.json()
}
async function downloaded(page, button) {
  const waiting = page.waitForEvent('download')
  await page.getByRole('button', { name: button, exact: true }).click()
  const file = await waiting; return fs.readFile(await file.path(), 'utf8')
}

for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }]) {
  test(`compiled harmonic evidence and raw downloads at ${viewport.width}px`, async ({ page }) => {
    await page.setViewportSize(viewport)
    const errors = []; page.on('pageerror', error => errors.push(error.message))
    await page.goto('/index.html?view=waveform')
    await page.getByText('Advanced: inspect or supply the canonical JSON request', { exact: true }).click()
    await page.getByLabel('Use JSON editor as calculation input', { exact: true }).check()
    await page.getByLabel('Canonical request', { exact: true }).fill(JSON.stringify({ waveform, spectrum, events, switching, conduction: omitted }))
    const current = await calculate(page, '/api/waveform/analyze', 'Calculate waveform losses')
    expect(current.spectrum.real_a).toEqual([1, 0, 1]); expect(current.spectrum.imaginary_a).toEqual([0, -2, 0])
    expect(current.spectrum.selected_bin_indices).toEqual([1]); expect(current.spectrum.phase_rad[1]).toBeCloseTo(-Math.PI / 2, 14)
    await expect(page.getByTestId('current-harmonics').locator('tbody tr')).toHaveCount(1)
    await expect(page.getByTestId('current-harmonics')).toContainText('-1.57080')
    const fft = page.locator('.curve-plotter').filter({ has: page.getByRole('heading', { name: 'Uniform-sample current spectrum', exact: true }) })
    await expect(fft.locator('canvas')).toBeVisible()
    await expect.poll(async () => {
      const drawn = await fft.evaluate(inspectRenderedChart)
      const targetY = drawn.chartArea.bottom - (2 - drawn.yMin) / (drawn.yMax - drawn.yMin) * (drawn.chartArea.bottom - drawn.chartArea.top)
      return Math.abs(drawn.datasets[0].points[1].pixel.y - targetY)
    }).toBeLessThan(.000001)
    const evidence = await fft.evaluate(inspectRenderedChart)
    expect(evidence.datasets[0].points.map(value => value.input)).toEqual([{ x: 0, y: 1 }, { x: .25, y: 2 }, { x: .5, y: 1 }])
    for (const point of evidence.datasets[0].points) expect(point.value).toBeCloseTo(point.input.y, 8)
    expect(evidence.cursor).toBe(null)
    await fft.evaluate(node => node.scrollIntoView({ block: 'start' }))
    await page.evaluate(() => window.scrollBy(0, -110))
    await page.screenshot({ path: test.info().outputPath(`current-spectrum-${viewport.width}.png`) })
    await page.getByTestId('current-harmonics').evaluate(node => node.scrollIntoView({ block: 'start' }))
    await page.evaluate(() => window.scrollBy(0, -110))
    await page.screenshot({ path: test.info().outputPath(`current-harmonics-${viewport.width}.png`) })
    const saved = JSON.parse(await downloaded(page, 'Download request + result JSON'))
    expect(saved.response.spectrum).toEqual(current.spectrum)
    await page.goto('/index.html?view=analysis-grid')
    await expect(page.getByLabel('Include grouped series statistics and harmonic exports', { exact: true })).not.toBeChecked()
    await page.getByText('Advanced canonical request, including explicit event masks', { exact: true }).click()
    await page.getByLabel('Use grid JSON as input', { exact: true }).check()
    await page.getByLabel('Grid request JSON', { exact: true }).fill(JSON.stringify(payload))
    const result = await calculate(page, '/api/analysis/grid', 'Calculate analysis grid')
    const projection = result.grouped_series_exports
    const row = projection.time_rows.find(value => value.operating_point_id === 'phasor' && value.series === 'current_a')
    expect(row).toMatchObject({ sample_count: 4, arithmetic_mean: 1, discrete_rms: 2, minimum: -2, maximum: 2, sample_standard_deviation: 2 })
    expect(row.time_weighted_mean).toBeCloseTo(4 / 3, 14); expect(row.time_weighted_rms).toBeCloseTo(Math.sqrt(28 / 9), 14)
    expect(projection.time_rows.filter(value => value.operating_point_id === 'phasor').map(value => value.unit)).toEqual(['A', 'V', 'V', 'V'])
    const phasor = projection.frequency_rows.find(value => value.operating_point_id === 'phasor' && value.series === 'current_a')
    expect(phasor).toMatchObject({ bin_index: 1, frequency_hz: .25, magnitude: 2, real: 0, imaginary: -2 })
    expect(phasor.phase_rad).toBeCloseTo(-Math.PI / 2, 14)
    expect(projection.frequency_rows.some(value => value.status === 'not_requested')).toBe(true)
    expect(projection.frequency_rows.some(value => value.status === 'unsupported')).toBe(true)
    await expect(page.getByTestId('grouped-series-evidence')).toContainText('Piecewise-linear time RMS')
    await page.getByLabel('Filter pair diagnostics').selectOption('unsupported')
    const timeJson = await downloaded(page, 'Download grouped time JSON')
    expect(timeJson).toBe(projection.time_json); expect(JSON.parse(timeJson)).toEqual(projection.time_rows)
    expect(await downloaded(page, 'Download grouped time CSV')).toBe(projection.time_csv)
    const frequencyJson = await downloaded(page, 'Download grouped frequency JSON')
    expect(frequencyJson).toBe(projection.frequency_json); expect(JSON.parse(frequencyJson)).toEqual(projection.frequency_rows)
    expect(await downloaded(page, 'Download grouped frequency CSV')).toBe(projection.frequency_csv)
    await page.getByTestId('grouped-series-evidence').evaluate(node => node.scrollIntoView({ block: 'start' }))
    await page.evaluate(() => window.scrollBy(0, -110))
    await page.screenshot({ path: test.info().outputPath(`grouped-harmonics-${viewport.width}.png`) })
    await page.getByTestId('grouped-time-table').evaluate(node => {
      node.scrollIntoView({ block: 'start' })
      node.querySelector('.table-scroll').scrollLeft = node.querySelectorAll('th')[4].offsetLeft
    })
    await page.evaluate(() => window.scrollBy(0, -110))
    await page.screenshot({ path: test.info().outputPath(`grouped-statistics-values-${viewport.width}.png`) })
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
    await page.getByLabel('Grid request JSON', { exact: true }).fill(JSON.stringify({ ...payload, grouped_series_exports: false }))
    await expect(page.getByTestId('grouped-series-evidence')).toHaveCount(0)
    await expect(page.getByRole('button', { name: 'Download grouped time JSON', exact: true })).toHaveCount(0)
    const defaultResult = await calculate(page, '/api/analysis/grid', 'Calculate analysis grid')
    expect(defaultResult.grouped_series_exports ?? null).toBe(null)
    expect(errors).toEqual([])
  })
}
