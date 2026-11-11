import { test, expect } from '@playwright/test'
import fs from 'node:fs/promises'
import { inspectRenderedChart } from '../fixtures/chartEvidence.js'

async function load(page, name = 'Load independent DAB fixture', compiled = false) {
  await page.goto(compiled ? 'http://localhost:8002/index.html?view=converter' : '/index.html?view=converter')
  await expect(page.getByRole('heading', { name: 'Converter loss workbench', exact: true })).toBeVisible()
  await page.getByRole('button', { name, exact: true }).click()
}
async function calculate(page) {
  const waiting = page.waitForResponse(response => response.url().includes('/api/converters/analyze') && response.request().method() === 'POST')
  await page.getByRole('button', { name: 'Calculate converter losses', exact: true }).click()
  const response = await waiting
  expect(response.status()).toBe(200)
  return response.json()
}

test('DAB distinct conduction/commutation signals, linked real charts and canonical exports', async ({ page }) => {
  const errors = []; page.on('pageerror', error => errors.push(error.message))
  await load(page); await page.getByTestId('converter-symmetry').check()
  const data = await calculate(page)
  expect(data.physical_member_count).toBe(8)
  expect(data.totals.total_power_w).toBeCloseTo(4040, 8)
  expect(data.branches[0].evidence.events.map(event => event.current_a)).toEqual([-2, -3])
  const result = page.getByTestId('converter-results')
  await expect(result).toContainText('4040.00 W'); await expect(result.locator('canvas')).toHaveCount(6)
  const plots = result.locator('.curve-plotter')
  const renderedCharts = []
  for (const plot of await plots.all()) renderedCharts.push(await plot.evaluate(inspectRenderedChart))
  const datasets = renderedCharts.map(chart => chart.datasets.map(series => series.points.map(point => point.input)))
  expect(datasets[0][0]).toEqual(data.branches[0].evidence.conduction_signals.time_s.map((x, i) => ({ x, y: data.branches[0].evidence.conduction_signals.current_a[i] })))
  expect(datasets[1][0]).toEqual(data.branches[0].evidence.switching_signals.time_s.map((x, i) => ({ x, y: data.branches[0].evidence.switching_signals.current_a[i] })))
  await page.getByTestId('converter-range-start').fill('0.000001'); await page.getByTestId('converter-range-end').fill('0.000003')
  await page.waitForTimeout(1200)
  const rendered = await plots.nth(1).evaluate(inspectRenderedChart)
  expect(rendered.xMin).toBe(1e-6); expect(rendered.xMax).toBe(3e-6)
  for (const point of rendered.datasets[0].points) expect(point.value).toBeCloseTo(point.input.y, 8)
  const waiting = page.waitForEvent('download'); await page.getByRole('button', { name: 'Download request + result JSON' }).click()
  const exported = JSON.parse(await fs.readFile(await (await waiting).path(), 'utf8'))
  // JSON serialization canonicalizes signed zero; all physical values and evidence remain identical.
  expect(exported.response).toEqual(JSON.parse(JSON.stringify(data))); expect(exported.request).toEqual(data.request)
  const csvWaiting = page.waitForEvent('download'); await page.getByRole('button', { name: 'Download branch CSV' }).click()
  const csv = await fs.readFile(await (await csvWaiting).path(), 'utf8')
  const first = csv.split(/\r?\n/)[1].split('","').map(cell => cell.replace(/^"|"$/g, ''))
  expect(first.slice(0, 3)).toEqual(['s12', 's1;s2', '2'])
  expect(first.slice(3, 7).map(Number)).toEqual(['conduction', 'turn_on', 'turn_off'].map(key => data.branches[0].scaled_components[key].power_w).concat(data.branches[0].known_component_power_w))
  await result.screenshot({ path: test.info().outputPath('converter-dab-results.png') })
  await page.getByTestId('converter-source').fill('Changed waveform identity'); await expect(result).toHaveCount(0)
  expect(errors).toEqual([])
})

test('LLC omissions stay unknown and held-envelope powers do not claim PWM events', async ({ page }) => {
  await load(page, 'Load LLC omission fixture'); await page.getByTestId('converter-symmetry').check(); await page.getByTestId('converter-omission').check()
  const llc = await calculate(page)
  expect(llc.totals.total_power_w).toBeNull(); expect(llc.totals.known_component_power_w).toBeCloseTo(2040, 8)
  await expect(page.getByTestId('converter-results')).toContainText('Complete total: Unknown W')
  await page.getByRole('button', { name: 'Load independent envelope fixture' }).click(); await page.getByTestId('converter-symmetry').check(); await page.getByTestId('converter-envelope').check()
  const envelope = await calculate(page)
  expect(envelope.operation).toBe('held-envelope-estimate'); expect(envelope.totals.components.conduction.power_w).toBeCloseTo(180, 8)
  expect(envelope.branches[2].evidence.intervals[0].current_a).toBe(30)
  const result = page.getByTestId('converter-results'); await expect(result).toContainText('No resolved PWM event count')
  await expect(result.locator('canvas')).toHaveCount(5)
})

test('held converter reply is invalidated and compiled 390px plots fit viewport', async ({ page }) => {
  const errors = []; page.on('pageerror', error => errors.push(error.message))
  await page.setViewportSize({ width: 390, height: 844 }); await load(page, 'Load independent DAB fixture', true)
  await page.getByTestId('converter-symmetry').check()
  let release, received
  const pending = new Promise(resolve => { received = resolve })
  const held = new Promise(resolve => { release = resolve })
  await page.route('**/api/converters/analyze', async route => { const response = await route.fetch(); received(); await held; await route.fulfill({ response }) })
  await page.getByRole('button', { name: 'Calculate converter losses', exact: true }).click(); await pending
  await page.getByTestId('signal-itp').fill('0,4,9,6,0'); release()
  await expect(page.getByRole('button', { name: 'Calculate converter losses', exact: true })).toBeEnabled()
  await expect(page.getByTestId('converter-results')).toHaveCount(0)
  await page.unroute('**/api/converters/analyze'); await calculate(page)
  const result = page.getByTestId('converter-results'); await expect(result.locator('canvas')).toHaveCount(6)
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
  await result.screenshot({ path: test.info().outputPath('converter-mobile-results.png') }); expect(errors).toEqual([])
})

test('compiled mobile envelope keeps readable tables inside scroll containers', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await load(page, 'Load independent envelope fixture', true)
  await page.getByTestId('converter-symmetry').check(); await page.getByTestId('converter-envelope').check()
  await calculate(page); await page.getByTestId('converter-branch').selectOption('neutral_igbts')
  const result = page.getByTestId('converter-results'); await expect(result.locator('canvas')).toHaveCount(5)
  await page.waitForTimeout(1200)
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
  const dimensions = await result.locator('.table-scroll').first().evaluate(node => ({ available: node.clientWidth, content: node.scrollWidth, table: node.querySelector('table').getBoundingClientRect().width }))
  expect(dimensions.available).toBeLessThan(390); expect(dimensions.content).toBeGreaterThanOrEqual(650); expect(dimensions.table).toBeGreaterThanOrEqual(650)
  await result.locator('.table-scroll').first().screenshot({ path: test.info().outputPath('converter-mobile-readable-table.png') })
  await result.screenshot({ path: test.info().outputPath('converter-mobile-envelope-readable.png') })
})
