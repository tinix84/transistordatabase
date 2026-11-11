import { test, expect } from '@playwright/test'
import fs from 'node:fs/promises'

async function start(page, fixture = 'Load independent DAB fixture') {
  await page.goto('http://localhost:8002/index.html?view=converter')
  await page.getByRole('button', { name: fixture, exact: true }).click()
  await page.getByTestId('converter-symmetry').check()
}
async function calculate(page) {
  const pending = page.waitForResponse(r => r.url().includes('/api/converters/analyze') && r.request().method() === 'POST')
  await page.getByRole('button', { name: 'Calculate converter losses', exact: true }).click()
  const reply = await pending
  expect(reply.status()).toBe(200)
  return reply.json()
}
async function download(page, name) {
  const waiting = page.waitForEvent('download')
  await page.getByRole('button', { name, exact: true }).click()
  return fs.readFile(await (await waiting).path(), 'utf8')
}

test('independent compiled DAB secondary direction, full evidence and exact downloads', async ({ page }) => {
  const errors = []; page.on('pageerror', error => errors.push(error.message))
  await start(page)
  let response = await calculate(page)
  const distinct = structuredClone(response.request)
  for (const model of Object.values(distinct.role_models)) {
    for (const [side, voltage, energy] of [['turn_on', 400, .002], ['turn_off', 600, .003]]) {
      for (const direction of ['positive', 'negative']) {
        Object.assign(model.switching[side][direction], {
          mode: 'voltage-scaling-estimate', selected: true, energy_j: [energy, energy],
          source: `Independent browser ${side} ${voltage}V fixture`,
        })
        model.switching[side][direction].recorded_conditions.v_supply_v = voltage
      }
    }
  }
  await page.getByText('Advanced canonical request import / inspection', { exact: true }).click()
  await page.getByTestId('converter-json').fill(JSON.stringify(distinct))
  await page.getByLabel('Use JSON editor as calculation input', { exact: true }).check()
  response = await calculate(page)
  const result = page.getByTestId('converter-results')
  expect(response.physical_member_count).toBe(8)
  expect(response.totals.total_power_w).toBeCloseTo(4040, 10)
  expect(response.bindings.map(row => row.members)).toEqual([['s1', 's2'], ['s3', 's4'], ['s5', 's6'], ['s7', 's8']])
  await page.getByTestId('converter-branch').selectOption('s78')
  const secondary = response.branches[3]
  expect(secondary.evidence.events.map(row => row.current_a)).toEqual([4, 5])
  expect(secondary.evidence.conduction_signals.current_a).toEqual([4, 4, 4, 4, 4])
  expect(secondary.evidence.events.map(row => row.recorded_conditions.v_supply_v)).toEqual([400, 600])
  expect(secondary.evidence.events.map(row => row.requested_conditions.v_supply_v)).toEqual([200, 200])
  await expect(result).toContainText('Supply: 600.000 V')
  await expect(result).toContainText('Supply: 400.000 V')
  const saved = JSON.parse(await download(page, 'Download request + result JSON'))
  expect(saved.response).toEqual(JSON.parse(JSON.stringify(response))); expect(saved.request).toEqual(response.request)
  const csv = await download(page, 'Download branch CSV')
  const rows = csv.trim().split(/\r?\n/)
  expect(rows).toHaveLength(5)
  expect(rows[4]).toContain('"s78","s7;s8","2"')
  const charts = result.locator('canvas')
  await expect(charts).toHaveCount(6)
  await page.waitForTimeout(1200)
  for (const chart of await charts.all()) {
    expect(await chart.evaluate(canvas => canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data.some((v, i) => i % 4 === 3 && v > 0))).toBe(true)
  }
  await result.screenshot({ path: test.info().outputPath('independent-dab-secondary.png') })
  expect(errors).toEqual([])
})

test('compiled omitted switching is unknown in view and downloaded CSV', async ({ page }) => {
  await start(page, 'Load LLC omission fixture')
  await page.getByTestId('converter-omission').check()
  const response = await calculate(page)
  expect(response.totals.total_power_w).toBeNull()
  expect(response.totals.known_component_power_w).toBeCloseTo(2040, 10)
  const result = page.getByTestId('converter-results')
  await expect(result).toContainText('Complete total: Unknown W')
  await expect(result).toContainText('not a complete converter loss')
  await page.getByTestId('converter-branch').selectOption('rectifier_hb1')
  await expect(result.locator('canvas')).toHaveCount(3)
  const csv = await download(page, 'Download branch CSV')
  const row = csv.split(/\r?\n/).find(line => line.startsWith('"rectifier_hb1"'))
  expect(row).toContain('"12","","","12","","partial-explicit-omissions"')
  expect(response.branches[2].scaled_components.turn_on.power_w).toBeNull()
})

test('compiled mobile held interval powers and stale successful response', async ({ page }) => {
  const errors = []; page.on('pageerror', error => errors.push(error.message))
  await page.setViewportSize({ width: 390, height: 844 })
  await start(page, 'Load independent envelope fixture')
  await page.getByTestId('converter-envelope').check()
  let release, reached
  const arrived = new Promise(resolve => { reached = resolve })
  const held = new Promise(resolve => { release = resolve })
  await page.route('**/api/converters/analyze', async route => {
    const reply = await route.fetch(); reached(); await held; await route.fulfill({ response: reply })
  })
  await page.getByRole('button', { name: 'Calculate converter losses', exact: true }).click()
  await arrived
  await page.getByTestId('signal-ia').fill('20,20')
  release()
  await expect(page.getByRole('button', { name: 'Calculate converter losses', exact: true })).toBeEnabled()
  await expect(page.getByTestId('converter-results')).toHaveCount(0)
  await expect(page.getByRole('status')).toContainText('Inputs changed')
  await page.unroute('**/api/converters/analyze')
  const response = await calculate(page)
  expect(response.totals.components.conduction.power_w).toBeCloseTo(360, 10)
  const inactive = structuredClone(response.request)
  inactive.preset = 'custom'; inactive.role_models = {}
  inactive.bindings = [{ role: 'absent', members: [], multiplicity: 0,
    conduction_current: { value: 0 }, inactive_reason: '   ' }]
  expect((await page.request.post('http://localhost:8002/api/converters/analyze', { data: inactive })).status()).toBe(422)
  inactive.bindings[0].inactive_reason = 'No physical positions in selected circuit'
  const zero = await page.request.post('http://localhost:8002/api/converters/analyze', { data: inactive })
  expect(zero.status()).toBe(200)
  expect((await zero.json()).totals.total_power_w).toBe(0)
  await page.getByTestId('converter-branch').selectOption('neutral_igbts')
  const result = page.getByTestId('converter-results')
  expect(response.branches[2].evidence.intervals.map(row => row.current_a)).toEqual([60, 60])
  expect(response.branches[2].evidence.intervals.map(row => row.frequency_hz)).toEqual([50, 50])
  await expect(result).toContainText('No resolved PWM event count')
  await expect(result.locator('canvas')).toHaveCount(5)
  await page.waitForTimeout(1200)
  const overflow = await page.evaluate(() => ({ width: document.documentElement.scrollWidth, viewport: innerWidth }))
  await page.screenshot({ path: test.info().outputPath('mobile-envelope-whole-layout.png'), fullPage: true })
  expect(overflow.width, JSON.stringify(overflow)).toBeLessThanOrEqual(overflow.viewport + 1)
  const tables = await result.locator('.table-scroll table').evaluateAll(nodes => nodes.map(table => ({
    width: table.getBoundingClientRect().width, container: table.parentElement.clientWidth,
    overflow: getComputedStyle(table.parentElement).overflowX,
  })))
  for (const table of tables) {
    expect(table.width).toBeGreaterThanOrEqual(650)
    expect(table.container).toBeLessThan(390)
    expect(table.overflow).toBe('auto')
  }
  await result.locator('.table-scroll').first().scrollIntoViewIfNeeded()
  await page.screenshot({ path: test.info().outputPath('independent-converter-readable-mobile-table.png') })
  await result.screenshot({ path: test.info().outputPath('independent-converter-envelope-mobile.png') })
  await page.getByText('Advanced canonical request import / inspection', { exact: true }).click()
  await page.getByTestId('converter-json').fill(JSON.stringify(inactive))
  await page.getByLabel('Use JSON editor as calculation input', { exact: true }).check()
  const absent = await calculate(page)
  expect(absent.totals.total_power_w).toBe(0)
  await expect(page.getByTestId('converter-results')).toContainText('Inactive role: No physical positions in selected circuit')
  expect(errors).toEqual([])
})

test('independent rendered branch switch and linked time bounds retain SI point positions', async ({ page }) => {
  await page.goto('http://localhost:5174/?view=converter')
  await page.getByRole('button', { name: 'Load independent DAB fixture', exact: true }).click()
  await page.getByTestId('converter-symmetry').check()
  const response = await calculate(page)
  await page.getByTestId('converter-branch').selectOption('s34')
  await page.getByTestId('converter-range-start').fill('0')
  await page.getByTestId('converter-range-end').fill('0.000003')
  await page.waitForTimeout(1200)
  const plots = await page.getByTestId('converter-results').locator('.curve-plotter').evaluateAll(nodes => nodes.map(node => {
    let component = node.__vueParentComponent
    while (component && !component.proxy?.chart) component = component.parent
    const chart = component.proxy.chart
    return { min: chart.scales.x.min, max: chart.scales.x.max,
      points: chart.data.datasets.map((series, si) => series.data.map((source, i) => {
        const point = chart.getDatasetMeta(si).data[i]
        return { source, drawnX: chart.scales.x.getValueForPixel(point.x), drawnY: chart.scales.y.getValueForPixel(point.y) }
      })) }
  }))
  expect(plots).toHaveLength(6)
  for (const plot of plots) {
    expect(plot.min).toBe(0); expect(plot.max).toBe(3e-6)
    for (const series of plot.points) for (const point of series) {
      expect(point.drawnX).toBeCloseTo(point.source.x, 12)
      expect(point.drawnY).toBeCloseTo(point.source.y, 8)
    }
  }
  const source = response.branches[1].evidence.switching_signals
  expect(plots[1].points[0].map(p => p.source)).toEqual(source.time_s.map((x, i) => ({ x, y: source.current_a[i] })))
})
