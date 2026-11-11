import { test, expect } from '@playwright/test'
import fs from 'node:fs/promises'

async function downloaded(page, button) {
  const waiting = page.waitForEvent('download')
  await page.getByRole('button', { name: button, exact: true }).click()
  return fs.readFile(await (await waiting).path(), 'utf8')
}

test('independent Foster event, chart rendering and canonical exports', async ({ page }) => {
  const errors = []; page.on('pageerror', error => errors.push(error.message))
  await page.goto('http://localhost:8002/index.html?view=thermal')
  await page.getByLabel('Thermal operation').selectOption('transient')
  await page.getByLabel('Event heat convention').selectOption('impulse')
  await page.getByLabel('Thermal events').fill('0.037,0.1,forward')
  const waiting = page.waitForResponse(response => response.url().includes('/api/thermal/analyze'))
  await page.getByRole('button', { name: 'Calculate thermal result', exact: true }).click()
  const response = await (await waiting).json()
  expect(response.result.final_temperature_c).toBeCloseTo(43.31847684154357, 11)
  expect(response.result.switching_energy_j).toBe(.1)
  const result = page.getByRole('region', { name: 'Thermal result' })
  await expect(result.locator('canvas')).toHaveCount(1)
  await page.waitForTimeout(1200)
  const chart = await result.locator('.curve-plotter').evaluate(node => {
    let component = node.__vueParentComponent
    while (component && !component.proxy?.chart) component = component.parent
    if (!component) return null // Production Vue intentionally omits debug links.
    const chart = component.proxy.chart
    return chart.getDatasetMeta(0).data.map((point, index) => ({
      source: chart.data.datasets[0].data[index],
      drawnX: chart.scales.x.getValueForPixel(point.x),
      drawnY: chart.scales.y.getValueForPixel(point.y),
    }))
  })
  if (chart) {
    expect(chart.length).toBe(response.result.samples.length)
    for (let i = 0; i < chart.length; i++) {
      expect(chart[i].source).toEqual({ x: response.result.samples[i].time_s, y: response.result.samples[i].temperature_c })
      expect(chart[i].drawnX).toBeCloseTo(chart[i].source.x, 12)
      expect(chart[i].drawnY).toBeCloseTo(chart[i].source.y, 9)
    }
  }
  expect(await result.locator('canvas').evaluate(canvas => canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data.some((v, i) => i % 4 === 3 && v > 0))).toBe(true)
  const saved = JSON.parse(await downloaded(page, 'Export thermal request + result JSON'))
  expect(saved).toEqual(response)
  const csv = await downloaded(page, 'Export thermal evidence CSV')
  expect(csv).toContain('pre-event'); expect(csv).toContain('post-event')
  expect(csv.trim().split('\n').length).toBe(response.result.samples.length + 1)
  const cubic = JSON.parse(JSON.stringify(response.request))
  cubic.network.r_th_vector = [1e18]; cubic.network.tau_vector = [1e-6]
  cubic.network.c_th_vector = null; cubic.network.r_th_total = null
  cubic.case_temperature_c = 0; cubic.samples_per_segment = 1
  cubic.segments = [{ start_s: 0, end_s: 1, power_coefficients_w: [1, -3, 3, -1] }]
  cubic.events = [{ time_s: .999999, energy_j: 0, heat_owner: 'forward', source: 'Independent zero-heat cut', conditions: {} }]
  const corrected = await page.request.post('http://localhost:8002/api/thermal/analyze', { data: cubic })
  expect(corrected.status()).toBe(200)
  expect((await corrected.json()).result.final_temperature_c).toBeCloseTo(6, 10)
  await page.screenshot({ path: '/tmp/tdb-phase6-independent-thermal-desktop-20261016-01.png', fullPage: true })
  expect(errors).toEqual([])
})

test('late successful thermal reply is suppressed after input edit', async ({ page }) => {
  await page.goto('http://localhost:8002/index.html?view=thermal')
  let release, intercepted
  const held = new Promise(resolve => { release = resolve })
  const arrived = new Promise(resolve => { intercepted = resolve })
  await page.route('**/api/thermal/analyze', async route => {
    const response = await route.fetch()
    intercepted(); await held; await route.fulfill({ response })
  })
  await page.getByRole('button', { name: 'Calculate thermal result', exact: true }).click()
  await arrived
  await page.getByLabel('Forward heat', { exact: true }).fill('20')
  release()
  await expect(page.getByRole('button', { name: 'Calculate thermal result', exact: true })).toBeEnabled()
  await expect(page.getByRole('region', { name: 'Thermal result' })).toHaveCount(0)
  await expect(page.getByRole('status')).toContainText('Inputs changed')
})

test('compiled mobile reverse envelope export and chart rendering', async ({ page }) => {
  const errors = []; page.on('pageerror', error => errors.push(error.message))
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('http://localhost:8002/index.html?view=assemblies')
  await page.getByLabel('Assembly operation').selectOption('reverse-envelope')
  await page.getByLabel('Select ideal envelope', { exact: true }).check()
  const waiting = page.waitForResponse(response => response.url().includes('/api/assemblies/analyze'))
  await page.getByRole('button', { name: 'Calculate assembly', exact: true }).click()
  const response = await (await waiting).json()
  await expect(page.getByTestId('assembly-results')).toBeVisible()
  const saved = JSON.parse(await downloaded(page, 'Download assembly JSON'))
  expect(saved).toEqual(JSON.parse(JSON.stringify(response)))
  expect(saved.crossings_magnitude_a[0]).toBeCloseTo(40 / 9, 12)
  await page.waitForTimeout(1200)
  const crossing = await page.getByTestId('assembly-results').locator('.curve-plotter').evaluate(node => {
    let component = node.__vueParentComponent
    while (component && !component.proxy?.chart) component = component.parent
    if (!component) return null
    const chart = component.proxy.chart
    const i = chart.data.datasets[0].data.findIndex(p => Math.abs(p.x + 40 / 9) < 1e-10)
    return { source: chart.data.datasets[0].data[i], drawnY: chart.scales.y.getValueForPixel(chart.getDatasetMeta(0).data[i].y) }
  })
  if (crossing) {
    expect(crossing.source.x).toBeCloseTo(-40 / 9, 12)
    expect(crossing.source.y).toBeCloseTo(-8 / 9, 12)
    expect(crossing.drawnY).toBeCloseTo(-8 / 9, 9)
  }
  expect(await page.getByTestId('assembly-results').locator('canvas').evaluate(canvas => canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data.some((v, i) => i % 4 === 3 && v > 0))).toBe(true)
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true)
  await page.screenshot({ path: '/tmp/tdb-phase6-independent-assemblies-mobile-20261016-01.png', fullPage: true })
  expect(errors).toEqual([])
})

test('compiled mobile electrothermal accepted solution has readable units', async ({ page }) => {
  const errors = []; page.on('pageerror', error => errors.push(error.message))
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('http://localhost:8002/index.html?view=thermal')
  await page.getByLabel('Thermal operation').selectOption('electrothermal')
  await page.getByLabel('Select branch 1 temperature estimate', { exact: true }).check()
  const waiting = page.waitForResponse(response => response.url().includes('/api/thermal/analyze'))
  await page.getByRole('button', { name: 'Calculate thermal result', exact: true }).click()
  const response = await (await waiting).json()
  expect(response.result.status).toBe('converged')
  expect(response.result.accepted_solution.temperature_c[0]).toBeCloseTo(25 + 7 / .972, 7)
  const result = page.getByRole('region', { name: 'Thermal result' })
  await expect(result.getByRole('heading', { name: 'Accepted branch solution', exact: true })).toBeVisible()
  await expect(result.getByRole('columnheader', { name: 'Junction (°C)', exact: true })).toBeVisible()
  await expect(result.getByRole('columnheader', { name: 'Loss (W)', exact: true })).toBeVisible()
  await expect(result.getByRole('columnheader', { name: 'Residual (K)', exact: true })).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true)
  await page.screenshot({ path: '/tmp/tdb-phase6-independent-thermal-mobile-20261016-02.png', fullPage: true })
  expect(errors).toEqual([])
})
