import { expect, test } from '@playwright/test'
import { readFile } from 'node:fs/promises'
import { inspectRenderedChart } from '../fixtures/chartEvidence.js'

const fixtureBase = process.env.RG_ENERGY_FIXTURE_URL || 'http://localhost:8003'
const field = (page, name) => page.getByTestId(`rg-${name}`)
const panel = page => page.getByTestId('rg-energy')
async function open(page, base = fixtureBase) {
  await page.goto(`${base}/index.html?view=analytical&model=parameters#rg-energy`)
  await expect(page.getByRole('button', { name: 'Device Parameter Operations', exact: true })).toHaveClass(/active/)
  await expect(panel(page).getByRole('heading', { name: 'Repository Rg–energy queries' })).toBeInViewport()
  await page.getByRole('link', { name: 'Go to repository Rg–energy queries', exact: true }).click()
  await expect(panel(page).getByRole('heading', { name: 'Repository Rg–energy queries' })).toBeInViewport()
}
async function selectContext(page, gate = 12, method = null) {
  const selector = field(page, 'context')
  await expect(selector).toBeVisible()
  const value = await selector.locator('option').evaluateAll((options, expected) => options.map(option => option.value).find(value => {
    if (!value) return false
    const condition = JSON.parse(value)
    return condition[0] === 400 && condition[1] === expected.gate && condition[8] === expected.method
  }), { gate, method })
  expect(value).toBeTruthy(); await selector.selectOption(value)
}
async function ready(page, device = 'Browser Rg energy') {
  await open(page); await field(page, 'device').selectOption(device)
  await expect(field(page, 'context')).toHaveValue('')
  await selectContext(page)
  await field(page, 'current').fill('10'); await field(page, 'temperature').fill('75'); await field(page, 'query').fill('1.5')
}
async function calculate(page, status = 200) {
  const waiting = page.waitForResponse(response => response.url().endsWith('/rg-energy') && response.request().method() === 'POST')
  await field(page, 'calculate').click()
  const response = await waiting, raw = await response.text()
  expect(response.status(), raw).toBe(status)
  if (status === 200) await expect(field(page, 'results')).toBeVisible()
  else { await expect(field(page, 'error')).toBeVisible(); await expect(field(page, 'results')).toHaveCount(0) }
  return { data: JSON.parse(raw), raw }
}
async function download(page, name) {
  const waiting = page.waitForEvent('download'); await field(page, name).click()
  const file = await waiting; expect(file.suggestedFilename()).toBe(`rg-energy.${name}`)
  return readFile(await file.path(), 'utf8')
}
for (const width of [1440, 390]) {
  test(`compiled repository queries, real bounded estimates and portable proof at ${width}px`, async ({ page }, info) => {
    const errors = []; page.on('pageerror', error => errors.push(error.message)); page.on('console', message => { if (message.type() === 'error') errors.push(message.text()) })
    await page.setViewportSize({ width, height: 900 }); await ready(page)
    const forward = await calculate(page)
    expect(forward.data.value).toBeCloseTo(.007, 12); expect(forward.data.unit).toBe('J')
    expect(forward.data.query.conditions.v_g).toBe(12); expect(forward.data.query.conditions.method).toBe(null)
    expect(forward.data.sources).toHaveLength(3); expect(forward.data.source_scope).toBe('considered-branch')
    expect(forward.data.sources[0].source_kind).toBe('repository-locator'); expect(forward.data.sources[0].source_sha256).toBe(null)
    expect(forward.data.sources[0].snapshot_sha256).toMatch(/^[a-f0-9]{64}$/)
    expect(await download(page, 'json')).toBe(forward.raw)
    const csv = await download(page, 'csv')
    expect(csv).toContain('accepted_conditions'); expect(csv).toContain('query'); expect(csv).toContain('source_scope'); expect(csv).toContain(forward.data.sources[0].snapshot_sha256)
    await page.waitForTimeout(1200)
    const plotNode = panel(page).locator('.curve-plotter'), plot = await plotNode.evaluate(inspectRenderedChart)
    expect(plot.datasets).toHaveLength(2); expect(plot.datasets[0].points.map(point => point.input)).toEqual([{ x: 1, y: .008 }, { x: 4, y: .002 }, { x: 9, y: .006 }])
    expect(plot.datasets.at(-1).points[0].input.x).toBe(1.5); expect(plot.datasets.at(-1).points[0].input.y).toBeCloseTo(.007, 12)
    for (const dataset of plot.datasets) for (const point of dataset.points) {
      expect(point.value).toBeCloseTo(point.input.y, 8)
      const radius = point.radius + point.borderWidth / 2
      expect(point.pixel.x - radius).toBeGreaterThan(plot.chartArea.left); expect(point.pixel.x + radius).toBeLessThan(plot.chartArea.right)
      expect(point.pixel.y - radius).toBeGreaterThan(plot.chartArea.top); expect(point.pixel.y + radius).toBeLessThan(plot.chartArea.bottom)
    }
    await field(page, 'results').screenshot({ path: info.outputPath(`rg-energy-result-${width}.png`) })
    await plotNode.evaluate(node => node.scrollIntoView({ block: 'center' })); await plotNode.screenshot({ path: info.outputPath(`rg-energy-plot-${width}.png`) })
    await field(page, 'operation').selectOption('rg-at-energy'); await expect(field(page, 'results')).toHaveCount(0)
    await field(page, 'query').fill('.007'); const inverse = await calculate(page)
    expect(inverse.data.value).toBeCloseTo(1.5, 12); expect(inverse.data.unit).toBe('ohm')
    await field(page, 'representation').selectOption('graph_i_e'); await expect(field(page, 'results')).toHaveCount(0); await expect(field(page, 'context')).toHaveValue('')
    await selectContext(page); await field(page, 'query').fill('.008'); const currentInverse = await calculate(page)
    expect(currentInverse.data.value).toBeCloseTo(5, 12); expect(currentInverse.data.unit).toBe('ohm')
    await expect(panel(page)).toContainText('Forward queries require an exact recorded Rg')
    await field(page, 'operation').selectOption('energy-at-rg'); await field(page, 'query').fill('9'); expect((await calculate(page)).data.value).toBeCloseTo(.004, 12)
    await field(page, 'branch').selectOption('e_off'); await field(page, 'representation').selectOption('graph_r_e'); await selectContext(page, 12, 'unknown')
    await expect(field(page, 'conditions')).toContainText('"unknown"'); await expect(panel(page)).toContainText('recorded.csv'); await field(page, 'query').fill('9')
    const zero = await calculate(page); expect(zero.data.value).toBe(0); expect(zero.data.query.conditions.method).toBe('unknown'); expect(zero.data.sources[0].source_sha256).toBe('a'.repeat(64))
    await expect(field(page, 'value')).toHaveText('0.00000 J'); expect(await download(page, 'json')).toBe(zero.raw)
    await field(page, 'branch').selectOption('e_rr'); await selectContext(page, null); await field(page, 'query').fill('9')
    const unknown = await calculate(page); expect(unknown.data.query.conditions.v_g).toBe(null); expect(unknown.data.value).toBe(0)
    await selectContext(page, 0); await expect(field(page, 'results')).toHaveCount(0)
    const knownZero = await calculate(page); expect(knownZero.data.query.conditions.v_g).toBe(0); expect(knownZero.data.value).toBeCloseTo(.006, 12)
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(width + 1)
    expect(errors).toEqual([])
  })
}

test('core ambiguity, exact recorded Rg, duplicate curves, inverse holes and stale context responses', async ({ page }) => {
  const errors = []; page.on('pageerror', error => errors.push(error.message))
  await ready(page); await field(page, 'operation').selectOption('rg-at-energy'); await field(page, 'query').fill('.004')
  const ambiguous = await calculate(page, 422); expect(ambiguous.data.detail).toMatch(/multiple|unique|ambiguous/i); await expect(field(page, 'json')).toHaveCount(0)
  await field(page, 'representation').selectOption('graph_i_e'); await selectContext(page); await field(page, 'operation').selectOption('energy-at-rg'); await field(page, 'query').fill('5')
  const unrecorded = await calculate(page, 422); expect(unrecorded.data.detail).toMatch(/gate resistance|r_g|recorded/i)
  await field(page, 'device').selectOption('Browser Rg duplicate'); await field(page, 'representation').selectOption('graph_r_e'); await selectContext(page); await field(page, 'query').fill('1.5')
  await expect(panel(page).getByRole('table')).toContainText('switch.e_on_data[3]')
  expect((await calculate(page, 422)).data.detail).toMatch(/unambiguous|ambiguous/i)
  await field(page, 'device').selectOption('Browser Rg current hole'); await field(page, 'representation').selectOption('graph_i_e'); await selectContext(page)
  await field(page, 'operation').selectOption('rg-at-energy'); await field(page, 'query').fill('.008')
  expect((await calculate(page, 422)).data.detail).toMatch(/supports|current|coverage/i)
  await field(page, 'device').selectOption('Browser Rg energy'); await selectContext(page)
  let release; const held = new Promise(resolve => { release = resolve })
  await page.route('**/api/transistors/*/rg-energy', async route => {
    if (route.request().method() !== 'POST') return route.continue()
    const response = await route.fetch(); await held; await route.fulfill({ response })
  })
  const sent = page.waitForRequest(request => request.url().endsWith('/rg-energy') && request.method() === 'POST')
  await field(page, 'calculate').click(); await sent; await field(page, 'context').selectOption(''); release()
  await page.waitForTimeout(200); await expect(field(page, 'results')).toHaveCount(0); await expect(field(page, 'json')).toHaveCount(0)
  await selectContext(page); await field(page, 'query').fill('1e-999'); await field(page, 'calculate').click(); await expect(field(page, 'error')).toContainText('too small')
  expect(errors).toEqual([])
})

test('an unavailable record at another temperature remains in repository and result proof', async ({ page }, info) => {
  const errors = []; page.on('pageerror', error => errors.push(error.message)); page.on('console', message => { if (message.type() === 'error') errors.push(message.text()) })
  await ready(page, 'Browser Rg invalid other temperature')
  await expect(field(page, 'unavailable-records')).toBeVisible()
  await expect(field(page, 'unavailable-records')).toContainText('switch.e_on_data[3]')
  const response = await calculate(page)
  expect(response.data.value).toBeCloseTo(.007, 12)
  const unavailable = response.data.sources.find(source => source.status === 'unavailable')
  expect(unavailable.t_j).toBe(100); expect(unavailable.snapshot).toBe(null); expect(unavailable.snapshot_sha256).toBe(null)
  expect(unavailable.preview).toBe(null); expect(unavailable.coverage).toBe(null); expect(unavailable.diagnostic).toBeTruthy()
  await expect(panel(page).getByRole('table')).toContainText(unavailable.diagnostic)
  await expect(panel(page).getByRole('table')).toContainText('Unavailable')
  await page.waitForTimeout(1200)
  const plot = await panel(page).locator('.curve-plotter').evaluate(inspectRenderedChart)
  expect(plot.datasets).toHaveLength(2)
  expect(await download(page, 'json')).toBe(response.raw)
  const csv = await download(page, 'csv'); expect(csv).toContain(unavailable.source_id); expect(csv).toContain('unavailable_fields')
  await field(page, 'unavailable-records').screenshot({ path: info.outputPath('rg-energy-unavailable-record.png') })
  expect(errors).toEqual([])
})

test('live repository inventory is separate from private numerical fixtures', async ({ page, request }) => {
  const errors = []; page.on('pageerror', error => errors.push(error.message))
  await open(page, 'http://localhost:8002'); await expect(field(page, 'device')).toHaveValue('')
  const response = await request.get('http://localhost:8002/api/transistors'), devices = await response.json()
  expect(response.status()).toBe(200); expect(devices.some(device => (device.id ?? device.metadata.name) === 'Browser Rg energy')).toBe(false)
  expect(devices.length).toBeGreaterThan(0)
  const device = devices[0], deviceId = device.id ?? device.metadata.name, waiting = page.waitForResponse(response => response.url().endsWith('/rg-energy'))
  await field(page, 'device').selectOption(deviceId); const inventoryResponse = await waiting
  expect(inventoryResponse.status(), await inventoryResponse.text()).toBe(200)
  const inventory = await inventoryResponse.json()
  expect(inventory.device_id).toBe(deviceId)
  for (const curve of inventory.curves) {
    if (curve.status === 'unavailable') { expect(curve.snapshot_sha256).toBe(null); expect(curve.diagnostic).toBeTruthy() }
    else expect(curve.snapshot_sha256).toMatch(/^[a-f0-9]{64}$/)
  }
  if (inventory.curves.some(curve => curve.branch === 'e_on' && curve.representation === 'graph_r_e' && curve.status !== 'unavailable' && curve.conditions)) await expect(field(page, 'context')).toHaveValue('')
  else await expect(panel(page)).toContainText(/No recorded|No selectable/)
  await expect(field(page, 'results')).toHaveCount(0)
  expect(errors).toEqual([])
})
