import { expect, test } from '@playwright/test'
import { readFile } from 'node:fs/promises'

const field = (page, name) => page.getByTestId(`stress-${name}`)
const serverTexts = new WeakMap()
const diagnostics = new WeakMap()
test.beforeEach(async ({ page }) => {
  const events = { console: [], page: [] }; diagnostics.set(page, events)
  page.on('console', message => { if (message.type() === 'error') events.console.push(message.text()) })
  page.on('pageerror', error => events.page.push(error.message))
})
test.afterEach(async ({ page }, info) => {
  const events = diagnostics.get(page)
  expect(events.page).toEqual([])
  if (info.title.startsWith('malformed inputs')) {
    expect(events.console.every(message => message.includes('422') || message.startsWith('[API Error]'))).toBe(true)
  } else expect(events.console).toEqual([])
})
async function calculate(page) {
  const waiting = page.waitForResponse(r => r.url().includes('/api/stress/calculate') && r.request().method() === 'POST')
  await field(page, 'calculate').click()
  const response = await waiting
  expect(response.status()).toBe(200)
  const rawJson = await response.text(), data = JSON.parse(rawJson)
  serverTexts.set(data, rawJson)
  await expect(field(page, 'results')).toBeVisible()
  return data
}
async function download(page, name) {
  const waiting = page.waitForEvent('download')
  await field(page, name).click()
  return readFile(await (await waiting).path(), 'utf8')
}

for (const width of [1440, 390]) {
  test(`stress fixture exposes exact Python proxies, provenance and downloads at ${width}px`, async ({ page }, info) => {
    const errors = []; page.on('pageerror', error => errors.push(error.message))
    await page.setViewportSize({ width, height: 844 }); await page.goto('/index.html')
    await page.getByRole('button', { name: 'More Tools' }).click()
    await page.getByRole('button', { name: 'Semiconductor stress indices', exact: true }).click()
    await expect(field(page, 'workbench')).toBeVisible(); await field(page, 'example').click()
    await field(page, 'source').fill('=Independent stress fixture. No measured device data.')
    await field(page, 'condition-junction_temperature_c').fill('25')
    const data = await calculate(page)
    expect(data.inputs).toMatchObject({ time_s: [0, 1], current_a: [0, 6], voltage_v: [400, 400], rds_on_ohm: .1, conditions: { junction_temperature_c: 25, gate_on_voltage_v: null }, transitions: [{ time_s: .5, frequency_hz: 1000, kind: 'turn_on' }] })
    expect(data.result.current_mean_square_a2).toBeCloseTo(12, 12)
    expect(data.result.current_rms_a).toBeCloseTo(Math.sqrt(12), 12)
    expect(data.result.metrics.conduction.value).toBeCloseTo(12, 12)
    expect(data.result.metrics.volume.value).toBeCloseTo(1920000, 7)
    expect(data.result.metrics.switching.value).toBeCloseTo(3600000, 7)
    expect(data.result.metrics.cost.value).toBeCloseTo(.0000625, 12)
    await expect(field(page, 'metric-conduction')).toContainText('12.0000 A²')
    await expect(field(page, 'metric-volume')).toContainText('1.92000e+6 A² V²')
    await expect(field(page, 'metric-switching')).toContainText('3.60000e+6 Hz A² V')
    await expect(field(page, 'metric-cost')).toContainText('0.0000625000 1/(ohm V²)')
    expect(data.result.transition_samples[0]).toMatchObject({ current_a: 3, voltage_v: 400, contribution: 3600000 })
    for (const key of ['input_sha256', 'result_sha256']) expect(data.evidence[key]).toMatch(/^[a-f0-9]{64}$/)
    const raw = await download(page, 'json'); expect(JSON.parse(raw)).toEqual(data)
    expect(raw).toBe(serverTexts.get(data))
    const csv = await download(page, 'csv'); expect(csv).toContain('"\'=Independent stress fixture. No measured device data."'); expect(csv).toContain(data.evidence.result_sha256)
    expect(csv).toContain('"volume","1920000","A^2 V^2","calculated"')
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(width + 1)
    await field(page, 'metric-conduction').scrollIntoViewIfNeeded(); await page.screenshot({ path: info.outputPath(`stress-results-${width}.png`), fullPage: true })
    await field(page, 'source').fill('Edited source'); await expect(field(page, 'results')).toHaveCount(0); await expect(field(page, 'json')).toHaveCount(0)
    expect(errors).toEqual([])
  })
}

test('unknown evidence, explicit absence and signed voltage retain distinct numerical meanings', async ({ page }) => {
  await page.goto('/index.html?view=stress-indices'); await field(page, 'example').click()
  await field(page, 'resistance').fill(''); await field(page, 'transition-mode').selectOption('unknown')
  let data = await calculate(page)
  expect(data.inputs.transitions).toBeNull(); expect(data.result.metrics.switching.value).toBeNull(); expect(data.result.metrics.cost.value).toBeNull()
  await expect(field(page, 'metric-switching')).toContainText('Unavailable'); await expect(field(page, 'metric-cost')).toContainText('Unavailable')
  await field(page, 'transition-mode').selectOption('none'); data = await calculate(page)
  expect(data.inputs.transitions).toEqual([]); expect(data.result.metrics.switching.value).toBe(0)
  await expect(field(page, 'metric-switching')).toContainText('0.00000')
  await field(page, 'samples').fill('time_s,current_a,voltage_v\n0,-3,-400\n1,-3,-200')
  await field(page, 'transition-mode').selectOption('provided'); await field(page, 'resistance').fill('.1'); data = await calculate(page)
  expect(data.result.peak_voltage_v).toBe(-200); expect(data.result.metrics.switching.value).toBe(-2700000)
  expect(data.result.metrics.cost.value).toBe(.00025)
  await expect(field(page, 'metric-switching')).toContainText('-2.70000e+6')
})

test('malformed inputs and out-of-bounds events surface validation without retaining exports', async ({ page }) => {
  await page.goto('/index.html?view=stress-indices'); await field(page, 'example').click()
  await field(page, 'samples').fill('time_s,current_a,voltage_v\n0,,400\n1,6,400'); await field(page, 'calculate').click()
  await expect(page.getByRole('alert')).toContainText('finite decimal value')
  await field(page, 'example').click(); await field(page, 'transitions').fill('time_s,frequency_hz,kind'); await field(page, 'calculate').click()
  await expect(page.getByRole('alert')).toContainText('at least one transition record')
  await field(page, 'example').click(); await field(page, 'samples').fill('time_s,current_a,voltage_v\n0,1e-400,400\n1,6,400'); await field(page, 'calculate').click()
  await expect(page.getByRole('alert')).toContainText('too small')
  await field(page, 'example').click(); await field(page, 'transitions').fill('time_s,frequency_hz,kind\n2,1000,turn_on')
  const waiting = page.waitForResponse(r => r.url().includes('/api/stress/calculate'))
  await field(page, 'calculate').click(); expect((await waiting).status()).toBe(422)
  await expect(page.getByRole('alert')).toContainText('interval'); await expect(field(page, 'results')).toHaveCount(0)
  await field(page, 'example').click(); await field(page, 'samples').fill('time_s,current_a,voltage_v\n0,0,400\n0,6,400')
  const badTime = page.waitForResponse(r => r.url().includes('/api/stress/calculate'))
  await field(page, 'calculate').click(); expect((await badTime).status()).toBe(422)
  await expect(page.getByRole('alert')).toContainText('increase strictly')
})

test('a held successful response cannot restore stale results or downloads after an edit', async ({ page }) => {
  await page.goto('/index.html?view=stress-indices'); await field(page, 'example').click()
  let release, fetched
  const held = new Promise(resolve => { release = resolve }), ready = new Promise(resolve => { fetched = resolve })
  await page.route('**/api/stress/calculate', async route => { const response = await route.fetch(); fetched(); await held; await route.fulfill({ response }) })
  await field(page, 'calculate').click(); await ready; await field(page, 'condition-gate_on_voltage_v').fill('15'); release()
  await expect(field(page, 'stale')).toBeVisible(); await expect(field(page, 'results')).toHaveCount(0); await expect(field(page, 'json')).toHaveCount(0)
  await expect(field(page, 'calculate')).toBeEnabled()
})
