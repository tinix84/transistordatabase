import { test, expect } from '@playwright/test'
import { readFile, writeFile } from 'node:fs/promises'
const synthetic = 'Verifier_Synthetic_Electrothermal'
const endpoint = '/api/thermal/device/analyze'
async function attach(info, name, value) { await info.attach(name, { body: JSON.stringify(value, null, 2), contentType: 'application/json' }) }
async function inventory(page, name) {
  await page.getByTestId('native-device-id').fill(name)
  const ready = page.waitForResponse(r => r.url().includes('/api/thermal/device/inventory/') && r.request().method() === 'GET')
  await page.getByTestId('native-inventory-refresh').click()
  const response = await ready
  expect(response.status()).toBe(200)
  return response.json()
}
async function fields(page, changes = {}) {
  const values = { current: '2', duty: '1', frequency: '0', sink: '25', resistance: '2', initial: '25',
    tolerance: '1e-8', iterations: '200', relaxation: '1', ...changes }
  for (const [name, value] of Object.entries(values)) await page.getByTestId(`native-${name}`).fill(String(value))
  await page.getByTestId('native-thermal-source').fill('Independent synthetic fixed-sink verification; not measured-device accuracy')
  await page.getByTestId('native-estimate-selected').check()
}
async function calculate(page) {
  const ready = page.waitForResponse(r => r.url().endsWith(endpoint) && r.request().method() === 'POST')
  await page.getByTestId('native-calculate').click()
  const response = await ready
  return { status: response.status(), body: await response.json(), raw_request: response.request().postData() }
}
async function download(page, id, info, name) {
  const ready = page.waitForEvent('download')
  await page.getByTestId(id).click()
  const item = await ready, path = info.outputPath(name)
  await item.saveAs(path)
  return readFile(path, 'utf8')
}
function close(value, expected, tolerance = 1e-7) { expect(Math.abs(value - expected)).toBeLessThan(tolerance) }
test.beforeEach(async ({ page }, info) => {
  info.capture = { console: [], pageerrors: [], failed_requests: [], http5xx: [], calculation_posts: 0 }
  page.on('console', item => info.capture.console.push({ type: item.type(), text: item.text() }))
  page.on('pageerror', error => info.capture.pageerrors.push(error.message))
  page.on('requestfailed', item => info.capture.failed_requests.push({ url: item.url(), failure: item.failure() }))
  page.on('request', item => { if (item.url().endsWith(endpoint) && item.method() === 'POST') info.capture.calculation_posts++ })
  page.on('response', item => { if (item.status() >= 500) info.capture.http5xx.push({ url: item.url(), status: item.status() }) })
  await page.goto('/index.html?view=thermal')
  await page.getByLabel('Thermal operation', { exact: true }).selectOption('native-device')
  await expect(page.getByTestId('native-device-panel')).toBeVisible()
})
test.afterEach(async ({ page }, info) => {
  await attach(info, 'browser-console-network', info.capture)
  expect(info.capture.pageerrors).toEqual([])
  expect(info.capture.failed_requests).toEqual([])
  expect(info.capture.http5xx).toEqual([])
  if (info.project.name === 'mobile-390') expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(391)
})

test('native device acceptance synthetic affine root stable dependency exports and stale invalidation', async ({ page }, info) => {
  const source = await inventory(page, synthetic)
  expect(source.repository_source).toBeNull()
  await fields(page)
  await page.getByTestId('native-group-channel-0').click()
  let response = await calculate(page)
  expect(response.status).toBe(200)
  const data = response.body, accepted = data.result.accepted_solution
  expect(data.result.status).toBe('converged')
  close(accepted.temperature_c[0], 25 + 8 / .92)
  close(accepted.power_w[0], 4 / .92)
  expect(Math.abs(accepted.residual_k[0])).toBeLessThanOrEqual(1e-8)
  expect(data.table.total_power_w).toEqual([4, 8])
  expect(data.table.temperature_domain_c).toEqual([25, 125])
  expect(data.table.active_contributions).toEqual({ channel: true, turn_on: false, turn_off: false })
  expect(data.table.source_identity.repository_source).toBeNull()
  expect(data.table.source_identity.selected_dependency_sha256).toMatch(/^[0-9a-f]{64}$/)
  await expect(page.getByTestId('native-accepted')).toContainText('33.6957')
  expect(JSON.parse(await download(page, 'native-thermal-json', info, 'synthetic-affine-evidence.json'))).toEqual(data)
  const csv = await download(page, 'native-thermal-csv', info, 'synthetic-affine-losses.csv')
  expect(csv.trim().split(/\r?\n/)).toHaveLength(3)
  for (const column of ['temperature_c', 'total_power_w', 'selected_dependency_sha256', 'repository_file_sha256']) expect(csv).toContain(column)
  expect(csv).not.toContain('record_evidence')
  response = await calculate(page)
  expect(response.status).toBe(200)
  expect(response.body.table.source_identity.selected_dependency_sha256).toBe(data.table.source_identity.selected_dependency_sha256)
  await attach(info, 'synthetic-affine-real-api', data)
  await page.screenshot({ path: info.outputPath('synthetic-affine-root.png'), fullPage: true })
  await page.getByTestId('native-current').fill('3')
  await expect(page.getByTestId('native-thermal-result')).toHaveCount(0)
  await expect(page.getByTestId('native-thermal-json')).toHaveCount(0)
  await expect(page.getByRole('status').filter({ hasText: 'Inputs or source selection changed' })).toBeVisible()
})

test('native device acceptance active zero-current events retain positive heat and reject missing E0 coverage', async ({ page }, info) => {
  await inventory(page, synthetic)
  await fields(page, { current: '0', duty: '.5', frequency: '1000' })
  for (const family of ['turn_on', 'turn_off']) for (const index of [0, 1]) await page.getByTestId(`native-record-${family}-${index}`).check()
  const response = await calculate(page)
  expect(response.status).toBe(200)
  const data = response.body
  expect(data.table.active_contributions).toEqual({ channel: false, turn_on: true, turn_off: true })
  expect(data.table.total_power_w).toEqual([5, 5])
  close(data.result.accepted_solution.temperature_c[0], 35)
  expect(data.accepted_component_power_w.channel).toBe(0)
  expect(data.accepted_component_power_w.turn_on).toBe(2)
  expect(data.accepted_component_power_w.turn_off).toBe(3)
  const off = data.table.record_evidence.find(row => row.family === 'turn_off')
  expect(off.recorded_conditions.v_g).toBe(-15)
  expect(off.recorded_conditions.v_g_off).toBeNull()
  expect(off.recorded_conditions.method).toBe('analytic-fixture')
  expect(JSON.parse(await download(page, 'native-thermal-json', info, 'synthetic-E0-evidence.json'))).toEqual(data)
  await attach(info, 'synthetic-E0-real-api', data)
  await page.screenshot({ path: info.outputPath('synthetic-active-E0.png'), fullPage: true })
  for (const index of [0, 1]) await page.getByTestId(`native-record-turn_on-${index}`).uncheck()
  for (const index of [2, 3]) await page.getByTestId(`native-record-turn_on-${index}`).check()
  const unsupported = await calculate(page)
  expect(unsupported.status).toBe(422)
  await expect(page.getByRole('alert')).toBeVisible()
  await expect(page.getByTestId('native-thermal-result')).toHaveCount(0)
})

test('native device acceptance domain exit iteration limit and original numeric boundaries remain explicit', async ({ page }, info) => {
  await inventory(page, synthetic)
  await fields(page, { sink: '125', initial: '125' })
  await page.getByTestId('native-group-channel-0').click()
  const exited = await calculate(page)
  expect(exited.status).toBe(200)
  expect(exited.body.result.status).toBe('source-domain-exit')
  expect(exited.body.result.accepted_solution).toBeNull()
  await expect(page.getByTestId('native-no-solution')).toContainText('last iterate is not a physical solution')
  await fields(page, { iterations: '1' })
  const limited = await calculate(page)
  expect(limited.status).toBe(200)
  expect(limited.body.result.status).toBe('iteration-limit')
  expect(limited.body.result.accepted_solution).toBeNull()
  let posts = info.capture.calculation_posts
  for (const [name, value] of [['duty', '0.999999999999999999999'], ['duty', '1.000000000000000000001'], ['current', '']]) {
    await page.getByTestId(`native-${name}`).fill(value)
    await page.getByTestId('native-calculate').click()
    await expect(page.getByRole('alert')).toBeVisible()
    expect(info.capture.calculation_posts).toBe(posts)
    await expect(page.getByTestId('native-thermal-result')).toHaveCount(0)
  }
  const base = limited.raw_request
  const rawResponses = []
  for (const raw of [base.replace('"duty":1', '"duty":0.999999999999999999999'),
    base.replace('"resistance_k_w":2', '"resistance_k_w":-1e-400')]) {
    const response = await page.evaluate(async raw => {
      const result = await fetch('/api/thermal/device/analyze', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: raw })
      return { status: result.status, body: await result.json(), raw_request: raw }
    }, raw)
    expect(response.status).toBe(422)
    rawResponses.push(response)
  }
  await attach(info, 'synthetic-status-and-raw-boundaries', { exited, limited, rawResponses })
})

test('native device acceptance actual copied Fuji oracle raw identity and stale-source rejection', async ({ page }, info) => {
  const id = 'Fuji_2MBI100XAA120-50'
  const source = await inventory(page, id)
  expect(source.repository_source.kind).toBe('repository-file')
  expect(source.repository_source.sha256).toMatch(/^[0-9a-f]{64}$/)
  await fields(page, { current: '50', duty: '.5', frequency: '1000', sink: '45', resistance: '.1', initial: '45' })
  for (const family of ['channel', 'turn_on', 'turn_off']) await page.getByTestId(`native-group-${family}-0`).click()
  const response = await calculate(page)
  expect(response.status).toBe(200)
  const data = response.body
  expect(data.result.status).toBe('converged')
  close(data.result.accepted_solution.temperature_c[0], 48.763050559454264)
  close(data.result.accepted_solution.power_w[0], 37.63050559454264)
  expect(data.table.source_identity.repository_source.sha256).toBe(source.repository_source.sha256)
  expect(data.table.record_evidence.filter(row => row.family === 'turn_off').every(row => row.recorded_conditions.v_g === -15 && row.recorded_conditions.v_g_off === null)).toBe(true)
  // Full native responses and downloads stay in the external audit, never attached to the public reporter.
  await writeFile(info.outputPath('actual-native-response-private.json'), JSON.stringify(data, null, 2), { flag: 'wx' })
  expect(JSON.parse(await download(page, 'native-thermal-json', info, 'actual-native-export-private.json'))).toEqual(data)
  const nativeCsv = await download(page, 'native-thermal-csv', info, 'actual-native-losses-private.csv')
  expect(nativeCsv).toContain(source.repository_source.sha256)
  expect(nativeCsv.trim().split(/\r?\n/)).toHaveLength(5)
  const request = JSON.parse(response.raw_request)
  request.expected_file_sha256 = '0'.repeat(64)
  const stale = await page.evaluate(async request => {
    const result = await fetch('/api/thermal/device/analyze', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(request) })
    return { status: result.status, body: await result.json() }
  }, request)
  expect(stale.status).toBe(409)
  await attach(info, 'actual-native-oracle-summary-no-curves', { device_id: id, raw_file_sha256: source.repository_source.sha256,
    selected_dependency_sha256: data.table.source_identity.selected_dependency_sha256,
    accepted_temperature_c: data.result.accepted_solution.temperature_c[0], accepted_power_w: data.result.accepted_solution.power_w[0],
    actual_residual_k: data.result.accepted_solution.residual_k[0], expected_temperature_c: 48.763050559454264,
    expected_power_w: 37.63050559454264, stale_source_status: stale.status, full_raw_evidence: 'Preserved outside tracked docs' })
})
