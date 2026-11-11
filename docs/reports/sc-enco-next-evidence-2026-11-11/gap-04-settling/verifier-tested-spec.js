import { test, expect } from '@playwright/test'
import { readFile } from 'node:fs/promises'
const defaults = { q_gs: 4e-9, q_gd: 2e-9, q_g: 12e-9, v_th: 2, v_plateau: 4,
  r_g: 1.5, r_g_int: 0.5, v_driver: 10, v_off: 0 }
const endpoint = '/api/analytical/gate_charge'
async function fields(page, changes = {}) {
  for (const [name, value] of Object.entries({ ...defaults, ...changes })) await page.locator(`.gate-calculator input[name="${name}"]`).fill(String(value))
}
async function calculate(page) {
  const ready = page.waitForResponse(r => r.url().endsWith(endpoint) && r.request().method() === 'POST')
  await page.getByRole('button', { name: 'Calculate switching intervals', exact: true }).click()
  const response = await ready
  return { status: response.status(), body: await response.json(), request: JSON.parse(response.request().postData()) }
}
async function target(page, value) {
  await page.getByTestId('gate-settling-enabled').check()
  await page.getByTestId('gate-settling-target').fill(String(value))
  return calculate(page)
}
async function download(page, id, info, name) {
  const ready = page.waitForEvent('download')
  await page.getByTestId(id).click()
  const item = await ready, path = info.outputPath(name)
  await item.saveAs(path)
  return readFile(path, 'utf8')
}
async function attach(info, name, value) { await info.attach(name, { body: JSON.stringify(value, null, 2), contentType: 'application/json' }) }
function close(actual, expected) { expect(Math.abs(actual / expected - 1)).toBeLessThan(1e-12) }
test.beforeEach(async ({ page }, info) => {
  info.capture = { console: [], pageerrors: [], failed_requests: [], http5xx: [] }
  page.on('console', item => info.capture.console.push({ type: item.type(), text: item.text() }))
  page.on('pageerror', error => info.capture.pageerrors.push(error.message))
  page.on('requestfailed', item => info.capture.failed_requests.push({ url: item.url(), failure: item.failure() }))
  page.on('response', item => { if (item.status() >= 500) info.capture.http5xx.push({ url: item.url(), status: item.status() }) })
  await page.goto('/index.html?view=analytical&model=gate_charge')
  await expect(page.getByRole('heading', { name: 'Gate charge timing estimate', exact: true })).toBeVisible()
  await page.getByLabel('Use the constant-capacitance RC / Miller estimate', { exact: true }).check()
  await fields(page)
})
test.afterEach(async ({ page }, info) => {
  await attach(info, 'browser-console-network', info.capture)
  expect(info.capture.pageerrors).toEqual([])
  expect(info.capture.failed_requests).toEqual([])
  expect(info.capture.http5xx).toEqual([])
  if (info.project.name === 'mobile-390') expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(391)
})

test('gate settling acceptance RC oracle preserves legacy phases exports and stale evidence', async ({ page }, info) => {
  const legacy = await calculate(page)
  expect(legacy.status).toBe(200)
  expect(legacy.request).not.toHaveProperty('settling_target_v')
  expect(legacy.body.parameters).not.toHaveProperty('settling_target_v')
  expect(legacy.body.estimate).not.toHaveProperty('turn_on_settling')
  const result = await target(page, 7)
  expect(result.status).toBe(200)
  const phase = result.body.estimate.turn_on_settling
  for (const key of ['turn_on', 'turn_off']) expect(result.body.estimate[key]).toEqual(legacy.body.estimate[key])
  for (const key of ['turn_on_times', 'turn_off_times']) expect(result.body[key]).toEqual(legacy.body[key])
  close(phase.duration_s, 2e-9 * Math.log(2))
  close(phase.post_plateau_capacitance_f, 1e-9)
  close(phase.incremental_charge_c, 3e-9)
  expect(phase.initial_gate_current_a).toBe(3)
  expect(phase.final_gate_current_a).toBe(1.5)
  expect(phase.v_gate_v[0]).toBe(4)
  expect(phase.v_gate_v.at(-1)).toBe(7)
  expect(phase.relative_time_s.at(-1)).toBe(phase.duration_s)
  await expect(page.getByTestId('gate-settling-duration')).toContainText('1.38629e-9')
  await expect(page.getByTestId('gate-settling-results')).toContainText('gate')
  expect(Object.keys(phase).some(key => key.includes('energy_j') || key.includes('drain_current'))).toBe(false)
  expect(JSON.parse(await download(page, 'gate-settling-json', info, 'settling-evidence.json'))).toEqual(result.body)
  const csv = await download(page, 'gate-settling-csv', info, 'settling-trace.csv')
  expect(csv.trim().split(/\r?\n/)).toHaveLength(34)
  for (const name of ['relative_time_s', 'anchor_s', 'duration_s', 'absolute_end_s', 'incremental_charge_c']) expect(csv).toContain(name)
  expect(csv).not.toContain('[{')
  await attach(info, 'legacy-and-RC-real-api', { legacy, result })
  await page.screenshot({ path: info.outputPath('post-miller-RC-oracle.png'), fullPage: true })
  await page.getByTestId('gate-settling-target').fill('6')
  await expect(page.getByTestId('gate-settling-results')).toHaveCount(0)
  await expect(page.getByTestId('gate-settling-json')).toHaveCount(0)
})

test('gate settling acceptance zero charge has no current pulse and strict target bounds reject', async ({ page }, info) => {
  await fields(page, { q_gs: 2 ** -28, q_gd: 2 ** -29, q_g: 2 ** -28 + 2 ** -29 })
  const result = await target(page, 7)
  expect(result.status).toBe(200)
  const phase = result.body.estimate.turn_on_settling
  expect(phase.status).toBe('instantaneous-ideal-crossing')
  expect(phase.duration_s).toBe(0)
  expect(phase.relative_time_s).toEqual([0, 0])
  expect(phase.i_gate_a).toEqual([null, null])
  expect(phase.initial_gate_current_a).toBeNull()
  await expect(page.getByTestId('gate-settling-results')).toContainText('No finite-time gate current pulse')
  await expect(page.getByRole('heading', { name: 'Post-Miller gate current estimate', exact: true })).toHaveCount(0)
  const csv = await download(page, 'gate-settling-csv', info, 'instantaneous-trace.csv')
  expect(csv.trim().split(/\r?\n/)).toHaveLength(3)
  expect(JSON.parse(await download(page, 'gate-settling-json', info, 'instantaneous-evidence.json'))).toEqual(result.body)
  const invalid = []
  for (const value of ['4', '10', '11', '3', '4.0000000000000000000001']) {
    const response = await target(page, value)
    expect(response.status).toBe(422)
    await expect(page.getByTestId('gate-settling-results')).toHaveCount(0)
    invalid.push(response)
  }
  await page.getByTestId('gate-settling-target').fill('')
  await page.getByRole('button', { name: 'Calculate switching intervals', exact: true }).click()
  await expect(page.getByRole('alert')).toBeVisible()
  await expect(page.getByTestId('gate-settling-results')).toHaveCount(0)
  const raw = '{"q_gs":1,"q_gd":0,"q_g":1.0000000000000000001,"v_th":2,"v_plateau":4,"r_g":2,"r_g_int":0,"v_driver":10,"v_off":0,"settling_target_v":7}'
  const transport = []
  for (const [body, expected] of [[raw, 422], [raw.replace(',"settling_target_v":7', ''), 200],
    [raw.replace('1.0000000000000000001', '1').replace('"r_g":2', '"r_g":-1e-400'), 422],
    [raw.replace('1.0000000000000000001', '1' + '0'.repeat(400)), 422]]) {
    const response = await page.evaluate(async body => {
      const response = await fetch('/api/analytical/gate_charge', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body })
      return { raw_request: body, status: response.status, body: await response.json() }
    }, body)
    expect(response.status).toBe(expected)
    transport.push(response)
  }
  await attach(info, 'zero-charge-and-invalid-target-real-api', { result, invalid, transport })
  await page.screenshot({ path: info.outputPath('strict-boundary-error.png'), fullPage: true })
})

test('gate settling acceptance near bounds collapsed absolute clock and huge finite SI duration remain honest', async ({ page }, info) => {
  let result = await target(page, '9.999999999999998')
  expect(result.status).toBe(200)
  const nearRail = result
  expect(result.body.estimate.turn_on_settling.final_gate_current_a).toBeGreaterThan(0)
  expect(result.body.estimate.turn_on_settling.v_gate_v.at(-1)).toBe(9.999999999999998)
  await fields(page, { q_gs: 4e-6, q_gd: 2e-6, q_g: 6.006e-6 })
  result = await target(page, '4.000000000000001')
  expect(result.status).toBe(200)
  const local = result.body.estimate.turn_on_settling
  expect(local.duration_s).toBeGreaterThan(0)
  expect(local.absolute_end_s).toBeNull()
  expect(local.absolute_time_status).toBe('unresolved-use-relative-time')
  await expect(page.getByTestId('gate-settling-results')).toContainText('Unavailable — unresolved-use-relative-time')
  const collapsed = result
  await fields(page, { q_gs: 1e-9, q_gd: 0, q_g: 4.440892098500626e283, v_th: .5,
    v_plateau: 1, v_driver: 1.0000000000000004, r_g: 100, r_g_int: 0 })
  result = await target(page, '1.0000000000000002')
  expect(result.status).toBe(200)
  const phase = result.body.estimate.turn_on_settling
  close(phase.duration_s, 1e301 * Math.log(2))
  expect(phase.relative_time_s.every(Number.isFinite)).toBe(true)
  await expect(page.getByTestId('gate-settling-duration')).toContainText('6.93147e+300')
  expect(await page.getByTestId('gate-settling-results').textContent()).not.toContain('Infinity')
  await expect(page.getByRole('heading', { name: 'Post-Miller gate voltage estimate', exact: true })).toBeVisible()
  expect(JSON.parse(await download(page, 'gate-settling-json', info, 'huge-SI-evidence.json'))).toEqual(result.body)
  await attach(info, 'finite-relative-time-boundary-real-api', { nearRail, collapsed, huge: result })
  await page.screenshot({ path: info.outputPath('finite-SI-duration.png'), fullPage: true })
})
