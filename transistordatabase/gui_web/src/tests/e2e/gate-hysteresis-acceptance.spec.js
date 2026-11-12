import { test, expect } from '@playwright/test'
import { readFile } from 'node:fs/promises'
import { createHash } from 'node:crypto'

const gate = [0, 8, 4, 6, 4, 2, 10]
const states = [false, true, true, true, true, false, true]
const source = 'Independent noisy-gate browser oracle; explicit command-state assumption'
const digitizer = { thresholds: { scale: 'absolute', low_v: 2.5, high_v: 7.5 } }
const model = energy => ({ mode: 'recorded', current_a: [0, 2], energy_j: [energy, energy], source: 'constant-energy test fixture', recorded_conditions: { t_j_c: 25, v_supply_v: 400 }, requested_conditions: { t_j_c: 25, v_supply_v: 400 } })
const fixture = { waveform: { time_s: gate.map((_, n) => n / 1000), current_a: gate.map(() => 1), gate_v: gate }, events: { mode: 'hysteresis', digitizer }, conduction: { omit: true, omission_reason: 'switching only' }, switching: { turn_on: { positive: model(.002) }, turn_off: { positive: model(.003) } }, provenance: { source, calibration_notes: 'SI fixture: seconds, amperes, volts; calibrated range independent of window' } }
const csv = values => 'time_s,current_a,gate_v\n' + values.map((value, i) => `${i / 1000},1,${value}`).join('\n')
const sha = text => createHash('sha256').update(text).digest('hex')

async function attachJson(info, name, value) { await info.attach(name, { body: JSON.stringify(value, null, 2), contentType: 'application/json' }) }
async function open(page) { await page.goto('/index.html?view=waveform'); await expect(page.getByRole('heading', { name: 'Waveform workbench', exact: true })).toBeVisible() }
async function allocate(page, panel) {
  const received = page.waitForResponse(r => r.url().endsWith('/api/waveform/gate-allocation') && r.request().method() === 'POST')
  await panel.getByTestId('allocation-calculate').click()
  const response = await received
  return { status: response.status(), body: await response.json(), request: response.request().postDataJSON() }
}
async function setupAllocation(page, values) {
  await page.getByTestId('waveform-csv').fill(csv(values))
  const panel = page.getByTestId('gate-allocation')
  await panel.getByText('Gate-state current allocation', { exact: true }).click()
  await panel.getByTestId('allocation-mode').selectOption('hysteresis')
  await panel.getByTestId('allocation-source').fill(source)
  await panel.getByTestId('allocation-selected').check()
  return panel
}
async function downloadText(page, button, info, name) {
  const ready = page.waitForEvent('download')
  await button.click()
  const download = await ready
  const path = info.outputPath(name)
  await download.saveAs(path)
  return readFile(path, 'utf8')
}

test.beforeEach(async ({ page }, info) => {
  const capture = { console: [], pageerrors: [], requestfailed: [], servererrors: [] }
  page.on('console', message => capture.console.push({ type: message.type(), text: message.text() }))
  page.on('pageerror', error => capture.pageerrors.push(error.message))
  page.on('requestfailed', request => capture.requestfailed.push({ url: request.url(), failure: request.failure() }))
  page.on('response', response => { if (response.status() >= 500) capture.servererrors.push({ url: response.url(), status: response.status() }) })
  info.capture = capture
  await open(page)
})
test.afterEach(async ({ page }, info) => {
  await attachJson(info, 'browser-console-network', info.capture)
  expect(info.capture.pageerrors).toEqual([])
  expect(info.capture.requestfailed).toEqual([])
  expect(info.capture.servererrors).toEqual([])
})

test('actual noisy waveform has two on one off brackets and independent seven millijoule oracle', async ({ page }, info) => {
  await page.getByRole('button', { name: 'Load independent switching fixture', exact: true }).click()
  await page.getByLabel('Event policy', { exact: true }).selectOption('hysteresis')
  const controls = page.locator('form').first().getByTestId('hysteresis-settings')
  await expect(controls.getByTestId('hysteresis-low')).toHaveValue('2.5')
  await expect(controls.getByTestId('hysteresis-high')).toHaveValue('7.5')
  await page.getByText('Advanced: inspect or supply the canonical JSON request', { exact: true }).click()
  await page.getByLabel('Use JSON editor as calculation input', { exact: true }).check()
  await page.getByLabel('Canonical request', { exact: true }).fill(JSON.stringify(fixture))
  const received = page.waitForResponse(r => r.url().endsWith('/api/waveform/analyze') && r.request().method() === 'POST')
  await page.getByRole('button', { name: 'Calculate waveform losses', exact: true }).click()
  const response = await received
  expect(response.status()).toBe(200)
  const body = await response.json()
  expect(body.gate_digitization.states).toEqual(states)
  expect(body.gate_digitization.turn_on_brackets).toEqual([0, 5])
  expect(body.gate_digitization.turn_off_brackets).toEqual([4])
  expect(body.events.map(e => e.event)).toEqual(['turn_on', 'turn_off', 'turn_on'])
  expect(body.known_component_energy_j).toBeCloseTo(.007, 14)
  expect(body.known_component_power_w).toBeCloseTo(7 / 6, 13)
  expect(body.total_power_w).toBeNull()
  expect(body.request.provenance.source).toBe(source)
  await expect(page.getByTestId('waveform-hysteresis-evidence')).toContainText('2.5 V, high 7.5 V')
  const gateCsv = await downloadText(page, page.getByTestId('waveform-gate-csv'), info, 'waveform-gate.csv')
  expect(gateCsv.trim().split(/\r?\n/)).toHaveLength(8)
  expect(gateCsv).toContain('initial_state_source')
  expect(gateCsv).not.toContain('"states"')
  const exported = JSON.parse(await downloadText(page, page.getByRole('button', { name: 'Download request + result JSON', exact: true }).last(), info, 'waveform-result.json'))
  expect(exported.response.gate_digitization.states).toEqual(states)
  await attachJson(info, 'waveform-real-api-and-fixture-identity', { request: response.request().postDataJSON(), response: body, fixture_sha256: sha(JSON.stringify(fixture)), response_sha256: sha(JSON.stringify(body)) })
  await page.screenshot({ path: info.outputPath('noisy-waveform.png'), fullPage: true })
  await controls.getByTestId('hysteresis-low').fill('2')
  await expect(page.getByTestId('waveform-results')).toHaveCount(0)
  await expect(page.getByRole('status').filter({ hasText: 'Inputs changed. Calculate again' })).toBeVisible()
})

test('actual allocation preserves signed sample states export metadata hashes and stale clearing', async ({ page }, info) => {
  const panel = await setupAllocation(page, gate)
  const response = await allocate(page, panel)
  expect(response.status).toBe(200)
  expect(response.body.result.samples.map(r => r.gate_on)).toEqual(states)
  expect(response.body.result.samples.map(r => r.gate_on_current_a + r.gate_off_current_a)).toEqual(gate.map(() => 1))
  expect(response.body.evidence.input_sha256).toMatch(/^[a-f0-9]{64}$/)
  expect(response.body.evidence.result_sha256).toMatch(/^[a-f0-9]{64}$/)
  expect(response.body.inputs.source).toBe(source)
  const exported = JSON.parse(await downloadText(page, panel.getByTestId('allocation-json'), info, 'allocation.json'))
  expect(exported.evidence).toEqual(response.body.evidence)
  const text = await downloadText(page, panel.getByTestId('allocation-csv'), info, 'allocation.csv')
  expect(text.trim().split(/\r?\n/)).toHaveLength(8)
  for (const name of ['low_v', 'high_v', 'initial_state_source', 'input_sha256', 'result_sha256']) expect(text).toContain(name)
  expect(text).not.toContain('"states"')
  await attachJson(info, 'allocation-real-api', response)
  await page.screenshot({ path: info.outputPath('noisy-allocation.png'), fullPage: true })
  await panel.getByTestId('hysteresis-high').fill('9')
  await expect(panel.getByTestId('allocation-results')).toHaveCount(0)
  const changed = await allocate(page, panel)
  expect(changed.status).toBe(200)
  expect(changed.body.evidence.input_sha256).not.toBe(response.body.evidence.input_sha256)
  expect(changed.body.evidence.result_sha256).not.toBe(response.body.evidence.result_sha256)
})

test('calibrated flat traces equality initial ambiguity and contradictory states use actual controls', async ({ page }, info) => {
  const panel = await setupAllocation(page, [20, 20])
  await panel.getByTestId('hysteresis-scale').selectOption('relative')
  await panel.getByTestId('hysteresis-minimum').fill('0')
  await panel.getByTestId('hysteresis-maximum').fill('10')
  let response = await allocate(page, panel)
  expect(response.status).toBe(200)
  expect(response.body.result.gate_digitization.states).toEqual([true, true])
  expect(response.body.result.gate_digitization.turn_on_brackets).toEqual([])
  await page.screenshot({ path: info.outputPath('calibrated-flat.png'), fullPage: true })
  await page.getByTestId('waveform-csv').fill(csv([5, 5]))
  response = await allocate(page, panel)
  expect(response.status).toBe(422)
  await expect(panel.getByRole('alert')).toContainText('declare initial_state')
  await panel.getByTestId('hysteresis-initial').selectOption('on')
  response = await allocate(page, panel)
  expect(response.status).toBe(200)
  expect(response.body.result.gate_digitization.states).toEqual([true, true])
  await page.getByTestId('waveform-csv').fill(csv([2.5, 7.5]))
  await panel.getByTestId('hysteresis-initial').selectOption('off')
  response = await allocate(page, panel)
  expect(response.status).toBe(200)
  expect(response.body.result.gate_digitization.states).toEqual([false, false])
  await page.getByTestId('waveform-csv').fill(csv([0, 10]))
  await panel.getByTestId('hysteresis-initial').selectOption('on')
  response = await allocate(page, panel)
  expect(response.status).toBe(422)
  await expect(panel.getByRole('alert')).toContainText('conflicts')
  await attachJson(info, 'last-invalid-policy-real-api', response)
})

test('precision guards reject rounded upper fractions and negative tiny wire literals', async ({ page, request }, info) => {
  const panel = await setupAllocation(page, [0, 10])
  await panel.getByTestId('hysteresis-scale').selectOption('relative')
  await panel.getByTestId('hysteresis-minimum').fill('0')
  await panel.getByTestId('hysteresis-maximum').fill('10')
  await panel.getByTestId('hysteresis-high').fill('1.00000000000000000001')
  await panel.getByTestId('allocation-calculate').click()
  await expect(panel.getByRole('alert')).toContainText('before numeric rounding')
  await panel.getByTestId('hysteresis-high').fill('1')
  await panel.getByTestId('hysteresis-low').fill('-1e-400')
  await panel.getByTestId('allocation-calculate').click()
  await expect(panel.getByRole('alert')).toContainText('too small')
  const raw = JSON.stringify({ ...fixture, events: { mode: 'hysteresis', digitizer: { thresholds: { scale: 'absolute', low_v: 0, high_v: 7.5 } } } }).replace('"low_v":0', '"low_v":-1e-400')
  const response = await request.post('/api/waveform/analyze', { data: raw, headers: { 'Content-Type': 'application/json' } })
  expect(response.status()).toBe(422)
  expect(await response.json()).toHaveProperty('detail', 'A JSON number exceeds the finite numerical range')
  await attachJson(info, 'tiny-negative-wire-rejection', { status: response.status(), body: await response.json(), request_sha256: sha(raw) })
})
