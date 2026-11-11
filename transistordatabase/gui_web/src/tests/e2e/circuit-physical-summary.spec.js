import { expect, test } from '@playwright/test'
import fs from 'node:fs/promises'

async function load(page, mobile = false) {
  if (mobile) await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('http://localhost:8002/index.html?view=converter')
  await expect(page.getByRole('heading', { name: 'Converter loss workbench', exact: true })).toBeVisible()
  await page.getByTestId('circuit-enable').check()
  await page.getByTestId('circuit-fixture').click()
}
async function calculate(page) {
  const waiting = page.waitForResponse(response => response.url().includes('/api/converters/circuit-summary') && response.request().method() === 'POST')
  await page.getByTestId('circuit-calculate').click()
  const response = await waiting
  expect(response.status()).toBe(200)
  return { data: await response.json(), request: response.request().postDataJSON() }
}
async function advanced(page, value) {
  if (!await page.getByTestId('circuit-json').isVisible()) await page.getByText('Advanced circuit request import / inspection', { exact: true }).click()
  await page.getByTestId('circuit-json').fill(typeof value === 'string' ? value : JSON.stringify(value))
  await page.getByTestId('circuit-advanced').check()
}

for (const mobile of [false, true]) test(`compiled ${mobile ? 'mobile' : 'desktop'} physical fixture and exact audit downloads`, async ({ page }) => {
  const errors = [], analysisRequests = []
  page.on('pageerror', error => errors.push(error.message))
  page.on('request', request => { if (request.url().includes('/api/converters/analyze')) analysisRequests.push(request) })
  await load(page, mobile)
  const { data } = await calculate(page)
  expect(analysisRequests).toHaveLength(0)
  expect(data.electrical.total_power_w).toBe(2)
  expect(data.members.map(member => member.electrical.total_power_w)).toEqual([1, 1])
  expect(data.maximum.tj_c).toBe(35)
  expect(data.input_identity.canonical_request_sha256).toBe(data.input_sha256)
  expect(data.input_identity.converter_request_sha256).toMatch(/^[a-f0-9]{64}$/)
  expect(data.input_identity.upstream_source_sha256).toBeNull()
  expect(data.maximum.junction_ids).toEqual(['Q1:shared-junction', 'Q2:shared-junction'])
  const result = page.getByTestId('circuit-results')
  await expect(result).toContainText('Q1:shared-junction, Q2:shared-junction')
  for (const [name, expected] of [['Download circuit audit JSON', data.export_json], ['Download circuit audit CSV', data.export_csv]]) {
    const waiting = page.waitForEvent('download'); await page.getByRole('button', { name, exact: true }).click()
    expect(await fs.readFile(await (await waiting).path(), 'utf8')).toBe(expected)
  }
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
  await result.screenshot({ path: test.info().outputPath(`circuit-${mobile ? 'mobile' : 'desktop'}.png`) })
  await page.getByTestId('circuit-boundary-source').fill('Changed boundary identity')
  await expect(result).toHaveCount(0)
  expect(errors).toEqual([])
})

test('compiled explicit unknowns, omissions, cold sink, zero resistance, shared junction and negative channel owner', async ({ page }) => {
  await load(page)
  const first = await calculate(page), base = first.request
  const cold = structuredClone(base); cold.boundary.temperature_c = 10
  await advanced(page, cold); expect((await calculate(page)).data.maximum.tj_c).toBe(20)
  const missing = structuredClone(base); missing.members[0].junctions[0].rtheta_k_w = null
  await advanced(page, missing); const missingData = (await calculate(page)).data
  expect(missingData.maximum.tj_c).toBeNull(); expect(missingData.members[0].junctions[0].tj_c).toBeNull()
  const ideal = structuredClone(base)
  ideal.members.forEach(member => { member.junctions[0].rtheta_k_w = 0; member.junctions[0].source = 'Declared ideal path'; member.junctions[0].conditions = { description: 'Explicit ideal zero-resistance junction to sink' } })
  await advanced(page, ideal); expect((await calculate(page)).data.maximum.tj_c).toBe(25)
  const reverse = structuredClone(base); reverse.converter.signals.current = [-1, -1]
  await advanced(page, reverse); const reverseData = (await calculate(page)).data
  expect(reverseData.members[0].die_losses.channel.power_w).toBe(1); expect(reverseData.maximum.tj_c).toBe(35)
  const shared = structuredClone(base)
  shared.members.forEach(member => { member.loss_owners.turn_off.positive.die = 'diode' })
  await advanced(page, shared); expect((await calculate(page)).data.maximum.tj_c).toBe(35)
  const omitted = structuredClone(base)
  omitted.provenance.source_sha256 = 'a'.repeat(64)
  omitted.converter.role_models.switch_pair.switching.turn_off = { omit: true, omission_reason: 'Unknown diode turn-off component' }
  omitted.members.forEach(member => { member.loss_owners.turn_off.positive.die = 'diode' })
  await advanced(page, omitted); const incomplete = (await calculate(page)).data
  expect(incomplete.electrical.total_power_w).toBeNull(); expect(incomplete.maximum.tj_c).toBeNull(); expect(incomplete.maximum.known_heat_tj_c).toBeNull()
  omitted.acknowledge_known_only = true
  await advanced(page, omitted); const known = (await calculate(page)).data
  expect(known.maximum.tj_c).toBeNull(); expect(known.maximum.known_heat_tj_c).toBe(35)
  expect(known.input_identity.upstream_source_sha256).toBe('a'.repeat(64))
  expect(known.input_identity.canonical_request_sha256).toBe(known.input_sha256)
  await expect(page.getByTestId('circuit-results')).toContainText('Acknowledged known-heat-only maximum: 35.0000 °C')
  await page.getByTestId('circuit-results').screenshot({ path: test.info().outputPath('circuit-partial-known-heat-ack.png') })
})

test('compiled suppresses late responses and rejects raw thermal underflow before inventing zero', async ({ page }) => {
  await load(page)
  let release, received
  const pending = new Promise(resolve => { received = resolve }), held = new Promise(resolve => { release = resolve })
  await page.route('**/api/converters/circuit-summary', async route => { const response = await route.fetch(); received(); await held; await route.fulfill({ response }) })
  await page.getByTestId('circuit-calculate').click(); await pending
  await page.getByTestId('circuit-provenance').fill('Changed while pending'); release()
  await expect(page.getByTestId('circuit-calculate')).toBeEnabled()
  await expect(page.getByTestId('circuit-results')).toHaveCount(0)
  await page.unroute('**/api/converters/circuit-summary')
  let sent = 0; page.on('request', request => { if (request.url().includes('/api/converters/circuit-summary')) sent++ })
  await page.getByTestId('junction-rtheta-0').fill('1e-999'); await page.getByTestId('circuit-calculate').click()
  await expect(page.getByRole('alert')).toContainText('too small'); expect(sent).toBe(0)
  await page.getByTestId('junction-rtheta-0').fill('10')
  const { request } = await calculate(page)
  const raw = JSON.stringify(request).replace('"rtheta_k_w":10', '"rtheta_k_w":1e-999')
  await advanced(page, raw)
  const waiting = page.waitForResponse(response => response.url().includes('/api/converters/circuit-summary'))
  await page.getByTestId('circuit-calculate').click(); const rejected = await waiting
  expect(rejected.status()).toBe(422); expect(rejected.request().postData()).toBe(raw)
  await expect(page.getByTestId('circuit-results')).toHaveCount(0)
})
