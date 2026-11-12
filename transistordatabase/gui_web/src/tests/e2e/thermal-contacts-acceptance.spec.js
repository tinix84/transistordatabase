import { test, expect } from '@playwright/test'
import { readFile } from 'node:fs/promises'
import { createHash } from 'node:crypto'
const sha = text => createHash('sha256').update(text).digest('hex')
const contact = (id, path, resistance) => ({ interface_id: id, path, resistivity_k_m2_w: resistance, area_m2: 1,
  source: `Independent ${id} SI contact oracle`, conditions: { pressure_pa: 0, ambient_c: null } })
function fixture() { return { operation: 'steady', source: 'Independent simultaneous-contact browser oracle', conditions: {},
  sink_temperature_c: 25, forward_power_w: 2, reverse_power_w: 1,
  common_resistance_k_w: 0, forward_resistance_k_w: 0, reverse_resistance_k_w: 0,
  interfaces: [contact('c-common', 'common', 1), contact('c-forward', 'forward', 2), contact('c-reverse', 'reverse', 3)] } }
async function attach(info, name, value) { await info.attach(name, { body: JSON.stringify(value, null, 2), contentType: 'application/json' }) }
async function calculate(page) {
  const ready = page.waitForResponse(r => r.url().endsWith('/api/thermal/analyze') && r.request().method() === 'POST')
  await page.getByRole('button', { name: 'Calculate thermal result', exact: true }).click()
  const response = await ready
  return { status: response.status(), body: await response.json(), raw_request: response.request().postData() }
}
async function advanced(page, value) {
  const checkbox = page.getByLabel('Use advanced JSON', { exact: true })
  if (!(await checkbox.isVisible())) await page.getByText('Advanced reproducible request', { exact: true }).click()
  await checkbox.check()
  await page.getByLabel('Thermal JSON', { exact: true }).fill(typeof value === 'string' ? value : JSON.stringify(value))
  return calculate(page)
}
async function download(page, button, info, name) {
  const ready = page.waitForEvent('download'); await button.click(); const item = await ready
  const path = info.outputPath(name); await item.saveAs(path); return readFile(path, 'utf8')
}
function temperatures(result) { return [result.shared_temperature_c, result.forward_temperature_c, result.reverse_temperature_c] }
test.beforeEach(async ({ page }, info) => {
  const capture = { console: [], pageerrors: [], failed_requests: [], http5xx: [] }
  page.on('console', item => capture.console.push({ type: item.type(), text: item.text() }))
  page.on('pageerror', error => capture.pageerrors.push(error.message))
  page.on('requestfailed', item => capture.failed_requests.push({ url: item.url(), failure: item.failure() }))
  page.on('response', item => { if (item.status() >= 500) capture.http5xx.push({ url: item.url(), status: item.status() }) })
  info.capture = capture
  await page.goto('/index.html?view=thermal')
  await expect(page.getByRole('button', { name: 'Calculate thermal result', exact: true })).toBeVisible()
})
test.afterEach(async ({ page }, info) => {
  await attach(info, 'browser-console-network', info.capture)
  expect(info.capture.pageerrors).toEqual([]); expect(info.capture.failed_requests).toEqual([]); expect(info.capture.http5xx).toEqual([])
  if (info.project.name === 'mobile-390') expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(391)
})

test('thermal contact acceptance actual controls preserve three contacts and 28 32 31 oracle with exports', async ({ page }, info) => {
  await page.getByLabel('Thermal source', { exact: true }).fill('Independent simultaneous-contact browser oracle')
  await page.getByLabel('Reference temperature', { exact: true }).fill('25')
  await page.getByLabel('Forward heat', { exact: true }).fill('2')
  await page.getByLabel('Reverse branch heat (W)', { exact: true }).fill('1')
  for (const label of ['Common path (K/W)', 'Forward branch path (K/W)', 'Reverse branch path (K/W)']) await page.getByLabel(label, { exact: true }).fill('0')
  await page.getByTestId('thermal-contacts-enabled').check()
  const entries = fixture().interfaces
  for (const [i, row] of entries.entries()) {
    if (i) await page.getByTestId('thermal-contact-add').click()
    const field = page.getByTestId(`thermal-contact-${i}`)
    await field.getByTestId('contact-id').fill(row.interface_id)
    await field.getByTestId('contact-path').selectOption(row.path)
    await field.getByTestId('contact-rho').fill(String(row.resistivity_k_m2_w))
    await field.getByTestId('contact-area').fill('1')
    await field.getByTestId('contact-source').fill(row.source)
    await field.getByTestId('contact-conditions').fill('pressure_pa,0\nambient_c,')
  }
  const result = await calculate(page); expect(result.status).toBe(200)
  expect(temperatures(result.body.result)).toEqual([28, 32, 31])
  expect(result.body.result.path_resistances_k_w).toEqual({ common: 1, forward: 2, reverse: 3 })
  expect(result.body.result.interface_contributions.map(row => row.interface_id)).toEqual(entries.map(row => row.interface_id))
  expect(result.body.result.interface_contributions[0].conditions).toEqual({ pressure_pa: 0, ambient_c: null })
  await expect(page.getByTestId('thermal-contact-evidence')).toContainText('K·m²/W')
  for (const row of entries) await expect(page.getByTestId('thermal-contact-evidence')).toContainText(row.source)
  const csv = await download(page, page.getByTestId('thermal-contacts-download'), info, 'thermal-contacts.csv')
  expect(csv.trim().split(/\r?\n/)).toHaveLength(4)
  for (const header of ['interface_id', 'resistivity_k_m2_w', 'area_m2', 'resistance_k_w', 'shared_temperature_c']) expect(csv).toContain(header)
  expect(csv).not.toContain('"interfaces"')
  const exported = JSON.parse(await download(page, page.getByRole('button', { name: 'Export thermal request + result JSON', exact: true }), info, 'thermal-request-result.json'))
  expect(exported).toEqual(result.body)
  await attach(info, 'contact-real-api-and-identity', { ...result, response_sha256: sha(JSON.stringify(result.body)) })
  await page.screenshot({ path: info.outputPath('three-contacts-oracle.png'), fullPage: true })
  await page.getByTestId('thermal-contact-0').getByTestId('contact-source').fill('Changed source must invalidate previous evidence')
  await expect(page.getByRole('region', { name: 'Thermal result', exact: true })).toHaveCount(0)
  await expect(page.getByRole('status').filter({ hasText: 'Inputs changed. Calculate again' })).toBeVisible()
})

test('thermal contact acceptance same-path stacks permutation and sourced ideal zero preserve identities', async ({ page }, info) => {
  const data = fixture()
  data.interfaces = [contact('common-a', 'common', .5), contact('common-b', 'common', .5), contact('forward', 'forward', 2), contact('reverse', 'reverse', 3), contact('ideal-zero', 'reverse', 0)]
  let result = await advanced(page, data); expect(result.status).toBe(200)
  expect(temperatures(result.body.result)).toEqual([28, 32, 31])
  expect(result.body.result.interface_contributions.map(row => row.resistance_k_w)).toEqual([.5, .5, 2, 3, 0])
  const original = result.body.result
  data.interfaces.reverse()
  result = await advanced(page, data); expect(result.status).toBe(200)
  expect(temperatures(result.body.result)).toEqual(temperatures(original))
  expect(result.body.result.path_resistances_k_w).toEqual(original.path_resistances_k_w)
  expect(result.body.result.interface_contributions.map(row => row.interface_id)).toEqual(data.interfaces.map(row => row.interface_id))
  await attach(info, 'stack-and-permutation-real-api', result)
  await page.screenshot({ path: info.outputPath('series-stack-and-ideal-zero.png'), fullPage: true })
})

test('thermal contact acceptance unknown paths duplicate IDs and singular plural combinations reject', async ({ page }, info) => {
  const variants = []
  const unknownPath = fixture(); unknownPath.common_resistance_k_w = null; variants.push(unknownPath)
  const unknownContact = fixture(); unknownContact.interfaces[0].resistivity_k_m2_w = null; variants.push(unknownContact)
  const duplicate = fixture(); duplicate.interfaces[1].interface_id = duplicate.interfaces[0].interface_id; variants.push(duplicate)
  const combined = fixture(); combined.interface = { path: 'common', resistivity_k_m2_w: 0, area_m2: 1, source: 'Conflicting singular contact', conditions: {} }; variants.push(combined)
  for (const [i, data] of variants.entries()) {
    const result = await advanced(page, data); expect(result.status).toBe(422)
    await expect(page.getByRole('alert')).toBeVisible()
    await expect(page.getByRole('region', { name: 'Thermal result', exact: true })).toHaveCount(0)
    await attach(info, `invalid-contact-${i}`, result)
  }
})

test('thermal contact acceptance raw lexical underflow Unicode rejects and legacy singular request stays valid', async ({ page }, info) => {
  const base = JSON.stringify(fixture())
  const variants = [base.replace('"resistivity_k_m2_w":1', '"resistivity_k_m2_w":1e-400'),
    base.replace('"resistivity_k_m2_w":1', '"resistivity_k_m2_w":-1e-400'),
    base.replace('"interface_id":"c-common"', '"interface_id":"\\ud800"')]
  for (const [i, raw] of variants.entries()) {
    const result = await advanced(page, raw); expect(result.status).toBe(422)
    expect(result.raw_request).toBe(raw)
    await attach(info, `raw-contact-rejection-${i}`, { ...result, request_sha256: sha(raw) })
  }
  const legacy = fixture(); delete legacy.interfaces
  legacy.interface = { path: 'forward', resistivity_k_m2_w: 1, area_m2: 1, source: 'Legacy single contact oracle', conditions: {} }
  const result = await advanced(page, legacy); expect(result.status).toBe(200)
  expect(result.body.result.forward_temperature_c).toBe(27); expect(result.body.result.reverse_temperature_c).toBe(25)
  expect(result.body.request).not.toHaveProperty('interfaces')
  await attach(info, 'legacy-singular-real-api', result)
})

test('thermal contact acceptance metadata postfix preserves prototype-like condition own keys in real requests and exports', async ({ page }, info) => {
  await page.getByLabel('Thermal source', { exact: true }).fill('Independent metadata preservation oracle')
  await page.getByLabel('Reference temperature', { exact: true }).fill('25')
  await page.getByLabel('Forward heat', { exact: true }).fill('2')
  await page.getByLabel('Reverse branch heat (W)', { exact: true }).fill('1')
  for (const label of ['Common path (K/W)', 'Forward branch path (K/W)', 'Reverse branch path (K/W)']) await page.getByLabel(label, { exact: true }).fill('0')
  await page.getByTestId('thermal-contacts-enabled').check()
  for (const [i, row] of fixture().interfaces.entries()) {
    if (i) await page.getByTestId('thermal-contact-add').click()
    const field = page.getByTestId(`thermal-contact-${i}`)
    await field.getByTestId('contact-id').fill(row.interface_id)
    await field.getByTestId('contact-path').selectOption(row.path)
    await field.getByTestId('contact-rho').fill(String(row.resistivity_k_m2_w))
    await field.getByTestId('contact-area').fill('1')
    await field.getByTestId('contact-source').fill(row.source)
    await field.getByTestId('contact-conditions').fill('__proto__,\nconstructor,7\ntoString,0')
  }
  const result = await calculate(page); expect(result.status).toBe(200)
  expect(temperatures(result.body.result)).toEqual([28, 32, 31])
  const sent = JSON.parse(result.raw_request)
  for (const rows of [sent.interfaces, result.body.request.interfaces, result.body.result.interface_contributions]) {
    for (const row of rows) {
      for (const key of ['__proto__', 'constructor', 'toString']) expect(Object.hasOwn(row.conditions, key)).toBe(true)
      expect(row.conditions.__proto__).toBeNull()
      expect(row.conditions.constructor).toBe(7)
      expect(row.conditions.toString).toBe(0)
    }
  }
  const csv = await download(page, page.getByTestId('thermal-contacts-download'), info, 'postfix-contact-conditions.csv')
  for (const key of ['__proto__', 'constructor', 'toString']) expect(csv).toContain(key)
  expect(csv.trim().split(/\r?\n/)).toHaveLength(4)
  const exported = JSON.parse(await download(page, page.getByRole('button', { name: 'Export thermal request + result JSON', exact: true }), info, 'postfix-contact-conditions.json'))
  expect(exported).toEqual(result.body)
  expect(Object.hasOwn(exported.request.interfaces[0].conditions, '__proto__')).toBe(true)
  await attach(info, 'postfix-metadata-real-api', { ...result, response_sha256: sha(JSON.stringify(result.body)) })
  await page.screenshot({ path: info.outputPath('postfix-contact-conditions.png'), fullPage: true })
})
