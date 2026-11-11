import { createHash } from 'node:crypto'
import { readFile } from 'node:fs/promises'
import { expect, test } from '@playwright/test'

const origin = 'http://localhost:8002'
const fixtures = '/home/tinix/claude_wsl/archive_ntbees2/sc/db/scData/'
const binary = 'Infineon IPX65R190CFDA_125C_HB_Sw+Sw.mat'
const octave = 'Infineon_IPW65R048CFDA_125C_HB_Sw+Sw.mat'
const branches = ['forwardConductionTrace', 'reverseConductionTrace', 'forwardTurnOnTrace',
  'reverseTurnOnTrace', 'forwardTurnOffTrace', 'reverseTurnOffTrace']
const field = (page, id) => page.getByTestId('mat-' + id)
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex')
const reply = (page, endpoint) => page.waitForResponse(response =>
  response.request().method() === 'POST' && response.url() === origin + '/api/mat/' + endpoint)

async function load(page, name) {
  const waiting = reply(page, 'preview')
  await field(page, 'file').setInputFiles(fixtures + name)
  const response = await waiting
  expect(response.status()).toBe(200)
  const original = await readFile(fixtures + name)
  // Chromium can evict large responses before its inspector reads their bodies.
  // Read the browser download and compare it with a separate HTTP request below.
  const archive = JSON.parse(await download(page, 'archive-json'))
  expect(archive.source.sha256).toBe(sha256(original))
  expect(Buffer.from(archive.source_base64, 'base64').equals(original)).toBe(true)
  return { archive, original }
}

async function download(page, button) {
  const waiting = page.waitForEvent('download')
  await field(page, button).click()
  return readFile(await (await waiting).path())
}

async function mapConduction(page) {
  for (const [name, value] of Object.entries({
    'identity-name': 'Independently declared part', 'identity-type': 'MOSFET',
    'identity-author': 'Migration review', 'identity-manufacturer': 'Declared vendor',
    'identity-housing_type': 'Declared package', 'rating-v_abs_max': 650,
    'rating-i_abs_max': 160, 'rating-i_cont': 60, 'rating-t_j_max': 150,
    'area-housing_area': 0.0001, 'area-cooling_area': 0.0002,
    'declaration-source': 'Independent browser fixture. Ratings and conditions are test declarations.',
  })) await field(page, name).fill(String(value))
  const branch = 'forwardConductionTrace'
  for (const other of branches.filter(name => name !== branch)) {
    await field(page, 'omit-' + other).fill('Preserve source without a physical declaration.')
  }
  await field(page, 'select-' + branch).check()
  for (const [name, value] of Object.entries({ target: 'switch', source: 'original-trace',
    convention: 'trace-positive-magnitudes', order: 'recorded-order', zero: 'reject-zero',
    extrapolation: 'reject' })) {
    await field(page, name + '-' + branch).selectOption(value)
  }
  await field(page, 'indices-' + branch).fill('0,1')
  await field(page, 'evidence-' + branch).fill('Original forward conduction trace, samples 0 and 1. Conditions declared for this test.')
  await field(page, 'condition-' + branch + '-t_j_c').fill('73')
  await field(page, 'condition-' + branch + '-gate_voltage_v').fill('15')
  await field(page, 'confirm-units').check()
}

function nativeRows(array) {
  expect(array.kind).toBe('ndarray')
  expect(array.values.kind).toBe('list')
  return array.values.values.map(row => {
    expect(row.kind).toBe('list')
    return row.values
  })
}

function summary(parent, name) {
  return parent.locator('summary').filter({ hasText: new RegExp('^' + name + ' ·') })
}

test('binary image pixels, source archive, declarations and stale requests survive a mobile round trip', async ({ page }, info) => {
  test.setTimeout(150000)
  await page.setViewportSize({ width: 390, height: 844 })
  const errors = []
  page.on('pageerror', error => errors.push(error.message))
  await page.goto(origin + '/index.html?view=legacy-mat')
  const { archive, original } = await load(page, binary)
  const { fields } = archive.variables.scData
  const image = fields.forwardConductionTrace.fields.image
  expect(image.sha256).toBe('a17ae362c8e2d3e7e9f75e23c56953019801a6d4077ae2302cb36edb78da7864')
  expect(archive.diagnostics).toEqual(expect.arrayContaining([expect.objectContaining({ code: 'conflicting-reference-voltages' })]))
  await summary(page, 'scData').click()
  const branch = summary(page, 'forwardConductionTrace')
  await branch.click()
  const originalImage = summary(branch.locator('..'), 'image')
  await originalImage.click()
  const panel = originalImage.locator('..')
  const waiting = reply(page, 'array')
  await field(panel, 'inspect-array').click()
  expect((await waiting).status()).toBe(200)
  const canvas = field(panel, 'original-image')
  await expect.poll(() => canvas.evaluate(element => [
    element.width, element.height,
    ...element.getContext('2d').getImageData(12, 233, 1, 1).data,
  ])).toEqual([451, 484, 79, 38, 24, 255])
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(390)
  await canvas.scrollIntoViewIfNeeded()
  await page.screenshot({ path: info.outputPath('actual-sc-rgb-mobile.png') })
  const originalDownload = await download(page, 'source-export')
  expect(originalDownload.equals(original)).toBe(true)
  await mapConduction(page)
  const response = reply(page, 'finalize')
  await field(page, 'finalize').click()
  expect((await response).status()).toBe(200)
  await expect(field(page, 'core-result')).toContainText('73 °C')
  const native = JSON.parse(await download(page, 'core-json'))
  const device = native.transistor.fields
  const record = device.switch.fields.channel_data.values[0].fields
  expect(nativeRows(record.graph_v_i)).toEqual([
    [1.0272873194221508, 1.3483146067415712], [2.9721362229102195, 3.9628482972136263],
  ])
  expect(device.switch.fields.metadata.values.legacy_mat_archive.values.source_base64).toBe(archive.source_base64)
  const midpoint = (2.9721362229102195 + 3.9628482972136263) / 2
  const volts = (1.0272873194221508 + 1.3483146067415712) / 2
  await field(page, 'calculate-target').selectOption('switch')
  await field(page, 'calculate-current').fill(String(midpoint))
  const calculated = reply(page, 'calculate')
  await field(page, 'calculate').click()
  const answer = await calculated
  expect(answer.status()).toBe(200)
  const point = await answer.json()
  expect(point.voltage_v).toBeCloseTo(volts, 12)
  expect(point.resistance_ohm).toBeCloseTo(volts / midpoint, 12)
  await expect(field(page, 'calculation-voltage')).toHaveText(String(point.voltage_v))
  expect(JSON.parse(await download(page, 'calculation-json'))).toEqual(point)
  const csv = (await download(page, 'calculation-csv')).toString().trim().split(/\r?\n/)
  expect(csv[1].split(',').map(Number)).toEqual([
    midpoint, point.voltage_v, point.resistance_ohm, 2.9721362229102195, 3.9628482972136263,
  ])
  await field(page, 'calculate-current').fill('5')
  const outOfRange = reply(page, 'calculate')
  await field(page, 'calculate').click()
  expect((await outOfRange).status()).toBe(422)
  await expect(field(page, 'calculation')).toHaveCount(0)
  await field(page, 'condition-forwardConductionTrace-t_j_c').fill('74')
  await expect(field(page, 'core-result')).toHaveCount(0)
  let announce, release
  const received = new Promise(done => { announce = done })
  const hold = new Promise(done => { release = done })
  await page.route('**/api/mat/finalize', async route => {
    const response = await route.fetch()
    announce()
    await hold
    await route.fulfill({ response })
  })
  await field(page, 'finalize').click()
  await received
  await field(page, 'condition-forwardConductionTrace-t_j_c').fill('75')
  const released = reply(page, 'finalize')
  release()
  await released
  await expect(field(page, 'core-result')).toHaveCount(0)
  expect(errors).toEqual([])
})

test('Octave archive and semantic MAT5 download retain original arrays and calibration', async ({ page }, info) => {
  test.setTimeout(150000)
  await page.setViewportSize({ width: 1280, height: 900 })
  await page.goto(origin + '/index.html?view=legacy-mat')
  const { archive } = await load(page, octave)
  expect(archive.source.format).toBe('octave-text-subset')
  expect(archive.calibrations['scData.forwardConductionTrace'].axes.x.outside_calibration_indices).toEqual([11])
  const direct = await page.request.post(origin + '/api/mat/preview', {
    data: { source_name: octave, mat_base64: archive.source_base64 },
  })
  expect(direct.status()).toBe(200)
  expect(await direct.json()).toEqual(archive)
  const exported = await download(page, 'legacy-export')
  const preview = await page.request.post(origin + '/api/mat/preview', {
    data: { mat_base64: exported.toString('base64'), source_name: 'converted.mat' },
  })
  expect(preview.status()).toBe(200)
  const converted = await preview.json()
  expect(converted.source.format).toBe('mat5')
  expect(converted.variables).toEqual(archive.variables)
  expect(converted.calibrations).toEqual(archive.calibrations)
  await mapConduction(page)
  const waiting = reply(page, 'finalize')
  await field(page, 'finalize').click()
  expect((await waiting).status()).toBe(200)
  const result = field(page, 'core-result')
  await expect(result).toContainText('0.0350263')
  await result.scrollIntoViewIfNeeded()
  await page.screenshot({ path: info.outputPath('actual-octave-declared-core.png') })
  const doc = JSON.parse(await download(page, 'core-json'))
  expect(nativeRows(doc.transistor.fields.switch.fields.channel_data.values[0].fields.graph_v_i))
    .toEqual([[0.0350263, 0.980736], [0.2834646, 10.77165]])
})
