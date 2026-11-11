import { test, expect } from '@playwright/test'
import { execFileSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { readFile } from 'node:fs/promises'

const origin = 'http://localhost:8002', python = '/home/tinix/miniconda3/envs/ntbees2/bin/python'
const repo = '/home/tinix/claude_wsl/transistordatabase'
const field = (page, id) => page.getByTestId('mat-' + id)
const mechanisms = ['forwardConductionTrace', 'reverseConductionTrace', 'forwardTurnOnTrace', 'reverseTurnOnTrace', 'forwardTurnOffTrace', 'reverseTurnOffTrace']
function fixture(mode = 'zero') {
  return Buffer.from(execFileSync(python, ['-c', `
import base64,sys
from io import BytesIO
import numpy as np
from scipy.io import savemat
mode=sys.argv[1]
current=[-1e100,0.,1e100] if mode=='placeholder' else [10.,20.]
energy=[0.,0.,0.] if mode=='placeholder' else ([-.001,.002] if mode=='negative' else [0.,0.])
sc={'fileFormatVersion':'0.2','name':np.empty((0,0)),
    'onStateCurrent':np.array([[-20.,-10.,10.,20.]]),'onStateVoltage':np.array([[-6.,-2.,2.,6.]]),
    'turnOnCurrent':np.array([current]),'turnOnEnergy':np.array([energy]),'turnOnVoltage':1. if mode=='placeholder' else 400.,
    'unknownInteger':np.array([[18446744073709551615]],dtype=np.uint64),
    'unknownText':np.array(list('A\\n '*2000),dtype='U1').reshape(1,-1)}
buffer=BytesIO();savemat(buffer,{'scData':sc},long_field_names=True,do_compression=True)
print(base64.b64encode(buffer.getvalue()).decode())
`, mode], { encoding: 'utf8', cwd: repo }).trim(), 'base64')
}
async function upload(page, bytes, name = 'untrusted_filename_125C_1200V.mat') {
  const waiting = page.waitForResponse(response => response.url() === origin + '/api/mat/preview' && response.request().method() === 'POST')
  await field(page, 'file').setInputFiles({ name, mimeType: 'application/octet-stream', buffer: bytes })
  const response = await waiting; expect(response.status()).toBe(200)
  return response.json()
}
async function declare(page, kind = 'conduction') {
  const branch = kind === 'conduction' ? 'forwardConductionTrace' : 'forwardTurnOnTrace'
  for (const [key, value] of Object.entries({ 'identity-name': 'Independent model', 'identity-type': 'MOSFET', 'identity-author': 'Browser fixture reviewer', 'identity-manufacturer': 'Declared vendor', 'identity-housing_type': 'Declared package', 'rating-v_abs_max': 650, 'rating-i_abs_max': 100, 'rating-i_cont': 60, 'rating-t_j_max': 150, 'area-housing_area': 0.0001, 'area-cooling_area': 0.0002, 'declaration-source': 'Independent fixture declaration, table 1' })) await field(page, key).fill(String(value))
  for (const key of mechanisms.filter(key => key !== branch)) await field(page, 'omit-' + key).fill('No supported physical declaration. Preserve in the source archive.')
  await field(page, 'select-' + branch).check()
  for (const [key, value] of Object.entries({ target: 'switch', source: 'combined', convention: 'signed-passive', order: 'sort-distinct-current', zero: 'reject-zero' })) await field(page, key + '-' + branch).selectOption(value)
  await field(page, 'evidence-' + branch).fill('Independent source samples. Tj 73 °C and gate 15 V are explicit declarations.')
  await field(page, 'condition-' + branch + '-t_j_c').fill('73')
  await field(page, 'condition-' + branch + '-gate_voltage_v').fill('15')
  if (kind !== 'conduction') {
    await field(page, 'condition-' + branch + '-gate_resistance_ohm').fill('2')
    await field(page, 'condition-' + branch + '-supply_voltage_v').fill('400')
  }
  await field(page, 'confirm-units').check()
  return branch
}
async function finalize(page, status = 200) {
  const waiting = page.waitForResponse(response => response.url() === origin + '/api/mat/finalize')
  await field(page, 'finalize').click(); const response = await waiting
  expect(response.status()).toBe(status); return response.json()
}
async function download(page, id) {
  const waiting = page.waitForEvent('download'); await field(page, id).click()
  return readFile(await (await waiting).path())
}
async function calculate(page, current, status = 200) {
  await field(page, 'calculate-target').selectOption('switch'); await field(page, 'calculate-current').fill(String(current))
  const waiting = page.waitForResponse(response => response.url() === origin + '/api/mat/calculate')
  await field(page, 'calculate').click(); const response = await waiting
  expect(response.status()).toBe(status); return response.json()
}
function originalField(page, name) { return page.locator('summary').filter({ hasText: new RegExp('^' + name + ' ·') }) }

test('signed source → explicit declarations → Python bounded point → native loader and exact downloads', async ({ page }, info) => {
  const errors = []; page.on('pageerror', error => errors.push(error.message))
  await page.goto(origin + '/index.html')
  await page.getByRole('button', { name: 'Legacy MAT', exact: true }).click()
  await expect(field(page, 'workbench')).toBeVisible()
  const bytes = fixture(), archive = await upload(page, bytes)
  expect(archive.source.sha256).toBe(createHash('sha256').update(bytes).digest('hex'))
  await expect(field(page, 'identity-name')).toHaveValue('')
  expect(JSON.parse(await download(page, 'archive-json'))).toEqual(archive)
  expect((await download(page, 'source-export')).equals(bytes)).toBe(true)
  await declare(page); const core = await finalize(page)
  const result = await calculate(page, 15)
  expect(result.current_coverage_a).toEqual([10, 20]); expect(result.voltage_v).toBe(4)
  expect(result.resistance_ohm).toBeCloseTo(4 / 15, 14)
  expect(result.conditions.t_j_c).toBe(73); expect(result.conditions.gate_voltage_v).toBe(15)
  await expect(field(page, 'calculation-voltage')).toHaveText('4')
  expect(JSON.parse(await download(page, 'calculation-json'))).toEqual(result)
  expect((await download(page, 'calculation-csv')).toString()).toBe(`current_A,voltage_V,resistance_ohm,coverage_min_A,coverage_max_A\r\n15,4,${result.resistance_ohm},10,20\r\n`)
  const native = await download(page, 'core-json')
  expect(JSON.parse(native)).toEqual(core.core_json_document)
  const loaded = JSON.parse(execFileSync(python, ['-c', `
import base64,hashlib,json,sys
from transistordatabase.core.repository import JsonTransistorLoader
device=JsonTransistorLoader().load_from_dict(json.load(sys.stdin))
curve=device.switch.channel_data[0]
archive=device.switch.metadata['legacy_mat_archive']
print(json.dumps({'name':device.metadata.name,'temperature':curve.t_j,'graph':curve.graph_v_i.tolist(),
'resistance':curve.get_resistance_at_current(15.),'source_sha256':hashlib.sha256(base64.b64decode(archive['source_base64'])).hexdigest()}))
`], { cwd: repo, input: native, encoding: 'utf8' }))
  expect(loaded).toEqual({ name: 'Independent model', temperature: 73, graph: [[2, 6], [10, 20]], resistance: result.resistance_ohm, source_sha256: archive.source.sha256 })
  await page.screenshot({ path: info.outputPath('mat-bounded-desktop.png'), fullPage: true })
  await page.setViewportSize({ width: 390, height: 844 })
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390)
  await field(page, 'calculation').scrollIntoViewIfNeeded(); await page.screenshot({ path: info.outputPath('mat-bounded-mobile.png') })
  await calculate(page, 5, 422); await expect(field(page, 'calculation')).toHaveCount(0)
  await expect(page.getByRole('alert')).toContainText('outside')
  await field(page, 'condition-forwardConductionTrace-t_j_c').fill('74'); await expect(field(page, 'core-result')).toHaveCount(0)
  expect(errors).toEqual([])
})

test('original binary uint64 and UTF-32 text retain exact values in the field inspector', async ({ page }) => {
  await page.goto(origin + '/index.html?view=legacy-mat'); await upload(page, fixture())
  await originalField(page, 'scData').click()
  for (const [name, expected] of [['unknownInteger', '18446744073709551615'], ['unknownText', 'A']]) {
    const summary = originalField(page, name); await summary.click(); const panel = summary.locator('..')
    const waiting = page.waitForResponse(response => response.url() === origin + '/api/mat/array')
    await field(panel, 'inspect-array').click(); expect((await waiting).status()).toBe(200)
    await expect(panel.locator('tbody td.literal').first()).toHaveText(expected)
    if (name === 'unknownText') expect(await panel.locator('tbody td.literal').nth(1).textContent()).toBe('\n')
  }
  const archive = JSON.parse(await download(page, 'archive-json'))
  expect(archive.variables.scData.fields.unknownInteger.encoding).toBe('source-array-reference')
  expect(archive.variables.scData.fields.unknownText.itemsize).toBe(4)
})

test('finite zero curves require explicit interpretation while placeholders and negative energy remain unsupported', async ({ page }) => {
  await page.goto(origin + '/index.html?view=legacy-mat'); await upload(page, fixture('zero'))
  const branch = await declare(page, 'turn-on')
  await finalize(page, 422); await expect(page.getByRole('alert')).toContainText('All-zero combined energy')
  await field(page, 'status-' + branch).selectOption('source-zero-unverified')
  await field(page, 'ack-' + branch).fill('Recorded zero values lack original trace and measurement evidence.')
  let result = await finalize(page)
  expect(result.declarations.curves[0].data_status).toBe('source-zero-unverified')
  await field(page, 'status-' + branch).selectOption('declared-ideal-model')
  await field(page, 'scope-' + branch).fill('Declared ideal turn-on branch for this simulation only.')
  result = await finalize(page); expect(result.declarations.curves[0].data_status).toBe('declared-ideal-model')
  await upload(page, fixture('placeholder')); await declare(page, 'turn-on')
  await field(page, 'condition-' + branch + '-supply_voltage_v').fill('1')
  await field(page, 'status-' + branch).selectOption('declared-ideal-model'); await field(page, 'scope-' + branch).fill('Cannot override source placeholder')
  await finalize(page, 422); await expect(page.getByRole('alert')).toContainText('placeholder')
  await upload(page, fixture('negative')); await declare(page, 'turn-on'); await finalize(page, 422)
  await expect(page.getByRole('alert')).toContainText('Negative source energy')
})

test('held upload, finalization, lookup and export responses cannot restore replaced source state', async ({ page }) => {
  await page.goto(origin + '/index.html?view=legacy-mat')
  const original = fixture(), replacement = fixture('negative')
  await upload(page, original); await declare(page); await finalize(page)
  let announce, release
  function hold() {
    return { started: new Promise(resolve => { announce = resolve }), held: new Promise(resolve => { release = resolve }) }
  }
  let gate = hold()
  await page.route('**/api/mat/calculate', async route => { const response = await route.fetch(); announce(); await gate.held; await route.fulfill({ response }) })
  await field(page, 'calculate-target').selectOption('switch'); await field(page, 'calculate-current').fill('15'); await field(page, 'calculate').click(); await gate.started
  await field(page, 'calculate-current').fill('16')
  let response = page.waitForResponse(response => response.url() === origin + '/api/mat/calculate')
  release(); await (await response).finished(); await expect(field(page, 'calculation')).toHaveCount(0)
  await page.unroute('**/api/mat/calculate')
  gate = hold()
  await page.route('**/api/mat/export', async route => { const response = await route.fetch(); announce(); await gate.held; await route.fulfill({ response }) })
  const downloads = []; page.on('download', item => downloads.push(item))
  await field(page, 'source-export').click(); await gate.started
  await upload(page, replacement, 'replacement.mat')
  response = page.waitForResponse(response => response.url() === origin + '/api/mat/export')
  release(); await (await response).finished()
  await expect(field(page, 'identity-name')).toHaveValue(''); await expect(field(page, 'core-result')).toHaveCount(0)
  await page.unroute('**/api/mat/export'); expect(downloads).toHaveLength(0)
  gate = hold(); let count = 0
  await page.route('**/api/mat/preview', async route => { const response = await route.fetch(); if (++count === 1) { announce(); await gate.held } await route.fulfill({ response }) })
  await field(page, 'file').setInputFiles({ name: 'held-old.mat', mimeType: 'application/octet-stream', buffer: original }); await gate.started
  await upload(page, replacement, 'current-source.mat')
  response = page.waitForResponse(response => response.url() === origin + '/api/mat/preview' && response.request().postDataJSON().source_name === 'held-old.mat')
  release(); await (await response).finished()
  await expect(field(page, 'hash')).toHaveText(createHash('sha256').update(replacement).digest('hex'))
  await expect(field(page, 'preview')).toContainText('current-source.mat'); await expect(field(page, 'preview')).not.toContainText('held-old.mat')
})
