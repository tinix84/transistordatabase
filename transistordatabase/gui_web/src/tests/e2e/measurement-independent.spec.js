import { execFileSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { existsSync, readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { test, expect } from '@playwright/test'

const root = fileURLToPath(new URL('../../../../../', import.meta.url))
const field = (page, id) => page.getByTestId('measurement-' + id)
const sourceLog = resolve(root, '../archive_ntbees2/sc/tests/DoublePulseTester_EPC8009_sync.log')
const sourceRaw = process.env.TDB_GENERIC_DPT_RAW
const start = '/index.html?view=measurements'
const python = process.env.MEASUREMENT_PYTHON ?? 'python3'
const fixtureScript = [
  'import json, runpy, sys',
  'from pathlib import Path',
  'from transistordatabase.core.repository import TransistorFactory',
  'from transistordatabase.core.serialization import core_to_json_document',
  'library = runpy.run_path("tests/test_measurement_import_independent.py")',
  'if sys.argv[1] == "native":',
  '    device = TransistorFactory.create_empty_transistor("Browser-DPT")',
  '    device._working_point = {"source_id": 9007199254740993}',
  '    print(json.dumps(core_to_json_document(device)))',
  'elif sys.argv[1] == "csv":',
  '    sys.stdout.buffer.write(library["csv_trace"]())',
  'else:',
  '    Path(sys.argv[2]).mkdir(parents=True, exist_ok=True)',
  '    sys.stdout.buffer.write(library["raw_trace"](Path(sys.argv[2])))',
].join('\n')
function fixture(format) {
  return execFileSync(python, ['-c', fixtureScript, format, test.info().outputDir], { cwd: root })
}
async function perform(page, id, endpoint, readBody = true) {
  const received = page.waitForResponse(r => r.url().endsWith('/api/measurement/' + endpoint))
  await field(page, id).click()
  const response = await received
  if (response.status() !== 200) throw new Error(response.status() + ' ' + (await response.text()).slice(0, 500))
  return readBody ? response.json() : null
}
async function upload(page, format, bytes, disclosure = 'Independent synthetic source. No calibrated device parameters.') {
  await page.goto(start)
  await field(page, 'format').selectOption(format)
  await field(page, 'file').setInputFiles({ name: 'source.' + format, mimeType: 'application/octet-stream', buffer: bytes })
  await expect(page.getByText(/Selected source:/)).toContainText(bytes.length + ' original bytes')
  await field(page, 'source-kind').selectOption('simulation')
  await field(page, 'disclosure').fill(disclosure)
  if (format === 'csv') {
    await field(page, 'delimiter').selectOption(';')
    await field(page, 'decimal').selectOption(',')
  }
  await perform(page, 'inspect', 'preview')
  await expect(field(page, 'hash')).toHaveText(createHash('sha256').update(bytes).digest('hex'))
}
async function nativeInput(page) {
  await field(page, 'native-file').setInputFiles({ name: 'independent.json', mimeType: 'application/json', buffer: fixture('native') })
  await expect(page.getByText(/Native device archive:/)).toContainText('independent.json')
}
async function downloaded(page, id) {
  const pending = page.waitForEvent('download')
  await field(page, id).click()
  return await (await pending).path()
}
async function selectMappings(page, format) {
  const names = format === 'csv' ? ['time;ns', 'Vds', 'probe "B"', 'Vgs'] : ['time', 'V(ds)', 'I(probe)', 'V(gs)']
  const units = format === 'csv' ? ['ns', 'V', 'mA', 'V'] : ['s', 'V', 'A', 'V']
  for (const [index, role] of ['time', 'voltage', 'current', 'gate'].entries()) {
    await field(page, 'map-' + role).selectOption(names[index])
    await field(page, 'unit-' + role).selectOption(units[index])
  }
  if (format === 'csv') await field(page, 'polarity-current').selectOption('-1')
  if (format === 'raw') await field(page, 'step-index').fill('0')
  for (const [key, value] of Object.entries({ t_j: 73, bus_voltage_v: 48, v_g_on: 12, v_g_off: -2, r_g_on: 3, r_g_off: 9 })) {
    await field(page, 'known-' + key).selectOption({ label: 'Known' })
    await field(page, 'condition-' + key).fill(String(value))
  }
}

// Read only the colored plot line. The actual pixel coordinates establish the visible time range.
async function voltageProfile(canvas, fractions) {
  return canvas.evaluate((element, fractions) => {
    const { width, height } = element
    const pixels = element.getContext('2d').getImageData(0, 0, width, height).data
    const columns = []
    for (let x = 0; x < width; x++) {
      const ys = []
      for (let y = 0; y < height; y++) {
        const p = (y * width + x) * 4
        if (pixels[p+3] > 195 && Math.abs(pixels[p]-41) < 15 &&
            Math.abs(pixels[p+1]-103) < 15 && Math.abs(pixels[p+2]-151) < 15) ys.push(y)
      }
      if (ys.length) columns.push([x, ys[Math.floor(ys.length / 2)]])
    }
    if (columns.length < width / 2) return []
    const first = columns[0][0], last = columns.at(-1)[0]
    return fractions.map(f => columns.find(([x]) => x >= first + f * (last-first))[1])
  }, fractions)
}

for (const format of ['csv', 'raw']) {
  test(format + ' waveform preview, plot and native download preserve independent SI evidence', async ({ page }) => {
    const errors = []
    page.on('pageerror', e => errors.push(e.message))
    if (format === 'csv') await page.setViewportSize({ width: 390, height: 844 })
    const bytes = fixture(format)
    await upload(page, format, bytes)
    await selectMappings(page, format)
    await nativeInput(page)
    const preview = await perform(page, 'preview', 'preview')
    await expect(field(page, 'sample-count')).toContainText('1001')
    const row = field(page, 'samples').locator('tbody tr').nth(1).locator('td')
    expect(Number(await row.nth(1).textContent())).toBeCloseTo(1e-9, 16)
    expect(Number(await row.nth(3).textContent())).toBe(5)
    const canvas = field(page, 'workbench').locator('canvas').first()
    await expect.poll(async () => {
      const [a, b, c, d] = await voltageProfile(canvas, [.1, .4, .7, .98])
      return a - b > 60 && c - d > 60 && Math.abs(a-c) < 4 && Math.abs(b-d) < 4
    }).toBe(true)
    await field(page, 'range-start').fill('1.5e-7')
    await field(page, 'range-end').fill('3e-7')
    await expect.poll(async () => {
      const [a, b, c] = await voltageProfile(canvas, [.15, .7, .85])
      return a - b > 60 && Math.abs(b-c) < 4
    }).toBe(true)
    await canvas.scrollIntoViewIfNeeded()
    await page.screenshot({ path: test.info().outputPath('measurement-' + format + '-waveform.png') })
    const result = await perform(page, 'analyze', 'extract')
    await expect(field(page, 'analysis')).toContainText('7.920000')
    const review = JSON.parse(readFileSync(await downloaded(page, 'review-download'), 'utf8'))
    expect(review.raw_record).toEqual(result.raw_record)
    expect(review.points).toEqual(result.points)
    expect(review.source.sha256).toBe(createHash('sha256').update(bytes).digest('hex'))
    expect(Buffer.from(review.source.original_base64, 'base64')).toEqual(bytes)
    expect(review.raw_record.gate_v).toEqual(preview.waveform.gate_v)
    const [on] = review.points.turn_on, [off] = review.points.turn_off
    expect([on.r_g, off.r_g]).toEqual([3, 9])
    expect(on.evidence.window_s[0]).toBeCloseTo(600.495e-9, 14)
    expect(off.evidence.window_s[1]).toBeCloseTo(249.5e-9, 14)
    const nativePath = await downloaded(page, 'native-download')
    const native = JSON.parse(readFileSync(nativePath, 'utf8'))
    expect(native).toEqual(result.core_json_document)
    const reload = [
      'import json, sys',
      'from pathlib import Path',
      'from transistordatabase.core.repository import JsonTransistorLoader',
      'device = JsonTransistorLoader().load_from_json(Path(sys.argv[1]))',
      'on, off = device.switch.e_on_data[0], device.switch.e_off_data[0]',
      'assert device.working_point["source_id"] == 9007199254740993',
      'print(json.dumps([on.e_x, off.e_x, device.switch.raw_measurement_data[0].evidence.sha256]))',
    ].join('\n')
    expect(JSON.parse(execFileSync(python, ['-c', reload, nativePath], { cwd: root }))).toEqual([on.e_x, off.e_x, review.source.sha256])
    const csv = readFileSync(await downloaded(page, 'csv-download'), 'utf8')
    expect(csv).toContain('turn_on')
    expect(csv).toContain('turn_off')
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual((await page.viewportSize()).width + 1)
    await field(page, 'analysis').scrollIntoViewIfNeeded()
    const energyCell = await field(page, 'analysis').locator('tbody tr').first().locator('td').boundingBox()
    expect(energyCell.x).toBeGreaterThanOrEqual(0)
    expect(energyCell.x + energyCell.width).toBeLessThanOrEqual((await page.viewportSize()).width)
    await page.screenshot({ path: test.info().outputPath('measurement-' + format + '-analysis.png') })
    await field(page, 'hash').scrollIntoViewIfNeeded()
    await page.screenshot({ path: test.info().outputPath('measurement-' + format + '-provenance.png') })
    await field(page, 'condition-r_g_off').fill('11')
    await expect(field(page, 'result')).toHaveCount(0)
    await expect(field(page, 'analyze')).toBeDisabled()
    expect(errors).toEqual([])
  })
}

test('30-step source log keeps supplied integrals separate from waveform analysis', async ({ page }) => {
  test.skip(!existsSync(sourceLog), 'Historical optional SC reference archive is absent')
  const bytes = readFileSync(sourceLog)
  await upload(page, 'log', bytes)
  for (const [name, value] of Object.entries({ energy_off: 'eoff', current: 'idc', bus: 'vdc', 'energy-unit': 'J', 'current-unit': 'A', 'bus-unit': 'V' })) {
    await field(page, 'log-' + name).selectOption(value)
  }
  await field(page, 'step-index').fill('2')
  const preview = await perform(page, 'preview', 'preview')
  await expect(field(page, 'log-records').locator('tbody tr')).toHaveCount(30)
  expect(preview.log_records.slice(0, 3).map(r => r.energy_j)).toEqual([1.0141e-7, 9.47901e-7, 2.69621e-6])
  await perform(page, 'analyze', 'extract')
  await expect(field(page, 'analysis')).toHaveCount(0)
  await expect(field(page, 'workbench').locator('canvas')).toHaveCount(0)
  const review = JSON.parse(readFileSync(await downloaded(page, 'review-download'), 'utf8'))
  const [point] = review.points.turn_off
  expect(review.raw_record).toBeNull()
  expect([point.i_x, point.v_supply, point.e_x]).toEqual([5, 5, 2.69621e-6])
  expect(point.evidence.window_s).toEqual([39e-6, 41e-6])
  expect(point.evidence.expression).toBeTruthy()
  await field(page, 'log-records').scrollIntoViewIfNeeded()
  await page.screenshot({ path: test.info().outputPath('measurement-original-log.png') })
})

test('genuine LTspice transient preserves all channel-current samples and generic scope', async ({ page }) => {
  test.skip(!sourceRaw, 'Set TDB_GENERIC_DPT_RAW to the November 10 LTspice binary')
  const bytes = readFileSync(sourceRaw)
  const scope = 'Generic uncalibrated NMOS simulation. Id(M_dut) is the channel current. The device name is a label.'
  await upload(page, 'raw', bytes, scope)
  await field(page, 'step-index').fill('0')
  for (const [role, name, unit] of [
    ['time', 'time', 's'], ['voltage', 'V(sw)', 'V'], ['current', 'Id(M_dut)', 'A'], ['gate', 'V(gate)', 'V'],
  ]) {
    await field(page, 'map-' + role).selectOption(name)
    await field(page, 'unit-' + role).selectOption(unit)
  }
  for (const [key, value] of Object.entries({ bus_voltage_v: 400, v_g_on: 15, v_g_off: -5, r_g_on: 10, r_g_off: 10 })) {
    await field(page, 'known-' + key).selectOption({ label: 'Known' })
    await field(page, 'condition-' + key).fill(String(value))
  }
  await perform(page, 'preview', 'preview')
  await expect(field(page, 'sample-count')).toContainText('18136')
  await perform(page, 'analyze', 'extract', false)
  await expect(field(page, 'analysis')).toBeVisible()
  const result = JSON.parse(readFileSync(await downloaded(page, 'review-download'), 'utf8'))
  expect(result.analysis.e_on).toBeCloseTo(.0001855909028250532, 14)
  expect(result.analysis.e_off).toBeCloseTo(.00017960023618601573, 14)
  expect(result.raw_record.time_s.at(-1)).toBe(18.04e-6)
  expect(result.raw_record.evidence.source_kind).toBe('simulation')
  expect(result.points.turn_off[0].evidence.disclosure).toBe(scope)
  expect(result.source.sha256).toBe('f851fb950bc03212585657b37d187d8ba9f18bd2ffd2aefe396ff8c0335dcca2')
  await field(page, 'analysis').scrollIntoViewIfNeeded()
  await page.screenshot({ path: test.info().outputPath('measurement-generic-simulation.png') })
})
