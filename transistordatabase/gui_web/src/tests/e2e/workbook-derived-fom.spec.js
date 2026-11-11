import { createHash } from 'node:crypto'
import { readFile } from 'node:fs/promises'
import { test, expect } from '@playwright/test'

const origin = 'http://localhost:8002'
const archivePath = '/home/tinix/claude_wsl/transistordatabase/transistordatabase/data/catalogs/archive_sc.json'
const button = (page, name) => page.getByRole('button', { name, exact: true })
async function download(page, name) {
  const waiting = page.waitForEvent('download')
  await button(page, name).click()
  return readFile(await (await waiting).path())
}
const exported = async page => JSON.parse(await download(page, 'Export JSON'))
async function ready(page) { await expect(page.locator('.summary-cards')).toBeVisible() }

test('derives real archive IGBT benchmarks only with declared estimate, currency and area evidence', async ({ page }) => {
  const original = await readFile(archivePath), catalog = JSON.parse(original)
  const sourceId = 'SemiconductorsData/IGBT'
  const record = catalog.records.find(row => row.source_id === sourceId && row.row === 2)
  const errors = []
  page.on('pageerror', error => errors.push(error.message))
  await page.goto(origin + '/index.html?view=catalog-fom')
  await ready(page)
  await page.getByLabel('Catalog sources', { exact: true }).selectOption(sourceId)
  await page.getByLabel('Part or manufacturer', { exact: true }).fill(record.part_number)
  await page.getByLabel('X metric', { exact: true }).selectOption('thermal_current_a')
  await page.getByLabel('Y metric', { exact: true }).selectOption('thermal_vi_per_price')
  await ready(page)
  expect((await exported(page)).result.candidates[0].metric_exclusions.thermal_current_a).toContain('acknowledgment-required')
  await page.getByLabel('Acknowledge thermal screening assumptions', { exact: true }).check()
  await ready(page)
  let proof = await exported(page), row = proof.result.candidates[0]
  expect(row.metrics.thermal_current_a).toBeCloseTo(45.60202526, 6)
  expect(row.metrics.thermal_power_budget_w).toBeCloseTo(70.75471698, 6)
  expect(row.metrics.thermal_vi_per_price).toBeNull()
  expect(row.metric_exclusions.thermal_vi_per_price).toContain('unknown-currency')
  expect(row.metrics.voltage_energy_benchmark_hz).toBeNull()
  expect(row.metric_exclusions.voltage_energy_benchmark_hz).toContain('err_j_a')
  expect(row.metrics.igbt_area_thermal_k_m2_w).toBeNull()
  expect(row.metric_exclusions.igbt_area_thermal_k_m2_w).toContain('unknown-area-unit')
  expect(row.metrics.rce_coes_s).toBeGreaterThan(0)
  expect(row.workbook_evidence.conditions.conduction_model_temperature_c).toBe(record.raw['21'].value)
  await page.getByLabel(`Declared currency ${sourceId}`, { exact: true }).fill('CHF')
  await page.getByLabel(`Declared area unit ${sourceId}`, { exact: true }).selectOption('mm2')
  await page.getByLabel('Comparable quote currency', { exact: true }).fill('CHF')
  await page.getByLabel('Recorded quote quantity', { exact: true }).fill('1000')
  await ready(page)
  proof = await exported(page); row = proof.result.candidates[0]
  expect(row.metrics.thermal_vi_per_price).toBeCloseTo(3989.41001622, 5)
  expect(row.metrics.igbt_area_thermal_k_m2_w).toBeNull()
  expect(row.metric_exclusions.igbt_area_thermal_k_m2_w).toContain('missing-input: igbt_area_source')
  expect(row.workbook_evidence.currency_basis).toBe('user declaration')
  expect(row.source.currency).toBe('unspecified')
  expect(row.raw).toEqual(record.raw)
  expect(row.formulas).toEqual(record.formulas)
  await button(page, record.part_number).click()
  await expect(page.getByLabel('Selected metric evidence')).toContainText('45.602')
  await expect(page.locator('.scatter-card').first().locator('.axis-labels')).toContainText('V·A/CHF')
  await expect(page.locator('.scatter-card').first().locator('.scatter-heading')).toContainText('1 points')
  const csv = (await download(page, 'Export CSV')).toString()
  expect(csv).toContain('"workbook_evidence_json"')
  expect(csv).toContain('source-constant-model-100C')
  expect(csv).toContain('user declaration')
  await page.locator('.scatter-card').first().scrollIntoViewIfNeeded()
  await page.screenshot({ path: test.info().outputPath('archive-thermal-benchmarks-desktop.png') })
  expect(await readFile(archivePath)).toEqual(original)
  expect(errors).toEqual([])
})

test('maps explicit source fields, preserves known zero and missing recovery, and plots declared SI benchmarks on mobile', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.addInitScript(() => {
    const fill = CanvasRenderingContext2D.prototype.fillText
    CanvasRenderingContext2D.prototype.fillText = function (text, ...args) {
      ;(this.canvas.workbookLabels ??= new Set()).add(String(text))
      return fill.call(this, text, ...args)
    }
  })
  const csv = Buffer.from('Part,V,I,Gate,Temp,R,V0,Tmax,Rth,Vtest,Eon,Eoff,Err,Coes,Qg,Area,RthJF,Price,Qty,Origin\nZERO,650,100,15,175,.014,.92,100,1.06,600,.00002,.00001,0,175,30,20,.5,7.43,1000,Independent quote evidence\nMISSING,650,100,15,175,.014,.92,175,1.06,600,.00002,.00001,,175,30,20,.5,7.43,1000,Independent quote evidence\n')
  await page.goto(origin + '/index.html?view=catalog-sources')
  await page.getByLabel('Catalog source file').setInputFiles({ name: 'declared-igbt.csv', mimeType: 'text/csv', buffer: csv })
  await button(page, 'Preview source').click()
  await expect(page.getByLabel('Source cell preview')).toContainText('ZERO')
  await expect(page.getByLabel('Workbook conduction profile').locator('option:checked')).toHaveText('Unknown · no workbook derivation')
  await page.getByLabel('Workbook conduction profile').selectOption('igbt')
  const fields = [
    ['part_number', 1], ['voltage_v', 2, 'V'], ['current_a', 3, 'A'], ['model_gate_voltage_v', 4, 'V'],
    ['model_temperature_c', 5, 'degC'], ['model_resistance_ohm', 6, 'Ohm'], ['model_threshold_v', 7, 'V'],
    ['junction_limit_c', 8, 'degC'], ['thermal_path_k_w', 9, 'K/W'], ['switch_voltage_v', 10, 'V'],
    ['eon_j_a', 11, 'J/A'], ['eoff_j_a', 12, 'J/A'], ['err_j_a', 13, 'J/A'], ['coes_f', 14, 'pF'],
    ['source_qg_c', 15, 'nC'], ['igbt_area_m2', 16, 'mm2'], ['igbt_rth_jf_k_w', 17, 'K/W'],
    ['price', 18, 'source currency'], ['quote_quantity', 19, 'pieces'], ['quote_origin', 20],
  ]
  for (const [field, column, unit] of fields) {
    await page.getByLabel(`Map ${field}`, { exact: true }).selectOption(String(column))
    if (unit) await page.getByLabel(`Unit ${field}`, { exact: true }).selectOption(unit)
  }
  await page.getByLabel('Source technology', { exact: true }).fill('IGBT')
  await button(page, 'Create catalog snapshot').click()
  const snapshot = JSON.parse(await download(page, 'Download catalog JSON'))
  expect(snapshot.source_file.sha256).toBe(createHash('sha256').update(csv).digest('hex'))
  expect(snapshot.records[0].err_j_a).toBe(0)
  expect(snapshot.records[1].err_j_a).toBeNull()
  expect(snapshot.records[0].igbt_area_m2).toBeCloseTo(20e-6, 14)
  await button(page, 'Explore snapshot in FOM').click()
  await page.getByLabel('X metric', { exact: true }).selectOption('thermal_current_a')
  await page.getByLabel('Y metric', { exact: true }).selectOption('voltage_v')
  await page.getByLabel('Acknowledge thermal screening assumptions', { exact: true }).check()
  await ready(page)
  let proof = await exported(page)
  expect(proof.result.candidates.find(row => row.part_number === 'ZERO').metrics.thermal_current_a).toBe(0)
  expect(proof.result.candidates.find(row => row.part_number === 'ZERO').metrics.voltage_energy_benchmark_hz).toBeCloseTo(20e6, 4)
  expect(proof.result.candidates.find(row => row.part_number === 'MISSING').metrics.voltage_energy_benchmark_hz).toBeNull()
  await expect(page.locator('.scatter-card').first().locator('.scatter-heading')).toContainText('2 points')
  await page.getByLabel('Y metric', { exact: true }).selectOption('voltage_energy_benchmark_hz')
  await ready(page)
  await expect(page.locator('.scatter-card').first().locator('.scatter-heading')).toContainText('1 points')
  await expect.poll(() => page.locator('.scatter-card').first().locator('canvas').evaluate(canvas => [...canvas.workbookLabels || []])).toEqual(expect.arrayContaining(['Test voltage / aggregate energy-current coefficient (Hz)']))
  await button(page, 'MISSING').click()
  await expect(page.getByLabel('Selected metric evidence')).toContainText('mapped-source-input: missing-or-nonnumeric: err_j_a')
  await expect(page.getByLabel('Selected metric evidence')).toContainText('Missing')
  const sourceId = snapshot.sources[0].id
  await page.getByLabel(`Declared currency ${sourceId}`, { exact: true }).fill('CHF')
  await page.getByLabel('X metric', { exact: true }).selectOption('igbt_area_thermal_k_m2_w')
  await page.getByLabel('Y metric', { exact: true }).selectOption('rce_coes_s')
  await ready(page)
  proof = await exported(page)
  expect(proof.result.candidates[0].metrics.igbt_area_thermal_k_m2_w).toBeCloseTo(1e-5, 14)
  expect(proof.result.candidates[0].metrics.rce_coes_s).toBeCloseTo(.014 * 175e-12, 20)
  await expect(page.locator('.scatter-card').first().locator('.axis-labels')).toContainText('K·m²/W')
  await page.locator('.scatter-card').first().scrollIntoViewIfNeeded()
  await page.screenshot({ path: test.info().outputPath('mapped-workbook-benchmarks-mobile.png') })
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390)
  expect((await download(page, 'Export CSV')).toString()).toContain(snapshot.source_file.sha256)
})

test('calculates inverse gate resistance through the compiled form with explicit acknowledgment and source proof', async ({ page }) => {
  await page.goto(origin + '/index.html')
  await page.getByRole('button', { name: 'More', exact: false }).click()
  await button(page, 'Analytical Models').click()
  await button(page, 'Device Parameter Operations').click()
  await page.getByLabel('Parameter calculation').selectOption('gate-resistance-estimate')
  const calculate = button(page, 'Calculate device parameters')
  await expect(calculate).toBeDisabled()
  await page.getByRole('textbox', { name: 'Data source / reference', exact: true }).fill('Independent inverse algebra fixture; not a measured device')
  for (const [name, value] of Object.entries({ target_dv_dt_v_s: '20000000000', c_rss_min_f: '1e-10', v_drive_high_v: '15', v_drive_low_v: '-5', v_plateau_low_v: '4', v_plateau_high_v: '7' })) await page.locator(`[name=${name}]`).fill(value)
  await page.getByRole('checkbox', { name: 'Use the gate resistance from target voltage slew with its stated assumptions', exact: true }).check()
  await calculate.click()
  const result = page.getByLabel('Device parameter results')
  await expect(result).toContainText('Minimum turn-on total series resistance (Ω)')
  await expect(result).toContainText('5.5')
  await expect(result).toContainText('minimum total series resistance')
  const proof = JSON.parse(await download(page, 'Download reproducible result'))
  expect(proof.result.r_on_ohm).toBeCloseTo(5.5, 12)
  expect(proof.result.r_off_ohm).toBeCloseTo(6, 12)
  expect(proof.result.model_source).toBe('SC mfiles/calcRg.m; sc-92afe9737e47b855')
  expect(proof.inputs.selected).toBe(true)
  await result.scrollIntoViewIfNeeded()
  await page.screenshot({ path: test.info().outputPath('inverse-gate-resistance-desktop.png') })
})
