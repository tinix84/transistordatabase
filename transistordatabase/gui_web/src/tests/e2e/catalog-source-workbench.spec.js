import { readFile } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { test, expect } from '@playwright/test'

const appUrl = 'http://localhost:8002/index.html?view=catalog-sources'
const archive = '/home/tinix/claude_wsl/archive_ntbees2/sc/db'
const csv = Buffer.from('\uFEFFPart;Voltage;Resistance;Price;Maker\r\n"A,""quoted""\npart";"1,2kV";"40mOhm @ 20A, 18V";"12,5";"Maker;Name"\r\nB;600V;20mOhm;3;Beta\r\n')

async function previewCsv(page) {
  await page.goto(appUrl)
  await page.getByLabel('Catalog source file', { exact: true }).setInputFiles({ name: 'quoted.csv', mimeType: 'text/csv', buffer: csv })
  await page.getByLabel('CSV separator', { exact: true }).selectOption(';')
  await page.getByLabel('Decimal convention', { exact: true }).selectOption(',')
  await page.getByRole('button', { name: 'Preview source', exact: true }).click()
  await expect(page.getByLabel('Source cell preview', { exact: true })).toContainText('A,"quoted"\npart')
}
async function mapCsv(page) {
  for (const [field, column] of Object.entries({ part_number: '1', voltage_v: '2', rds_ohm: '3', price: '4', manufacturer: '5' })) {
    await page.getByLabel(`Map ${field}`, { exact: true }).selectOption(column)
  }
  await page.getByLabel('Unit rds_ohm', { exact: true }).selectOption('mOhm')
  await page.getByLabel('Unit voltage_v', { exact: true }).selectOption('V')
  await page.getByLabel('Unit price', { exact: true }).selectOption('source currency')
  await page.getByLabel('Recorded currency', { exact: true }).fill('CHF')
  await page.getByLabel('Recorded quote quantity', { exact: true }).fill('1000')
  await page.getByLabel('Source snapshot context', { exact: true }).fill('Historical supplied quote; date unknown')
}

test('compiled source upload maps quoted CSV, downloads exact bytes and explores its snapshot in FOM', async ({ page }) => {
  const errors = []; page.on('pageerror', e => errors.push(e.message))
  await previewCsv(page); await mapCsv(page)
  const exactDownload = page.waitForEvent('download')
  await page.getByRole('button', { name: 'Download exact source bytes', exact: true }).click()
  expect(await readFile(await (await exactDownload).path())).toEqual(csv)
  await page.getByRole('button', { name: 'Create catalog snapshot', exact: true }).click()
  await expect(page.getByLabel('Created catalog snapshot', { exact: true })).toContainText('2 accepted rows')
  const snapshotDownload = page.waitForEvent('download')
  await page.getByRole('button', { name: 'Download catalog JSON', exact: true }).click()
  const snapshot = JSON.parse(await readFile(await (await snapshotDownload).path(), 'utf8'))
  expect(snapshot.records[0]).toMatchObject({ part_number: 'A,"quoted"\npart', voltage_v: 1200, rds_ohm: .04, price: 12.5, manufacturer: 'Maker;Name' })
  expect(snapshot.source_file.sha256).toBe(createHash('sha256').update(csv).digest('hex'))
  expect(Buffer.from(snapshot.source_file.content_base64, 'base64')).toEqual(csv)
  expect(snapshot.records[0].conditions.junction_temperature_c).toBeNull()
  await page.getByRole('button', { name: 'Explore snapshot in FOM', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'FOM & Pareto explorer', exact: true })).toBeVisible()
  await page.getByLabel('X metric', { exact: true }).selectOption('voltage_v')
  await page.getByLabel('Y metric', { exact: true }).selectOption('rds_ohm')
  await expect(page.locator('.results-table')).toContainText('A,"quoted"')
  await expect(page.locator('.results-table')).toContainText('B')
  const fomDownload = page.waitForEvent('download')
  await page.getByRole('button', { name: 'Export JSON', exact: true }).click()
  const fom = JSON.parse(await readFile(await (await fomDownload).path(), 'utf8'))
  expect(fom.request.snapshot.records).toEqual(snapshot.records)
  expect(fom.result.matched_count).toBe(2)
  await page.screenshot({ path: test.info().outputPath('source-snapshot-fom.png'), fullPage: true })
  expect(errors).toEqual([])
})

test('compiled browser retains every real XLSM link and reads documented coefficient conditions without changing source bytes', async ({ page }) => {
  const sourcePath = `${archive}/SemiconductorsData.xlsm`, original = await readFile(sourcePath)
  await page.goto(appUrl)
  await page.getByLabel('Catalog source file', { exact: true }).setInputFiles(sourcePath)
  await page.getByRole('button', { name: 'Preview source', exact: true }).click()
  await expect(page.getByLabel('Worksheet census', { exact: true })).toContainText('MOSFET')
  await expect(page.getByLabel('Worksheet census', { exact: true })).toContainText('IGBT')
  await page.getByRole('button', { name: 'Inspect static links', exact: true }).click()
  await expect(page.locator('.source-workspace')).toContainText('59 stored link objects · 63 GetURL formulas (not evaluated)')
  const linksDownload = page.waitForEvent('download')
  await page.getByRole('button', { name: 'Download link evidence', exact: true }).click()
  const links = JSON.parse(await readFile(await (await linksDownload).path(), 'utf8'))
  expect(links.entries.filter(e => e.kind === 'stored-link' && e.sheet === 'MOSFET')).toHaveLength(23)
  expect(links.entries.filter(e => e.kind === 'stored-link' && e.sheet === 'IGBT')).toHaveLength(36)
  expect(links.geturl_formula_locations).toHaveLength(63)
  expect(await page.getByLabel('Static link evidence', { exact: true }).locator('a').count()).toBe(0)
  await page.getByLabel('Link cell batch', { exact: true }).fill('BA25')
  await page.getByRole('button', { name: 'Inspect static links', exact: true }).click()
  await expect(page.getByLabel('Static link evidence', { exact: true })).toContainText('absent')
  await page.getByRole('button', { name: 'Inspect J/A coefficients', exact: true }).click()
  await expect(page.getByLabel('Switching coefficient evidence', { exact: true })).toContainText('MOSFET!Y2')
  const coefficientDownload = page.waitForEvent('download')
  await page.getByRole('button', { name: 'Download coefficient evidence', exact: true }).click()
  const evidence = JSON.parse(await readFile(await (await coefficientDownload).path(), 'utf8'))
  const coefficient = evidence.coefficients.find(c => c.sheet === 'MOSFET' && c.cell.cell === 'Y2')
  expect(coefficient.recognized_literal_coefficient_j_per_a).toBeCloseTo(.000632 / 36, 16)
  expect(coefficient.reference_current_a).toBe(36)
  expect(coefficient.reference_temperature_c).toBe(175)
  expect(coefficient.unit).toBe('J/A')
  expect(coefficient.gate_resistance_ohm).toBeNull()
  expect(await readFile(sourcePath)).toEqual(original)
  await page.screenshot({ path: test.info().outputPath('source-xlsm-evidence.png'), fullPage: true })
})

test('compiled explicit refresh comparison reports duplicate keys and added, changed, missing rows', async ({ page }) => {
  await page.goto(appUrl)
  const source = { schema_version: 1, sources: [{ id: 'old' }], records: [{ id: 'old:2', row: 2, part_number: 'P', voltage_v: 600 }, { id: 'old:3', row: 3, part_number: 'P', voltage_v: 650 }, { id: 'old:4', row: 4, part_number: 'Missing', voltage_v: 400 }] }
  const refreshed = { schema_version: 1, sources: [{ id: 'new' }], records: [{ id: 'new:2', row: 2, part_number: 'P', voltage_v: 1200 }, { id: 'new:3', row: 3, part_number: 'P', voltage_v: 650 }, { id: 'new:5', row: 5, part_number: 'Added', voltage_v: 900 }] }
  for (const [side, value] of [['Before', source], ['After', refreshed]]) await page.getByLabel(`${side} snapshot JSON`, { exact: true }).setInputFiles({ name: `${side}.json`, mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(value)) })
  await page.getByLabel('Compare key part_number', { exact: true }).check()
  await page.getByRole('button', { name: 'Compare snapshots', exact: true }).click()
  await expect(page.getByRole('alert')).toContainText('disambiguate duplicate rows')
  await page.getByLabel('Compare key row', { exact: true }).check()
  await page.getByRole('button', { name: 'Compare snapshots', exact: true }).click()
  await expect(page.getByLabel('Snapshot comparison', { exact: true })).toContainText('1 added · 1 changed · 1 missing')
  await expect(page.getByLabel('Snapshot differences', { exact: true })).toContainText('voltage_v')
  await expect(page.getByLabel('Snapshot comparison', { exact: true })).toContainText('no stored source overwritten')
})

test('compiled mobile source workflow scrolls wide tables and ignores stale snapshot responses after edits', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await previewCsv(page); await mapCsv(page)
  let release, received
  const held = new Promise(resolve => { received = resolve })
  const unblock = new Promise(resolve => { release = resolve })
  await page.route('**/api/catalog-sources/snapshot', async route => {
    const response = await route.fetch(); received(); await unblock; await route.fulfill({ response })
  })
  const returned = page.waitForResponse('**/api/catalog-sources/snapshot')
  await page.getByRole('button', { name: 'Create catalog snapshot', exact: true }).click()
  await held
  await page.getByLabel('Unit rds_ohm', { exact: true }).selectOption('Ohm')
  release()
  await returned
  await expect(page.getByLabel('Created catalog snapshot', { exact: true })).toHaveCount(0)
  const widths = await page.locator('.source-workspace .table-wrap').evaluateAll(nodes => nodes.map(n => ({ width: n.getBoundingClientRect().width, scroll: n.scrollWidth, client: n.clientWidth, overflow: getComputedStyle(n).overflowX })))
  expect(widths.every(n => n.width > 0 && n.width <= 390 && n.overflow === 'auto')).toBe(true)
  expect(widths.some(n => n.scroll > n.client)).toBe(true)
  expect(await page.locator('.source-workspace').evaluate(n => n.getBoundingClientRect().right)).toBeLessThanOrEqual(390)
  await page.screenshot({ path: test.info().outputPath('source-mobile.png'), fullPage: true })
})
