import { createHash } from 'node:crypto'
import { readFile } from 'node:fs/promises'
import { existsSync } from 'node:fs'
import { test, expect } from '@playwright/test'

const origin = 'http://localhost:8002'
const xlsm = '/home/tinix/claude_wsl/archive_ntbees2/sc/db/SemiconductorsData.xlsm'
const hash = bytes => createHash('sha256').update(bytes).digest('hex')
const button = (page, name) => page.getByRole('button', { name, exact: true })
async function download(page, name) {
  const waiting = page.waitForEvent('download')
  await button(page, name).click()
  return readFile(await (await waiting).path())
}

test('maps source bytes, inspects actual chart units and reproduces both normalization populations', async ({ page }) => {
  const uncaught = []
  page.on('pageerror', event => uncaught.push(event.message))
  await page.setViewportSize({ width: 390, height: 844 })
  await page.addInitScript(() => {
    const renderText = CanvasRenderingContext2D.prototype.fillText
    CanvasRenderingContext2D.prototype.fillText = function (text, ...args) {
      const labels = this.canvas.reviewLabels ??= new Set()
      labels.add(String(text))
      return renderText.call(this, text, ...args)
    }
  })
  await page.goto(origin + '/index.html?view=catalog-sources')
  const bytes = Buffer.from('\uFEFFPart;V;I;R;Qg;Price\r\n"LOW;quoted";200;20;2;40;1\r\nMID;400;40;4;80;3\r\nTOP;800;80;8;;\r\n')
  await page.getByLabel('Catalog source file').setInputFiles({ name: 'independent.csv', mimeType: 'text/csv', buffer: bytes })
  await page.getByLabel('CSV separator').selectOption(';')
  await button(page, 'Preview source').click()
  await expect(page.getByLabel('Source cell preview')).toContainText('LOW;quoted')
  for (const [field, column, unit] of [
    ['part_number', '1'], ['voltage_v', '2', 'V'], ['current_a', '3', 'A'],
    ['rds_ohm', '4', 'mOhm'], ['qg_c', '5', 'nC'], ['price', '6', 'source currency'],
  ]) {
    await page.getByLabel('Map ' + field, { exact: true }).selectOption(column)
    if (unit) await page.getByLabel('Unit ' + field, { exact: true }).selectOption(unit)
  }
  await page.getByLabel('Recorded currency', { exact: true }).fill('CHF')
  await button(page, 'Create catalog snapshot').click()
  const saved = JSON.parse(await download(page, 'Download catalog JSON'))
  expect(saved.source_file.sha256).toBe(hash(bytes))
  expect(Buffer.from(saved.source_file.content_base64, 'base64')).toEqual(bytes)
  expect(saved.records.map(r => r.rds_ohm)).toEqual([.002, .004, .008])
  await button(page, 'Use as before snapshot').click()
  const newer = structuredClone(saved)
  newer.records[0].current_a = 21
  await page.getByLabel('After snapshot JSON').setInputFiles({ name: 'after.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(newer)) })
  await page.getByLabel('Compare key row').check()
  await button(page, 'Compare snapshots').click()
  await expect(page.getByLabel('Snapshot differences')).toContainText('current_a')
  const diff = JSON.parse(await download(page, 'Download comparison'))
  expect(diff.changed).toMatchObject([{ key: [2], fields: ['current_a'] }])
  await button(page, 'Explore snapshot in FOM').click()
  await page.getByLabel('Y metric', { exact: true }).selectOption('qg_c')
  const charts = page.locator('.scatter-card')
  await expect(charts.first().locator('.scatter-heading')).toContainText('2 points')
  await expect(charts.nth(2).locator('.scatter-heading')).toContainText('3 points')
  await expect(page.getByRole('row').filter({ has: button(page, 'TOP') })).toContainText('Unavailable')
  const labels = () => charts.first().locator('canvas').evaluate(c => [...(c.reviewLabels || [])])
  await expect.poll(labels).toEqual(expect.arrayContaining(['Gate charge (nC)', '40', '80']))
  await button(page, 'MID').click()
  await expect(page.getByLabel('Selected catalog device')).toContainText('Source row 3')
  const image = await download(page, 'Export JSON')
  expect(JSON.parse(image).result.points.map(p => p.metrics.qg_c).sort()).toEqual([4e-8, 8e-8])
  await expect.poll(labels).toEqual(expect.arrayContaining(['Gate charge (nC)', '40', '80']))
  await page.getByLabel('Axis values').selectOption('fixed')
  await page.getByLabel('Min voltage').fill('400')
  await expect(page.locator('.summary-cards strong').first()).toHaveText('2')
  const fixed = JSON.parse(await download(page, 'Export JSON')).result
  expect(fixed.normalization.metrics.rds_ohm.denominator).toBe(.002)
  expect(fixed.candidates.map(p => p.normalization.rds_ohm.relative)).toEqual([2, 4])
  await page.getByLabel('Axis values').selectOption('visible')
  await expect(page.locator('.method[open]').first()).toContainText('Filtered population')
  const visible = JSON.parse(await download(page, 'Export JSON')).result
  expect(visible.normalization.metrics.rds_ohm.denominator).toBe(.004)
  expect(visible.candidates.map(p => p.normalization.rds_ohm.relative)).toEqual([1, 2])
  expect(visible.normalization.population_hash).not.toBe(fixed.normalization.population_hash)
  const csv = (await download(page, 'Export CSV')).toString()
  expect(csv).toContain(saved.source_file.sha256)
  expect(csv).toContain(saved.snapshot_id)
  expect(csv).toContain(visible.normalization.population_hash)
  const layout = await page.evaluate(() => ({
    width: document.documentElement.scrollWidth,
    outside: [...document.querySelectorAll('main, section, article, .source-note, .source-note code, .filter-panel, .objectives, .plot-help, .method, aside, .summary-cards, .results-table')].filter(el => el.getBoundingClientRect().right > 390).map(el => [el.tagName, el.className, Math.round(el.getBoundingClientRect().right)]),
  }))
  expect(layout.width, JSON.stringify(layout)).toBeLessThanOrEqual(390)
  expect(uncaught).toEqual([])
})

test('reads links and physical coefficients from the actual XLSM without contacting a target', async ({ page }) => {
  test.skip(!existsSync(xlsm), 'The historical read-only source archive is unavailable')
  const raw = await readFile(xlsm)
  const external = []
  page.on('request', request => {
    const url = new URL(request.url())
    if (['http:', 'https:'].includes(url.protocol) && url.hostname !== 'localhost' && url.hostname !== '127.0.0.1') external.push(url.href)
  })
  await page.goto(origin + '/index.html?view=catalog-sources')
  await page.getByLabel('Catalog source file').setInputFiles(xlsm)
  await button(page, 'Preview source').click()
  await button(page, 'Inspect static links').click()
  const evidence = JSON.parse(await download(page, 'Download link evidence'))
  expect(evidence.source_file.sha256).toBe(hash(raw))
  expect(evidence.native_link_count).toBe(59)
  expect(evidence.geturl_formula_count).toBe(63)
  expect(evidence.entries.filter(e => e.kind === 'stored-link' && e.sheet === 'IGBT')).toHaveLength(36)
  await button(page, 'Inspect J/A coefficients').click()
  const row = page.getByLabel('Switching coefficient evidence').getByRole('row').filter({ hasText: 'MOSFET!Y2 · e_on' })
  await expect(row).toContainText('36 A')
  await expect(row).toContainText('600 V')
  await expect(row).toContainText('175 °C')
  const typed = JSON.parse(await download(page, 'Download coefficient evidence'))
  const y2 = typed.coefficients.find(c => c.sheet === 'MOSFET' && c.cell.cell === 'Y2')
  expect(y2.recognized_literal_coefficient_j_per_a).toBeCloseTo(.000632 / 36, 16)
  expect(y2.unit).toBe('J/A')
  expect(await download(page, 'Download exact source bytes')).toEqual(raw)
  expect(await readFile(xlsm)).toEqual(raw)
  expect(external).toEqual([])
})

// Delay a completed real HTTP response to exercise out-of-order browser delivery.
function deferred() {
  let resolve
  const promise = new Promise(done => { resolve = done })
  return { promise, resolve }
}

test('keeps the newer FOM filter after an older server response arrives', async ({ page }) => {
  await page.goto(origin + '/index.html?view=catalog-fom')
  await expect(page.locator('.summary-cards')).toBeVisible()
  const ready = deferred(), release = deferred(), delivered = deferred()
  await page.route('**/api/catalog/screen', async route => {
    if (route.request().postDataJSON().search !== 'NTHL020N120SC1') return route.continue()
    const response = await route.fetch()
    ready.resolve()
    await release.promise
    await route.fulfill({ response })
    delivered.resolve()
  })
  const filter = page.getByLabel('Part or manufacturer', { exact: true })
  await filter.fill('NTHL020N120SC1')
  await ready.promise
  try {
    await filter.fill('no-such-device-in-this-survey')
    await expect(page.locator('.summary-cards strong').first()).toHaveText('0')
  } finally { release.resolve() }
  await delivered.promise
  const saved = JSON.parse(await download(page, 'Export JSON'))
  expect(saved.request.search).toBe('no-such-device-in-this-survey')
  expect(saved.result.candidates).toEqual([])
  expect(saved.result.matched_count).toBe(0)
})

test('keeps the selected source after a late preview of different bytes', async ({ page }) => {
  await page.goto(origin + '/index.html?view=catalog-sources')
  const ready = deferred(), release = deferred(), delivered = deferred()
  await page.route('**/api/catalog-sources/preview', async route => {
    if (route.request().postDataJSON().filename !== 'previous.csv') return route.continue()
    const response = await route.fetch()
    ready.resolve()
    await release.promise
    await route.fulfill({ response })
    delivered.resolve()
  })
  const file = page.getByLabel('Catalog source file')
  await file.setInputFiles({ name: 'previous.csv', mimeType: 'text/csv', buffer: Buffer.from('part\nprevious-row\n') })
  await button(page, 'Preview source').click()
  await ready.promise
  const current = Buffer.from('part\ncurrent-row\n')
  try {
    await file.setInputFiles({ name: 'current.csv', mimeType: 'text/csv', buffer: current })
    await button(page, 'Preview source').click()
    await expect(page.getByLabel('Source cell preview')).toContainText('current-row')
  } finally { release.resolve() }
  await delivered.promise
  await expect(page.getByLabel('Source cell preview')).not.toContainText('previous-row')
  expect(await download(page, 'Download exact source bytes')).toEqual(current)
})
