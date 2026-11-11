import { expect, test } from '@playwright/test'
import { readFile } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { xyzText } from '../fixtures/xyzEnergy.js'

const field = (page, name) => page.getByLabel(`XYZ ${name}`, { exact: true })
const endpoint = '/api/analysis/energy-map/xyz'
async function ready(page) {
  await page.goto('/index.html?view=energy-map')
  await page.getByLabel('Energy map input', { exact: true }).selectOption('xyz')
  await field(page, 'source text').fill(xyzText)
  await field(page, 'source identifier').fill('=HYPERLINK("source evidence")')
  await field(page, 'energy unit').selectOption('µJ')
  await field(page, 'energy convention').fill('Original recorded switching energy; no synthetic samples')
  await field(page, 'common conditions').check()
}
async function parse(page, status = 200) {
  const waiting = page.waitForResponse(response => response.url().endsWith(endpoint) && response.request().method() === 'POST')
  await page.getByTestId('xyz-parse').click()
  const response = await waiting, raw = await response.text()
  expect(response.status(), raw).toBe(status)
  if (status === 200) await expect(page.getByTestId('xyz-results')).toBeVisible()
  else { await expect(page.getByTestId('xyz-error')).toBeVisible(); await expect(page.getByTestId('xyz-results')).toHaveCount(0) }
  return { data: JSON.parse(raw), raw, request: response.request().postDataJSON() }
}
async function download(page, id) {
  const waiting = page.waitForEvent('download'); await page.getByTestId(id).click()
  const file = await waiting
  return { name: file.suggestedFilename(), bytes: await readFile(await file.path()) }
}
async function matrixProof(page, data) {
  const matrix = page.getByTestId('xyz-matrix')
  const evidence = await matrix.locator('tbody td').evaluateAll(cells => cells.map(cell => {
    const box = cell.getBoundingClientRect()
    return { i: Number(cell.dataset.current), v: Number(cell.dataset.voltage), e: cell.dataset.energy, status: cell.dataset.status, x: box.x, y: box.y, width: box.width, height: box.height, color: getComputedStyle(cell).backgroundColor, text: cell.textContent }
  }))
  expect(evidence).toHaveLength(data.currents_a.length * data.voltages_v.length)
  for (const row of data.cells) for (const expected of row) {
    const cell = evidence.find(cell => cell.i === expected.current_a && cell.v === expected.voltage_v)
    expect(cell.status).toBe(expected.status); expect(cell.e).toBe(expected.energy_j === null ? 'NULL' : String(expected.energy_j))
    expect(cell.width).toBeGreaterThan(0); expect(cell.height).toBeGreaterThan(0)
  }
  const low = evidence.filter(cell => cell.v === 200), high = evidence.filter(cell => cell.v === 600)
  if (low.length && high.length) {
    expect(high[0].y).toBeLessThan(low[0].y)
    expect(low.map(cell => cell.i)).toEqual(data.currents_a)
    expect(low[0].x).toBeLessThan(low[1].x); expect(low[1].x).toBeLessThan(low[2].x)
  }
  return evidence
}

for (const width of [1440,390]) {
  test(`compiled XYZ paste, holes, duplicate evidence and exact downloads at ${width}px`, async ({ page }, info) => {
    const errors = []; page.on('pageerror', error => errors.push(error.message))
    await page.setViewportSize({ width, height: 900 }); await ready(page)
    expect(await page.locator('form').count()).toBe(1)
    const accepted = await parse(page)
    expect(accepted.data.currents_a).toEqual([0,3,9]); expect(accepted.data.voltages_v).toEqual([200,600])
    expect(accepted.data.cells.map(row => row.map(cell => cell.energy_j))).toEqual([[0,.00003,.00006],[.000012,null,.00009]])
    expect(accepted.data.census).toMatchObject({ source_rows: 5, available_cells: 5, missing_cells: 1, ambiguous_cells: 0 })
    expect(accepted.data.source.sha256).toBe(createHash('sha256').update(xyzText).digest('hex'))
    const proof = await matrixProof(page, accepted.data)
    expect(proof.find(cell => cell.i === 0 && cell.v === 200).text).toContain('0 J')
    expect(proof.find(cell => cell.i === 3 && cell.v === 600).text).toContain('hole')
    expect(proof.find(cell => cell.i === 0 && cell.v === 200).color).not.toBe(proof.find(cell => cell.i === 9 && cell.v === 600).color)
    expect((await download(page, 'xyz-json')).bytes.toString('utf8')).toBe(accepted.raw)
    expect((await download(page, 'xyz-source-download')).bytes).toEqual(Buffer.from(xyzText))
    const sourceCsv = (await download(page, 'xyz-rows-csv')).bytes.toString('utf8'), cellCsv = (await download(page, 'xyz-cells-csv')).bytes.toString('utf8')
    expect(sourceCsv).toContain('"\'=HYPERLINK'); expect(sourceCsv).toContain('9,600,90'); expect(sourceCsv).toContain(accepted.data.source.sha256)
    expect(cellCsv).toContain('"NULL"'); expect(cellCsv).toContain('NULL means unknown or unavailable; 0 is recorded zero')
    await page.getByTestId('xyz-matrix').evaluate(node => node.scrollIntoView({ block: 'center' }))
    await page.screenshot({ path: info.outputPath(`xyz-matrix-${width}.png`) })
    await page.getByTestId('xyz-results').screenshot({ path: info.outputPath(`xyz-results-${width}.png`) })
    await field(page, 'source text').fill(`${xyzText}\n3,600,45`); await expect(page.getByTestId('xyz-results')).toHaveCount(0)
    const complete = await parse(page); expect(complete.data.census.available_cells).toBe(6); expect(complete.data.cells[1][1].energy_j).toBe(.000045)
    await field(page, 'source text').fill(`${xyzText}\n3,600,45\n3,600,46`)
    const duplicate = await parse(page); expect(duplicate.data.cells[1][1]).toMatchObject({ energy_j: null, status: 'ambiguous', row_ids: ['line-6','line-7'] })
    expect(duplicate.data.rows.slice(-2).map(row => row.energy_j)).toEqual([.000045,.000046])
    await page.getByTestId('xyz-matrix').locator('td[data-current="3"][data-voltage="600"] button').click()
    await expect(page.getByTestId('xyz-cell-evidence')).toContainText('3,600,45'); await expect(page.getByTestId('xyz-cell-evidence')).toContainText('3,600,46')
    await field(page, 'source text').fill(`${xyzText}\n3,600,NA`); await field(page, 'missing energy tokens').fill('NA')
    const unknown = await parse(page); expect(unknown.data.cells[1][1]).toMatchObject({ energy_j: null, status: 'unknown' })
    await expect(page.getByTestId('xyz-matrix').locator('td[data-current="3"][data-voltage="600"]')).toContainText('NULL J')
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(width + 1)
    await page.getByLabel('Energy map input', { exact: true }).selectOption('prepared')
    await expect(page.getByLabel('Map currents', { exact: true })).toBeVisible(); await expect(page.getByTestId('xyz-json')).toHaveCount(0)
    expect(errors).toEqual([])
  })
}

test('original BOM/CRLF file bytes, reordered TSV columns, declared units and actual source download', async ({ page }) => {
  await ready(page); await field(page, 'source input').selectOption('file')
  const bytes = Buffer.from('\uFEFFenergy\tvoltage\tcurrent\r\n90\t.6\t9000\r\n0\t0\t0\r\n', 'utf8')
  await field(page, 'source file').setInputFiles({ name: 'original-µJ.tsv', mimeType: 'text/tab-separated-values', buffer: bytes })
  await field(page, 'has header').check(); await field(page, 'delimiter').selectOption('tsv')
  await field(page, 'current column').selectOption('2'); await field(page, 'energy column').selectOption('0')
  await field(page, 'current unit').selectOption('mA'); await field(page, 'voltage unit').selectOption('kV')
  const response = await parse(page)
  expect(Buffer.from(response.request.content_base64, 'base64')).toEqual(bytes)
  expect(response.data.source).toMatchObject({ sha256: createHash('sha256').update(bytes).digest('hex'), hash_scope: 'original-file-bytes', byte_count: bytes.length })
  expect(response.data.currents_a).toEqual([0,9]); expect(response.data.voltages_v).toEqual([0,600])
  expect(response.data.rows[0].lexemes).toEqual(['90','.6','9000']); expect(response.data.cells[1][1].energy_j).toBe(.00009)
  const source = await download(page, 'xyz-source-download'); expect(source.name).toBe('original-µJ.tsv'); expect(source.bytes).toEqual(bytes)
  expect((await download(page, 'xyz-json')).bytes.toString('utf8')).toBe(response.raw)
  await field(page, 'voltage unit').selectOption('V'); await expect(page.getByTestId('xyz-results')).toHaveCount(0)
  await expect(page.getByTestId('xyz-source-download')).toHaveCount(0)
})

test('singleton observed V0, whitespace, blank energy, repeated identical coordinates and parser errors', async ({ page }) => {
  await ready(page); await field(page, 'source text').fill('0 0 0'); await field(page, 'delimiter').selectOption('whitespace')
  const singleton = await parse(page); expect(singleton.data.cells[0][0]).toMatchObject({ current_a: 0, voltage_v: 0, energy_j: 0, status: 'available' })
  await field(page, 'delimiter').selectOption('csv'); await field(page, 'source text').fill('0,0,')
  const blank = await parse(page); expect(blank.data.cells[0][0].status).toBe('unknown'); expect(blank.data.cells[0][0].energy_j).toBe(null)
  await field(page, 'source text').fill('0,0,0\n0,0,0'); expect((await parse(page)).data.cells[0][0].status).toBe('ambiguous')
  for (const source of ['-1,0,0','0,0,1e-999','0,0,NaN','0,0,1,2']) {
    await field(page, 'source text').fill(source); const rejected = await parse(page,422); expect(rejected.data.detail).toBeTruthy()
    await expect(page.getByTestId('xyz-json')).toHaveCount(0)
  }
  await field(page, 'source text').fill('0,0,0'); await field(page, 'Junction temperature (°C)').fill('1e-400')
  await page.getByTestId('xyz-parse').click(); await expect(page.getByTestId('xyz-error')).toContainText('too small')
  await field(page, 'Junction temperature (°C)').fill('0'); expect((await parse(page)).data.conditions.t_j_c).toBe(0)
  await field(page, 'Junction temperature (°C)').fill(''); expect((await parse(page)).data.conditions.t_j_c).toBe(null)
})

test('source byte and axis budgets, stale response discard and invalidation after a file change', async ({ page }) => {
  await ready(page)
  await field(page, 'source text').fill(Array.from({ length: 257 }, (_, index) => `${index},200,1`).join('\n'))
  expect((await parse(page,413)).data.detail).toMatch(/256|current|axis/i)
  let posts = 0; page.on('request', request => { if (request.url().endsWith(endpoint) && request.method() === 'POST') posts++ })
  await field(page, 'source text').evaluate(node => { node.value = 'é'.repeat(2_000_001); node.dispatchEvent(new Event('input', { bubbles: true })) })
  await page.getByTestId('xyz-parse').click(); await expect(page.getByTestId('xyz-error')).toContainText('4 MB'); expect(posts).toBe(0)
  await field(page, 'source input').selectOption('file')
  await field(page, 'source file').setInputFiles({ name: 'too-big.csv', mimeType: 'text/csv', buffer: Buffer.alloc(4_000_001, '0') })
  await page.getByTestId('xyz-parse').click(); await expect(page.getByTestId('xyz-error')).toContainText('4 MB'); expect(posts).toBe(0)
  await field(page, 'source file').setInputFiles({ name: 'source.csv', mimeType: 'text/csv', buffer: Buffer.from(xyzText) }); await parse(page)
  await field(page, 'source file').setInputFiles({ name: 'changed.csv', mimeType: 'text/csv', buffer: Buffer.from('0,0,0') })
  await expect(page.getByTestId('xyz-results')).toHaveCount(0); await expect(page.getByTestId('xyz-json')).toHaveCount(0)
  await field(page, 'source input').selectOption('paste'); await field(page, 'source text').fill(xyzText)
  let release; const held = new Promise(resolve => { release = resolve })
  await page.route(`**${endpoint}`, async route => { const response = await route.fetch(); await held; await route.fulfill({ response }) })
  const sent = page.waitForRequest(request => request.url().endsWith(endpoint) && request.method() === 'POST')
  await page.getByTestId('xyz-parse').click(); await sent; await field(page, 'source identifier').fill('Changed during request'); release()
  await page.waitForTimeout(250); await expect(page.getByTestId('xyz-results')).toHaveCount(0); await expect(page.getByTestId('xyz-json')).toHaveCount(0)
})
