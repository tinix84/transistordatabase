import { test, expect } from '@playwright/test'
import fs from 'node:fs/promises'
import { createHash } from 'node:crypto'

const root = '/home/tinix/claude_wsl/transistordatabase'
const archive = '/home/tinix/claude_wsl/archive_ntbees2/sc'
const enco = '/mnt/d/OneDrive/claude/bhy/enco'
async function open(page) {
  await page.goto('http://localhost:8002/index.html?view=plecs-data')
  await expect(page.getByRole('heading', { name: 'PLECS data workbench', exact: true })).toBeVisible()
}
async function observeCanvas(page) {
  await page.addInitScript(() => {
    const clear = CanvasRenderingContext2D.prototype.clearRect
    const arc = CanvasRenderingContext2D.prototype.arc
    const text = CanvasRenderingContext2D.prototype.fillText
    CanvasRenderingContext2D.prototype.clearRect = function (...args) { this.canvas.recordedDraw = { points: [], labels: [] }; return clear.apply(this, args) }
    CanvasRenderingContext2D.prototype.arc = function (x, y, radius, ...args) {
      const transform = this.getTransform()
      if (radius === 3 && this.canvas.recordedDraw) this.canvas.recordedDraw.points.push({ x: transform.a * x + transform.c * y + transform.e, y: transform.b * x + transform.d * y + transform.f })
      return arc.call(this, x, y, radius, ...args)
    }
    CanvasRenderingContext2D.prototype.fillText = function (label, x, y, ...args) {
      const transform = this.getTransform()
      if (this.canvas.recordedDraw && /^[0-9]+(?:\.[0-9]+)?$/.test(String(label))) this.canvas.recordedDraw.labels.push({ label: Number(label), x: transform.a * x + transform.c * y + transform.e, y: transform.b * x + transform.d * y + transform.f })
      return text.call(this, label, x, y, ...args)
    }
  })
}
async function inspect(page, filename) {
  await page.getByTestId('plecs-file').setInputFiles(filename)
  await expect(page.getByTestId('plecs-xml')).not.toHaveValue('')
  const pending = page.waitForResponse(response => response.url().endsWith('/api/plecs/preview') && response.request().method() === 'POST')
  await page.getByRole('button', { name: 'Inspect PLECS document', exact: true }).click()
  const response = await pending
  expect(response.status()).toBe(200)
  return response.json()
}
async function downloaded(page, name) {
  const pending = page.waitForEvent('download')
  await page.getByRole('button', { name, exact: true }).click()
  return fs.readFile(await (await pending).path())
}
test('compiled table document shows Python SI slices, exact source hash and semantic XML round trip', async ({ page, request }) => {
  const errors = []; page.on('pageerror', error => errors.push(error.message))
  await observeCanvas(page)
  await open(page)
  const filename = `${root}/tests/test_data/sample_plecs_switch.xml`, bytes = await fs.readFile(filename)
  const data = await inspect(page, filename), results = page.getByTestId('plecs-results')
  expect(data.source.sha256).toBe(createHash('sha256').update(bytes).digest('hex'))
  expect(data.source.hash_scope).toBe('original-bytes')
  expect(data.can_export).toBe(true); expect(data.packages[0].ratings).toBeNull()
  await expect(results).toContainText('Absolute voltage/current ratings: Unknown')
  await expect(results.locator('canvas')).toHaveCount(3)
  const table = data.packages[0].blocks[0].tables.find(value => value.kind === 'turn-on')
  const card = results.getByTestId('plecs-table').filter({ has: page.getByRole('heading', { name: /^turn-on ·/ }) })
  const index = table.slices.findIndex(value => value.temperature_c === 150 && value.supply_voltage_v === 800)
  await card.getByTestId('plecs-slice').selectOption(String(index))
  await expect(card).toContainText('Signed supply 800 V')
  await expect(card).toContainText('0.0027')
  const displayedPoints = await card.locator('tbody tr').evaluateAll(rows => rows.map(row => [...row.querySelectorAll('td')].map(cell => Number(cell.textContent))))
  expect(displayedPoints).toEqual(table.slices[index].plot_data)
  await expect.poll(() => card.locator('canvas').evaluate(canvas => {
    const pixels = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data
    let painted = 0
    for (let index = 3; index < pixels.length; index += 4) if (pixels[index] > 0) painted++
    return painted
  })).toBeGreaterThan(500)
  const renderedValue = () => card.locator('canvas').evaluate(canvas => {
    const draw = canvas.recordedDraw
    if (!draw?.points.length) return null
    const low = draw.labels.find(row => row.label === .0005), high = draw.labels.find(row => row.label === .0025)
    if (!low || !high) return null
    const endpoint = draw.points.reduce((left, right) => right.x > left.x ? right : left)
    return low.label + (endpoint.y - low.y) * (high.label - low.label) / (high.y - low.y)
  })
  await expect.poll(renderedValue).toBeCloseTo(.0027, 7)
  const json = JSON.parse((await downloaded(page, 'Download canonical document JSON')).toString('utf8'))
  expect(json).toEqual(data)
  expect(await downloaded(page, 'Download preserved source XML')).toEqual(bytes)
  // Export state must not restart unchanged curve animations after they settled.
  await page.waitForTimeout(100)
  expect(await renderedValue()).toBeCloseTo(.0027, 7)
  const rendered = await downloaded(page, 'Download supported table XML')
  const roundTrip = await request.post('http://localhost:8002/api/plecs/preview', { data: { xml_base64: rendered.toString('base64'), source_name: 'roundtrip.xml' } })
  expect(roundTrip.status()).toBe(200)
  const parsed = await roundTrip.json()
  expect(parsed.packages[0].blocks[0].tables.map(value => value.slices)).toEqual(data.packages[0].blocks[0].tables.map(value => value.slices))
  expect(parsed.packages[0].thermal.map(value => value.elements)).toEqual(data.packages[0].thermal.map(value => value.elements))
  await expect.poll(renderedValue).toBeCloseTo(.0027, 7)
  await card.screenshot({ path: test.info().outputPath('plecs-recorded-slice-desktop.png') })
  const mapping = page.getByTestId('plecs-core-mapping'); await mapping.locator('summary').click()
  await mapping.getByTestId('plecs-map-package').selectOption(data.packages[0].id)
  for (const [key, value] of Object.entries({ v_abs_max: 1200, i_abs_max: 80, i_cont: 40, t_j_max: 175, housing_area: .0001, cooling_area: .00005 })) await mapping.getByTestId(`plecs-declare-${key}`).fill(String(value))
  await mapping.getByTestId('plecs-target-0').selectOption('switch')
  for (const gate of await mapping.getByTestId('plecs-declare-gate').all()) await gate.fill('15')
  await mapping.getByTestId('plecs-recovery-convention').selectOption('preserve-signed')
  await mapping.getByTestId('plecs-thermal-owner').selectOption('switch')
  const finalizing = page.waitForResponse(response => response.url().endsWith('/api/plecs/finalize') && response.request().method() === 'POST')
  await mapping.getByRole('button', { name: 'Validate explicit core mapping', exact: true }).click()
  const finalResponse = await finalizing; expect(finalResponse.status()).toBe(200)
  const finalized = await finalResponse.json()
  await expect(mapping.getByTestId('plecs-core-result')).toContainText('finalized-not-persisted')
  expect(finalized.core.switch.channel_data).toHaveLength(2)
  expect(finalized.core.switch.e_on_data).toHaveLength(4)
  expect(JSON.parse((await downloaded(page, 'Download declared core JSON')).toString('utf8'))).toEqual(finalized.core_json_document)
  expect(await downloaded(page, 'Download source-backed core XML')).toEqual(Buffer.from(finalized.roundtrip_xml_base64, 'base64'))
  await mapping.screenshot({ path: test.info().outputPath('plecs-declared-core-desktop.png') })
  await mapping.getByTestId('plecs-declare-i_cont').fill('35')
  await expect(mapping.getByTestId('plecs-core-result')).toHaveCount(0)
  expect(await renderedValue()).toBeCloseTo(.0027, 7)
  expect(errors).toEqual([])
})

test('actual ENCO repeated conduction formulas and Cauer remain readable unsupported evidence on mobile', async ({ page }) => {
  const errors = []; page.on('pageerror', error => errors.push(error.message))
  await page.setViewportSize({ width: 390, height: 844 }); await open(page)
  const filename = `${enco}/enco/packages/enco-components/tests/mosfet/_mock/G3R40MT12K.xml`
  const data = await inspect(page, filename), result = page.getByTestId('plecs-results')
  expect(data.status).toBe('unsupported-content'); expect(data.can_export).toBe(false)
  expect(data.packages[0].blocks).toHaveLength(1)
  expect(data.packages[0].blocks[0].tables.filter(value => value.kind === 'conduction')).toHaveLength(2)
  await expect(result).toContainText('Unexecuted formula')
  await expect(result).toContainText('CustomTables')
  await expect(result).toContainText('Cauer chain retained as source')
  const cauer = result.locator('.thermal-evidence').filter({ hasText: 'Cauer chain retained as source' })
  await expect(cauer).toContainText('Block')
  await expect(cauer.locator('tbody tr').first()).toHaveText('10.0050505Unknown0.00018279')
  await expect(result.locator('.thermal-evidence').filter({ hasText: 'Package thermal networks' })).toContainText('StateSpaceModel')
  await expect(page.getByRole('button', { name: 'Download supported table XML', exact: true })).toBeDisabled()
  const supportedTables = data.packages.flatMap(pkg => pkg.blocks.flatMap(block => block.tables.filter(value => value.status === 'supported-table' && value.slices.length)))
  await expect(result.locator('canvas')).toHaveCount(supportedTables.length)
  expect(await downloaded(page, 'Download preserved source XML')).toEqual(await fs.readFile(filename))
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
  await result.locator('.diagnostics').screenshot({ path: test.info().outputPath('plecs-unsupported-mobile-report.png') })
  await result.getByTestId('plecs-table').first().screenshot({ path: test.info().outputPath('plecs-source-table-mobile.png') })
  await cauer.screenshot({ path: test.info().outputPath('plecs-cauer-mobile.png') })
  expect(errors).toEqual([])
})

test('actual SC diode retains signed supply convention and recorded curves without invented ratings', async ({ page }) => {
  await open(page)
  const data = await inspect(page, `${archive}/db/plecs/Infineon_IGBT/IKW20N60TA_Diode.xml`)
  expect(data.packages[0].ratings).toBeNull()
  const negative = data.packages.flatMap(pkg => pkg.blocks.flatMap(block => block.tables.flatMap(table => table.slices))).filter(slice => slice.supply_voltage_v < 0)
  expect(negative.length).toBeGreaterThan(0)
  await expect(page.getByTestId('plecs-results')).toContainText('Signed supply -')
  await expect(page.getByTestId('plecs-results')).toContainText('Absolute voltage/current ratings: Unknown')
})

test('held successful preview is discarded when source changes and malformed input displays a safe error', async ({ page }) => {
  await open(page)
  await page.getByTestId('plecs-file').setInputFiles(`${root}/tests/test_data/sample_plecs_switch.xml`)
  await expect(page.getByTestId('plecs-xml')).not.toHaveValue('')
  let release, received
  const held = new Promise(resolve => { release = resolve }), started = new Promise(resolve => { received = resolve })
  await page.route('**/api/plecs/preview', async route => { const response = await route.fetch(); received(); await held; await route.fulfill({ response }) })
  await page.getByRole('button', { name: 'Inspect PLECS document', exact: true }).click(); await started
  await page.getByTestId('plecs-source').fill('Changed source identity'); release()
  await expect(page.getByRole('button', { name: 'Inspect PLECS document', exact: true })).toBeEnabled()
  await expect(page.getByTestId('plecs-results')).toHaveCount(0)
  await page.unroute('**/api/plecs/preview')
  await page.getByTestId('plecs-xml').fill('<SemiconductorLibrary>')
  await page.getByRole('button', { name: 'Inspect PLECS document', exact: true }).click()
  await expect(page.getByRole('alert')).toBeVisible()
  await expect(page.getByTestId('plecs-results')).toHaveCount(0)
})

test('compiled generic core export selects exact contexts and preserves original recorded SI corners at 390px', async ({ page, request }) => {
  const errors = []; page.on('pageerror', error => errors.push(error.message))
  await observeCanvas(page)
  // The fixture serves the production app with an in-memory repository. No stored
  // database record is created, overwritten or removed by this browser workflow.
  await page.route('**/api/transistors', async route => { const response = await route.fetch({ url: 'http://localhost:8031/api/transistors' }); await route.fulfill({ response }) })
  await page.route('**/api/plecs/core/**', async route => { const response = await route.fetch({ url: route.request().url().replace('localhost:8002', 'localhost:8031') }); await route.fulfill({ response }) })
  await page.setViewportSize({ width: 390, height: 844 }); await open(page)
  const form = page.getByTestId('plecs-core-export')
  await expect(form.getByTestId('plecs-existing-device')).toHaveValue('')
  await form.getByTestId('plecs-existing-device').selectOption('bounded-a')
  await form.getByTestId('plecs-existing-component').selectOption('switch')
  for (const key of ['conduction', 'turn_on', 'turn_off']) {
    await expect(form.getByTestId(`plecs-include-${key}`)).not.toBeChecked()
    await form.getByTestId(`plecs-include-${key}`).check()
    await expect(form.getByTestId(`plecs-context-${key}`)).toHaveValue('')
    await form.getByTestId(`plecs-context-${key}`).selectOption('0')
  }
  await form.getByTestId('plecs-omit-thermal').fill('Thermal model is absent from this supplied algebraic fixture')
  await form.getByTestId('plecs-core-convention').selectOption('preserve-signed')
  const pending = page.waitForResponse(response => response.url().endsWith('/api/plecs/core/bounded-a/preview') && response.request().method() === 'POST')
  await form.getByRole('button', { name: 'Preview selected core export', exact: true }).click()
  const response = await pending; expect(response.status()).toBe(200)
  const data = await response.json(), result = form.getByTestId('plecs-core-export-result')
  expect(data.report.selection.turn_on).toEqual({ gate_voltage_v: 15, gate_resistance_ohm: 4 })
  expect(data.report.selection.turn_off).toEqual({ gate_voltage_v: 15, gate_resistance_ohm: 7 })
  expect(data.report.evidence[1].additional_conditions.gate_off_voltage_v).toBe(-3)
  expect(data.report.omitted_mechanisms).toEqual(['thermal'])
  const tables = data.document.packages[0].blocks[0].tables
  expect(tables[0].slices[0].plot_data).toEqual([[0, 0], [10, 2], [20, 4]])
  expect(tables[1].slices[0].plot_data).toEqual([[0, 0], [10, .001], [20, .002]])
  expect(tables[2].slices[0].plot_data).toEqual([[0, 0], [10, .002], [20, .004]])
  await expect(result.locator('canvas')).toHaveCount(3)
  expect(JSON.parse((await downloaded(page, 'Download selected core evidence JSON')).toString('utf8'))).toEqual(data)
  const xml = await downloaded(page, 'Download selected core PLECS XML')
  const parsedResponse = await request.post('http://localhost:8002/api/plecs/preview', { data: { xml_base64: xml.toString('base64'), source_name: 'selected-core.xml' } })
  expect(parsedResponse.status()).toBe(200)
  const parsed = await parsedResponse.json()
  expect(parsed.packages[0].blocks[0].tables.map(table => table.slices)).toEqual(tables.map(table => table.slices))
  expect(parsed.packages[0].fixed_conditions.selection).toEqual(data.report.selection)
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
  for (const [index, expected] of [4, .002, .004].entries()) {
    await expect.poll(() => result.locator('canvas').nth(index).evaluate(canvas => {
      const draw = canvas.recordedDraw
      if (!draw?.points.length) return null
      const left = Math.min(...draw.points.map(point => point.x))
      const ticks = draw.labels.filter(label => label.x < left - 2).sort((a, b) => a.label - b.label)
      if (ticks.length < 2) return null
      const low = ticks[0], high = ticks.at(-1)
      const endpoint = draw.points.reduce((left, right) => right.x > left.x ? right : left)
      return low.label + (endpoint.y - low.y) * (high.label - low.label) / (high.y - low.y)
    })).toBeCloseTo(expected, 7)
  }
  await result.screenshot({ path: test.info().outputPath('plecs-generic-core-mobile.png') })
  await form.getByTestId('plecs-omit-thermal').fill('Changed explicit export scope')
  await expect(result).toHaveCount(0)
  expect(errors).toEqual([])
})
