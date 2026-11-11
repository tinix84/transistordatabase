import { test, expect } from '@playwright/test'
import fs from 'node:fs/promises'
import { createHash } from 'node:crypto'

const origin = 'http://localhost:8002'
const source = '/home/tinix/claude_wsl/transistordatabase/tests/test_data/sample_plecs_switch.xml'

async function inspect(page, filename = source) {
  await page.goto(`${origin}/index.html?view=plecs-data`)
  await page.getByTestId('plecs-file').setInputFiles(filename)
  await expect(page.getByTestId('plecs-xml')).not.toHaveValue('')
  const pending = page.waitForResponse(r => r.url().endsWith('/api/plecs/preview') && r.request().method() === 'POST')
  await page.getByRole('button', { name: 'Inspect PLECS document', exact: true }).click()
  const reply = await pending; expect(reply.status()).toBe(200)
  return reply.json()
}

async function observeCanvas(page) {
  await page.addInitScript(() => {
    const clear = CanvasRenderingContext2D.prototype.clearRect
    const arc = CanvasRenderingContext2D.prototype.arc
    const fillText = CanvasRenderingContext2D.prototype.fillText
    CanvasRenderingContext2D.prototype.clearRect = function (...args) {
      this.canvas.reviewDraw = { points: [], labels: [] }; return clear.apply(this, args)
    }
    CanvasRenderingContext2D.prototype.arc = function (x, y, radius, ...args) {
      const t = this.getTransform()
      if (radius === 3 && this.canvas.reviewDraw) this.canvas.reviewDraw.points.push({ x: t.a * x + t.c * y + t.e, y: t.b * x + t.d * y + t.f })
      return arc.call(this, x, y, radius, ...args)
    }
    CanvasRenderingContext2D.prototype.fillText = function (label, x, y, ...args) {
      const t = this.getTransform(), value = Number(String(label).replaceAll(',', ''))
      if (String(label).trim() && Number.isFinite(value) && this.canvas.reviewDraw) this.canvas.reviewDraw.labels.push({ value, x: t.a * x + t.c * y + t.e, y: t.b * x + t.d * y + t.f })
      return fillText.call(this, label, x, y, ...args)
    }
  })
}

async function points(card, expected = null) {
  const source = expected ?? await card.locator('tbody tr').evaluateAll(rows => rows.map(row => [...row.querySelectorAll('td')].map(cell => Number(cell.textContent))))
  return card.locator('canvas').evaluate((canvas, source) => {
    const draw = canvas.reviewDraw, minX = Math.min(...draw.points.map(p => p.x)), maxY = Math.max(...draw.points.map(p => p.y))
    const xLabels = draw.labels.filter(p => p.y > maxY + 2 && p.x >= minX - 1).sort((a, b) => a.value - b.value)
    const yLabels = draw.labels.filter(p => p.x < minX - 1).sort((a, b) => a.value - b.value)
    if (xLabels.length < 2 || yLabels.length < 2) throw new Error(`Missing independently painted numeric ticks ${JSON.stringify(draw)}`)
    const [x0, x1] = [xLabels[0], xLabels.at(-1)], [y0, y1] = [yLabels[0], yLabels.at(-1)]
    return draw.points.slice(0, source.length).map((p, i) => ({ source: { x: source[i][0], y: source[i][1] },
      x: x0.value + (p.x - x0.x) * (x1.value - x0.value) / (x1.x - x0.x),
      y: y0.value + (p.y - y0.y) * (y1.value - y0.value) / (y1.y - y0.y) }))
  }, source)
}

function exactPoints(values) {
  // Canvas transform matrices use browser float precision. This is less than
  // 0.0001 display pixel across these charts, far below the animation defect.
  const xSpan = Math.max(...values.map(p => p.source.x)) - Math.min(...values.map(p => p.source.x))
  const ySpan = Math.max(...values.map(p => p.source.y)) - Math.min(...values.map(p => p.source.y))
  for (const point of values) {
    expect(Math.abs(point.x - point.source.x)).toBeLessThan(xSpan * 1e-7)
    expect(Math.abs(point.y - point.source.y)).toBeLessThan(ySpan * 1e-7)
  }
}

test('independent source SI points remain exact through unrelated export and mapping updates', async ({ page }) => {
  const errors = []; page.on('pageerror', error => errors.push(error.message))
  await observeCanvas(page)
  const data = await inspect(page)
  expect(data.source.sha256).toBe(createHash('sha256').update(await fs.readFile(source)).digest('hex'))
  const table = data.packages[0].blocks[0].tables.find(t => t.kind === 'turn-on')
  const index = table.slices.findIndex(s => s.temperature_c === 150 && s.supply_voltage_v === 800)
  const card = page.getByTestId('plecs-table').filter({ has: page.getByRole('heading', { name: /^turn-on ·/ }) })
  await card.getByTestId('plecs-slice').selectOption(String(index))
  await page.waitForTimeout(1300)
  exactPoints(await points(card))
  expect((await points(card)).at(-1).source).toEqual({ x: 50, y: .0027 })
  let release, arrived
  const held = new Promise(resolve => { release = resolve }), started = new Promise(resolve => { arrived = resolve })
  await page.route('**/api/plecs/export', async route => { const response = await route.fetch(); arrived(); await held; await route.fulfill({ response }) })
  const download = page.waitForEvent('download')
  await page.getByRole('button', { name: 'Download preserved source XML', exact: true }).click()
  await started; await page.waitForTimeout(100)
  exactPoints(await points(card))
  release()
  expect(await fs.readFile(await (await download).path())).toEqual(await fs.readFile(source))
  await page.waitForTimeout(100); exactPoints(await points(card))
  const mapping = page.getByTestId('plecs-core-mapping')
  await mapping.locator('summary').click(); await mapping.getByTestId('plecs-map-package').selectOption('0')
  await mapping.getByTestId('plecs-declare-i_cont').fill('40')
  await page.waitForTimeout(100); exactPoints(await points(card))
  for (const [key, value] of Object.entries({ v_abs_max: 1200, i_abs_max: 80, i_cont: 40, t_j_max: 175, housing_area: .0001, cooling_area: .0002 })) await mapping.getByTestId(`plecs-declare-${key}`).fill(String(value))
  await mapping.getByTestId('plecs-target-0').selectOption('switch')
  for (const gate of await mapping.getByTestId('plecs-declare-gate').all()) await gate.fill('15')
  await mapping.getByTestId('plecs-recovery-convention').selectOption('preserve-signed')
  await mapping.getByTestId('plecs-thermal-owner').selectOption('switch')
  const finalize = page.waitForResponse(r => r.url().endsWith('/api/plecs/finalize'))
  await mapping.getByRole('button', { name: 'Validate explicit core mapping', exact: true }).click()
  const finalReply = await finalize; expect(finalReply.status()).toBe(200)
  const mapped = await finalReply.json()
  expect(mapped.core.electrical_ratings.v_abs_max).toBe(1200)
  expect(mapped.core.switch.e_on_data).toHaveLength(4)
  const nativeDownload = page.waitForEvent('download')
  await mapping.getByRole('button', { name: 'Download declared core JSON', exact: true }).click()
  const native = JSON.parse(await fs.readFile(await (await nativeDownload).path(), 'utf8'))
  expect(native).toEqual(mapped.core_json_document)
  expect(native.schema).toBe('transistordatabase-core')
  await page.waitForTimeout(100); exactPoints(await points(card))
  await card.screenshot({ path: test.info().outputPath('independent-plecs-si-after-export.png') })
  expect(errors).toEqual([])
})

test('independent390px actual ENCO source preserves unsupported models and rejects stale success', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  const filename = '/mnt/d/OneDrive/claude/bhy/enco/enco/packages/enco-components/tests/mosfet/_mock/G3R40MT12K.xml'
  const doc = await inspect(page, filename)
  expect(doc.packages[0].blocks).toHaveLength(1)
  expect(doc.packages[0].blocks[0].tables.filter(t => t.kind === 'conduction')).toHaveLength(2)
  expect(doc.can_export).toBe(false)
  await expect(page.getByTestId('plecs-results')).toContainText('Cauer chain retained as source')
  await expect(page.getByTestId('plecs-results')).toContainText('StateSpaceModel')
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390)
  await page.getByTestId('plecs-results').locator('.diagnostics').screenshot({ path: test.info().outputPath('independent-enco-unsupported-mobile.png') })
  let release, arrived
  const held = new Promise(resolve => { release = resolve }), started = new Promise(resolve => { arrived = resolve })
  await page.route('**/api/plecs/preview', async route => { const response = await route.fetch(); arrived(); await held; await route.fulfill({ response }) })
  await page.getByRole('button', { name: 'Inspect PLECS document', exact: true }).click(); await started
  await page.getByTestId('plecs-source').fill('Changed identity'); release()
  await expect(page.getByRole('button', { name: 'Inspect PLECS document', exact: true })).toBeEnabled()
  await expect(page.getByTestId('plecs-results')).toHaveCount(0)
})

test('independent390px real comparison HTTP preserves305W partial rows and readable cells', async ({ page }) => {
  const errors = []; page.on('pageerror', error => errors.push(error.message))
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('http://localhost:8031/index.html?view=compare-advanced')
  await page.getByTestId('comparison-device-1').selectOption('bounded-a')
  await page.getByTestId('comparison-device-3').selectOption('bounded-b')
  await page.getByTestId('comparison-estimate').check()
  for (const [key, value] of Object.entries({ t_j: 25, i_channel: 10, f_sw: 100000, duty: .25, output_power_w: 1000 })) await page.getByTestId(`comparison-${key}`).fill(String(value))
  for (const group of await page.locator('fieldset.config-panel').all()) {
    await group.getByLabel('Supply voltage (V)', { exact: true }).fill('400')
    await group.getByLabel('Gate on voltage (V)', { exact: true }).fill('15')
    await group.locator('select').selectOption({ label: 'Known' })
    await group.getByLabel('Gate off voltage (V)', { exact: true }).fill('-3')
    await group.getByLabel('Gate resistance on (Ω)', { exact: true }).fill('4')
    await group.getByLabel('Gate resistance off (Ω)', { exact: true }).fill('7')
  }
  await page.getByTestId('comparison-supply-3').fill('401')
  const pending = page.waitForResponse(r => r.url().endsWith('/api/comparison/advanced') && r.request().method() === 'POST')
  await page.getByTestId('run-comparison').click()
  const response = await pending; expect(response.status()).toBe(200)
  const data = await response.json()
  expect(data.loss_breakdown.rows.map(r => r.total)).toEqual([305, null])
  expect(data.efficiency.rows[1].efficiency_percent).toBeNull()
  expect(data.switching_losses_eoff.curves.map(r => r.conditions.r_g)).toEqual([7, 7])
  await page.getByRole('button', { name: 'Loss breakdown', exact: true }).click()
  const result = page.getByTestId('comparison-results')
  await expect(result).toContainText('305.0000'); await expect(result).toContainText('Unavailable')
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390)
  expect(await result.locator('tbody td').first().evaluate(node => node.getBoundingClientRect().width)).toBeGreaterThanOrEqual(110)
  const download = page.waitForEvent('download'); await page.getByRole('button', { name: 'Download evidence JSON' }).click()
  expect(JSON.parse(await fs.readFile(await (await download).path(), 'utf8'))).toEqual(data)
  await result.screenshot({ path: test.info().outputPath('independent-comparison-partial-mobile.png') })
  await page.getByTestId('comparison-f_sw').fill('200000'); await expect(result).toHaveCount(0)
  expect(errors).toEqual([])
})

test('independent generic core exact selectors export real HTTP records without PLECS source provenance', async ({ page }) => {
  const errors = []; page.on('pageerror', error => errors.push(error.message))
  await observeCanvas(page)
  await page.goto('http://localhost:8031/index.html?view=plecs-data')
  const form = page.getByTestId('plecs-core-export')
  const contextsReady = page.waitForResponse(r => r.url().endsWith('/api/plecs/core/bounded-a/contexts'))
  await form.getByTestId('plecs-existing-device').selectOption('bounded-a')
  const contextsResponse = await contextsReady; expect(contextsResponse.status()).toBe(200)
  const contexts = await contextsResponse.json()
  await form.getByTestId('plecs-existing-component').selectOption('switch')
  for (const kind of ['conduction', 'turn_on', 'turn_off']) {
    expect(contexts.components.switch[kind]).toHaveLength(1)
    await form.getByTestId(`plecs-include-${kind}`).check()
    await form.getByTestId(`plecs-context-${kind}`).selectOption('0')
  }
  await form.getByTestId('plecs-omit-thermal').fill('Fixture has no measured Foster network')
  await form.getByTestId('plecs-core-convention').selectOption('preserve-signed')
  const pending = page.waitForResponse(r => r.url().endsWith('/api/plecs/core/bounded-a/preview'))
  await form.getByRole('button', { name: 'Preview selected core export', exact: true }).click()
  const response = await pending; expect(response.status()).toBe(200)
  const result = await response.json()
  expect(result.report.evidence.map(row => row.conditions)).toEqual([
    { gate_voltage_v: 15 }, { gate_voltage_v: 15, gate_resistance_ohm: 4 }, { gate_voltage_v: 15, gate_resistance_ohm: 7 },
  ])
  expect(result.report.omitted_mechanisms).toEqual(['thermal'])
  const curves = result.document.packages[0].blocks[0].tables
  expect(curves.find(row => row.kind === 'turn-on').slices[0].plot_data).toEqual([[0, 0], [10, .001], [20, .002]])
  const view = form.getByTestId('plecs-core-export-result')
  await expect(view.locator('canvas')).toHaveCount(3)
  await page.waitForTimeout(1200)
  const charts = await view.locator('.curve-plotter').all()
  for (let i = 0; i < charts.length; i++) exactPoints(await points(charts[i], curves[i].slices[0].plot_data))
  const jsonDownload = page.waitForEvent('download')
  await form.getByRole('button', { name: 'Download selected core evidence JSON', exact: true }).click()
  expect(JSON.parse(await fs.readFile(await (await jsonDownload).path(), 'utf8'))).toEqual(result)
  const xmlDownload = page.waitForEvent('download')
  await form.getByRole('button', { name: 'Download selected core PLECS XML', exact: true }).click()
  const xml = await fs.readFile(await (await xmlDownload).path())
  const reparsed = await page.request.post('http://localhost:8031/api/plecs/preview', { data: { xml_base64: xml.toString('base64') } })
  expect(reparsed.status()).toBe(200)
  expect((await reparsed.json()).packages[0].fixed_conditions.selection).toEqual(result.report.selection)
  await view.screenshot({ path: test.info().outputPath('independent-generic-core-export-desktop.png') })
  await form.getByTestId('plecs-omit-thermal').fill('Different scope declaration')
  await expect(view).toHaveCount(0)
  expect(errors).toEqual([])
})
