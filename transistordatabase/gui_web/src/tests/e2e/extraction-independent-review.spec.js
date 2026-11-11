import { expect, test } from '@playwright/test'
import { readFile } from 'node:fs/promises'

async function openWorkbench(page, operation) {
  await page.goto('/index.html')
  await page.getByRole('button', { name: /More Tools/ }).click()
  await page.getByRole('button', { name: 'Analytical Models', exact: true }).click()
  await page.getByRole('button', { name: 'Device Parameter Operations' }).click()
  const workbench = page.locator('.parameter-workbench')
  await workbench.getByLabel('Parameter calculation').selectOption(operation)
  return workbench
}

async function calculate(page, workbench, expectedStatus = 200) {
  const response = page.waitForResponse(r =>
    new URL(r.url()).pathname === '/api/parameters/calculate' && r.request().method() === 'POST')
  await workbench.getByRole('button', { name: 'Calculate device parameters' }).click()
  const verified = await response
  expect(verified.status()).toBe(expectedStatus)
  return verified.json()
}

async function download(page, button, name) {
  const pending = page.waitForEvent('download')
  await button.click()
  const file = await pending
  await file.saveAs(test.info().outputPath(name))
  return readFile(await file.path(), 'utf8')
}

test('mobile transfer plot, table and CSV use server rows and source edits cancel a pending application', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.addInitScript(() => {
    const labels = new WeakMap()
    const original = CanvasRenderingContext2D.prototype.fillText
    CanvasRenderingContext2D.prototype.fillText = function (content, x, y, ...rest) {
      const text = String(content)
      if (/^-?\d+(?:\.\d+)?e[-+]\d+$/i.test(text)) {
        const bounds = this.measureText(text)
        const transform = this.getTransform()
        const left = new DOMPoint(x - bounds.actualBoundingBoxLeft, y).matrixTransform(transform).x
        const right = new DOMPoint(x + bounds.actualBoundingBoxRight, y).matrixTransform(transform).x
        if (!labels.has(this.canvas)) labels.set(this.canvas, new Map())
        labels.get(this.canvas).set(text, { text, left, right, width: this.canvas.width })
      }
      return original.call(this, content, x, y, ...rest)
    }
    window.readDrawnScientificLabels = canvas => Array.from(labels.get(canvas)?.values() || [])
  })
  const workbench = await openWorkbench(page, 'transfer-fit')
  const source = '=laboratory-fixture'
  await workbench.locator('[name=source]').fill(source)
  await workbench.getByLabel('Curve 1 Vds samples (V)').fill('9, 11')
  await workbench.getByLabel('Curve 1 current samples (A)').fill('1.1, 3.1')
  const envelope = await calculate(page, workbench)
  expect(envelope.inputs.curves[0].voltage_v).toEqual([9, 11])
  expect(envelope.inputs.curves[0].current_a).toEqual([1.1, 3.1])
  expect(envelope.result.current_a[0]).toBeCloseTo(2.1, 12)
  const result = workbench.getByLabel('Device parameter results')
  const table = result.getByLabel('Exact transfer fit samples')
  const rows = table.locator('tbody tr')
  await expect(rows).toHaveCount(5)
  for (let i = 0; i < 5; i++) {
    const cells = (await rows.nth(i).locator('td').allTextContents()).map(Number)
    expect(cells.slice(0, 2)).toEqual([4+i, envelope.result.current_a[i]])
    expect(cells[2]).toBeCloseTo(envelope.result.fitted_current_a[i], 7)
    expect(cells[4]).toBeCloseTo(envelope.result.secant_gm_s[i], 6)
  }
  const csv = await download(page, result.getByRole('button', { name: /CSV/i }), 'mobile-transfer.csv')
  const lines = csv.trimEnd().split(/\r?\n/)
  expect(lines.shift()).toBe('gate_voltage_v,current_a,fitted_current_a,residual_a,secant_gm_s,source,v_ds_v,t_j_c,v_th_v')
  expect(lines).toHaveLength(5)
  for (let i = 0; i < lines.length; i++) {
    const cells = lines[i].split(',')
    const expected = ['gate_voltage_v', 'current_a', 'fitted_current_a', 'residual_a', 'secant_gm_s']
    expect(cells.slice(0, 5).map(Number)).toEqual(expected.map(key => envelope.result[key][i]))
    expect(cells[5]).toBe("'" + source)
    expect(cells.slice(6).map(Number)).toEqual([10, 25, 3])
  }
  const json = await download(page, result.getByRole('button', { name: 'Download reproducible result' }), 'mobile-transfer.json')
  expect(JSON.parse(json)).toEqual(envelope)
  await table.scrollIntoViewIfNeeded()
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(391)
  await page.screenshot({ path: test.info().outputPath('transfer-table-390.png') })
  const scroller = result.getByRole('region', { name: 'Scrollable transfer fit samples' })
  const rightEdge = await scroller.evaluate(element => {
    element.scrollLeft = element.scrollWidth
    return element.getBoundingClientRect().right
  })
  const lastColumn = await table.getByRole('columnheader', { name: 'Secant gm (S)' }).boundingBox()
  expect(lastColumn.x + lastColumn.width).toBeLessThanOrEqual(rightEdge + 1)
  await page.screenshot({ path: test.info().outputPath('transfer-table-residuals-390.png') })
  const plots = result.locator('.curve-plotter')
  await expect(plots).toHaveCount(2)
  await plots.first().screenshot({ path: test.info().outputPath('transfer-fit-chart-390.png') })
  await plots.last().screenshot({ path: test.info().outputPath('transfer-residual-chart-390.png') })
  const ticks = await plots.last().locator('canvas').evaluate(canvas => window.readDrawnScientificLabels(canvas))
  expect(ticks.some(({ text }) => Number(text) < 0)).toBeTruthy()
  expect(ticks.some(({ text }) => Number(text) > 0)).toBeTruthy()
  for (const tick of ticks) {
    expect(tick.text).toMatch(/^-?\d(?:\.\d{1,2})?e[-+]\d+$/i)
    expect(tick.left).toBeGreaterThanOrEqual(0)
    expect(tick.right).toBeLessThanOrEqual(tick.width)
  }
  await result.getByRole('button', { name: 'Review extracted parameters in Biela' }).click()
  const biela = page.locator('.biela-calculator')
  await expect(biela.getByLabel('Review extraction application')).toBeVisible()
  await page.getByRole('button', { name: 'Device Parameter Operations' }).click()
  await workbench.getByLabel('Curve 1 current samples (A)').fill('2.1, 3.1')
  await expect(result).toHaveCount(0)
  await page.getByRole('button', { name: 'Biela Switching Loss Model' }).click()
  await expect(biela.getByLabel('Review extraction application')).toHaveCount(0)
  await expect(biela.locator('[name=k_1]')).toHaveValue('10')
})

test('recovery form converts GA/s to SI, retains omitted Trr and rejects a blank test condition', async ({ page }) => {
  const workbench = await openWorkbench(page, 'reverse-recovery-extraction')
  await workbench.getByLabel(/slope.*GA\/s/i).fill('2.4')
  await workbench.locator('[name=source]').fill('RR laboratory fixture')
  const envelope = await calculate(page, workbench)
  expect(envelope.inputs.di_dt_a_s).toBe(2.4e9)
  expect(envelope.inputs.t_rr_s).toBeNull()
  expect(envelope.result.q_rr_star * 1e9).toBeCloseTo(88, 10)
  expect(envelope.result.tau_rr * 1e9).toBeCloseTo(6.716666666667, 9)
  const result = workbench.getByLabel('Device parameter results')
  const csv = await download(page, result.getByRole('button', { name: /CSV/i }), 'recovery-evidence.csv')
  const [head, values] = csv.trimEnd().split(/\r?\n/).map(line => line.split(','))
  const row = Object.fromEntries(head.map((key, i) => [key, values[i]]))
  expect(row.t_rr_s).toBe('')
  expect(row.source).toBe('RR laboratory fixture')
  for (const key of ['q_rr_c', 'q_oss_c', 'i_rr_a', 'di_dt_a_s', 'i_forward_test_a']) {
    expect(Number(row[key])).toBe(envelope.inputs[key])
  }
  for (const key of ['tau_rr', 'tau_c', 't_m']) expect(Number(row[key + '_s'])).toBe(envelope.result[key])
  const temperature = workbench.getByRole('spinbutton', { name: /^Source conditions: Junction temperature/ })
  await temperature.fill('')
  await expect(result).toHaveCount(0)
  const rejected = await calculate(page, workbench, 422)
  expect(rejected.detail.some(issue => issue.loc.includes('t_j_c'))).toBeTruthy()
  await expect(temperature).toHaveValue('')
  await expect(workbench.getByRole('alert')).toBeVisible()
})

test('invalid advanced transfer input leaves the editor usable', async ({ page }) => {
  const errors = []
  page.on('pageerror', err => errors.push(err.message))
  const workbench = await openWorkbench(page, 'transfer-fit')
  await workbench.locator('.data-editor summary').click()
  const editor = workbench.getByLabel('Parameter input data')
  const valid = await editor.inputValue()
  const invalid = JSON.parse(valid)
  invalid.curves[0] = { v_g_v: 4, t_j_c: 25 }
  await editor.fill(JSON.stringify(invalid))
  await calculate(page, workbench, 422)
  await expect(workbench.getByRole('alert')).toBeVisible()
  await editor.fill(valid)
  await calculate(page, workbench)
  await expect(workbench.getByLabel('Exact transfer fit samples')).toBeVisible()
  expect(errors).toEqual([])
})

test('Biela estimates a 400 V operating point from two independent source records', async ({ page }) => {
  const workbench = await openWorkbench(page, 'transfer-fit')
  const biela = page.locator('.biela-calculator')
  const workbenchTab = page.getByRole('button', { name: 'Device Parameter Operations' })
  const bielaTab = page.getByRole('button', { name: 'Biela Switching Loss Model' })
  const source = workbench.locator('[name=source]')
  await source.fill('Transfer measurements at 10 V')
  const transfer = await calculate(page, workbench)
  const review = workbench.getByRole('button', { name: 'Review extracted parameters in Biela' })
  const apply = biela.getByRole('button', { name: 'Apply these extracted parameters' })
  await review.click()
  await apply.click()
  await workbenchTab.click()
  await workbench.getByLabel('Parameter calculation').selectOption('reverse-recovery-extraction')
  await source.fill('Recovery measurements at 400 V')
  await workbench.getByLabel(/Source conditions: Switching supply voltage/).fill('400')
  await workbench.getByLabel(/Source conditions: Qoss reference voltage/).fill('400')
  const recovery = await calculate(page, workbench)
  await review.click()
  await apply.click()
  for (const [key, value] of Object.entries({ v_dc: '100', q_oss_v_ref: '100', c_ds: '1e-9', q_oss: '2e-7' })) {
    expect(Number(await biela.locator('[name=' + key + ']').inputValue())).toBe(Number(value))
  }
  for (const [key, value] of Object.entries({ v_dc: '400', q_oss_v_ref: '400', q_oss: '12e-9', c_ds: '15e-12', c_gd: '15e-12' })) {
    await biela.locator('[name=' + key + ']').fill(value)
  }
  await biela.getByRole('checkbox', { name: 'Use the corrected analytical half-bridge estimate' }).check()
  const estimateResponse = page.waitForResponse(r => new URL(r.url()).pathname === '/api/analytical/biela')
  await biela.getByRole('button', { name: 'Calculate half-bridge estimate' }).click()
  const verified = await estimateResponse
  expect(verified.status()).toBe(200)
  const estimate = await verified.json()
  expect(estimate.operating_point).toEqual({ v_dc: 400, t_j: 25, i_load: 10 })
  expect(estimate.extraction_evidence).toEqual([{ status: 'applied', ...transfer }, { status: 'applied', ...recovery }])
  const exported = await download(page, biela.getByRole('button', { name: 'Download reproducible estimate JSON' }), '400v-combined-estimate.json')
  expect(JSON.parse(exported)).toEqual(estimate)
  await workbenchTab.click()
  await source.fill('Another recovery source under review')
  await bielaTab.click()
  await expect(biela.getByLabel('Biela estimate results')).toBeVisible()
  await biela.locator('[name=k_1]').fill('2.01')
  await expect(biela.getByLabel('Biela estimate results')).toHaveCount(0)
  const evidence = biela.getByLabel('Applied extraction evidence')
  await expect(evidence).toContainText('Edited manual parameters')
  await expect(evidence).toContainText('Applied extracted parameters')
})
