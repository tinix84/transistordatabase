import { readFile } from 'node:fs/promises'
import { test, expect } from '@playwright/test'

test('a late workbook response cannot relabel the displayed or downloaded normalization proof', async ({ page }) => {
  const source = 'SemiconductorsData/IGBT'
  const metric = 'thermal_vi_per_price'
  const exportButton = page.getByRole('button', { name: 'Export JSON', exact: true })
  await page.goto('http://localhost:8002/index.html?view=catalog-fom')
  await expect(exportButton).toBeEnabled()
  await page.getByLabel('Catalog sources', { exact: true }).selectOption(source)
  await page.getByLabel('Part or manufacturer', { exact: true }).fill('RJH65D27BDPQ-A0')
  await page.getByLabel('X metric', { exact: true }).selectOption('thermal_current_a')
  await page.getByLabel('Y metric', { exact: true }).selectOption(metric)
  await page.getByLabel('Axis values', { exact: true }).selectOption('fixed')
  await page.getByLabel('Acknowledge thermal screening assumptions', { exact: true }).check()
  await expect(exportButton).toBeEnabled()

  let firstResponseReady, releaseFirst, firstResponseSent
  const waiting = new Promise(resolve => { firstResponseReady = resolve })
  const gate = new Promise(resolve => { releaseFirst = resolve })
  const sent = new Promise(resolve => { firstResponseSent = resolve })
  await page.route('**/api/catalog/screen', async route => {
    const request = route.request().postDataJSON()
    if (request.workbook_derivation?.source_declarations?.[0]?.currency !== 'CHF') {
      await route.continue()
      return
    }
    const response = await route.fetch()
    firstResponseReady(await response.json())
    await gate
    await route.fulfill({ response })
    firstResponseSent()
  })
  const currency = page.getByLabel(`Declared currency ${source}`, { exact: true })
  await currency.fill('CHF')
  const first = await waiting
  try {
    expect(first.candidates[0].workbook_evidence.currency).toBe('CHF')
    await expect(exportButton).toBeDisabled()
    await expect(page.locator('.scatter-card')).toHaveCount(0)
    await currency.fill('EUR')
    await expect(exportButton).toBeEnabled()
  } finally {
    releaseFirst()
  }
  await sent
  // Observe a full paint after the delayed response reached the browser.
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))))
  const download = page.waitForEvent('download')
  await exportButton.click()
  const exported = JSON.parse(await readFile(await (await download).path(), 'utf8'))
  const row = exported.result.candidates[0]
  expect(exported.request.workbook_derivation.source_declarations[0].currency).toBe('EUR')
  expect(row.workbook_evidence.currency).toBe('EUR')
  expect(row.workbook_evidence.original_currency).toBe('unspecified')
  expect(row.metrics[metric]).toBeCloseTo(3989.41001622, 5)
  expect(exported.result.normalization.mode).toBe('fixed')
  expect(exported.result.normalization.population_hash).not.toBe(first.normalization.population_hash)
  expect(exported.result.normalization.metrics[metric].population_hash).toBe(exported.result.normalization.population_hash)
  await expect(page.locator('.scatter-card').first().locator('.scatter-heading')).toContainText('1 points')
})
