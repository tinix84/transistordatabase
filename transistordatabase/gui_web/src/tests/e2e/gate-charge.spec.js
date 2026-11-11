import { test, expect } from '@playwright/test'

async function openGate(page) {
  await page.goto('/')
  await page.getByRole('button', { name: 'More', exact: false }).click()
  await page.getByRole('button', { name: 'Analytical Models', exact: true }).click()
  await page.getByRole('button', { name: 'Gate Charge Timing Model', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Gate charge timing estimate', exact: true })).toBeVisible()
  await page.getByLabel('Use the constant-capacitance RC / Miller estimate').check()
}

test('real gate estimate shows independently checked intervals, SI traces and explicit drain scope', async ({ page }) => {
  const errors = []
  page.on('pageerror', error => errors.push(error.message))
  await openGate(page)
  await page.getByRole('button', { name: 'Calculate switching intervals', exact: true }).click()
  const result = page.getByLabel('Gate timing results')
  await expect(result).toContainText('13.865581 ns')
  await expect(result).toContainText('40.609092 ns')
  await expect(result).toContainText('2.677723')
  await expect(result).toContainText('NORMALIZED')
  await expect(result.locator('canvas')).toHaveCount(6)
  await page.screenshot({ path: test.info().outputPath('gate-interval-estimates.png'), fullPage: true })
  await page.getByLabel('Voff: driver off voltage (V)', { exact: true }).fill('-5')
  await expect(result).toHaveCount(0)
  await page.getByRole('button', { name: 'Calculate switching intervals', exact: true }).click()
  await expect(page.getByLabel('Gate timing results')).toContainText('17.317766 ns')
  expect(errors).toEqual([])
})

test('real API domains show actionable errors and zero-charge stages are explicit', async ({ page }) => {
  await openGate(page)
  await page.getByLabel('Von: driver on voltage (V)', { exact: true }).fill('5')
  await page.getByRole('button', { name: 'Calculate switching intervals', exact: true }).click()
  await expect(page.getByRole('alert')).toContainText('Vplateau < Von')
  await expect(page.getByLabel('Gate timing results')).toHaveCount(0)
  await page.getByLabel('Von: driver on voltage (V)', { exact: true }).fill('15')
  await expect(page.getByRole('alert')).toHaveCount(0)
  await page.getByLabel('Qg: total at Von (C)', { exact: true }).fill('2.5e-8')
  await page.getByRole('button', { name: 'Calculate switching intervals', exact: true }).click()
  await expect(page.getByLabel('Gate timing results')).toContainText('(instantaneous)')
  await expect(page.getByLabel('Gate timing results').locator('canvas')).toHaveCount(6)
})
