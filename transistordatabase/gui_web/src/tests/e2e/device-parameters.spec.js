import { test, expect } from '@playwright/test'

async function openParameters(page) {
  await page.goto('/')
  await page.getByRole('button', { name: 'More', exact: false }).click()
  await page.getByRole('button', { name: 'Analytical Models', exact: true }).click()
  await page.getByRole('button', { name: 'Device Parameter Operations', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Device parameter workbench', exact: true })).toBeVisible()
}

test('real parameter fit, bounded capacitance plots and reproducible download', async ({ page }) => {
  const errors = []
  page.on('pageerror', error => errors.push(error.message))
  await openParameters(page)
  const result = page.getByLabel('Device parameter results')
  await page.getByLabel('Measured current samples (A)', { exact: true }).fill('2, 4, 6')
  await page.getByLabel('Measured voltage samples (V)', { exact: true }).fill('1.26\n1.32\n1.38')
  await page.getByRole('button', { name: 'Calculate device parameters', exact: true }).click()
  await expect(result).toContainText('Resistance (Ω)')
  await expect(result).toContainText('0.03')
  await expect(result.locator('canvas')).toHaveCount(1)
  await page.getByLabel('Parameter calculation').selectOption('capacitance-integration')
  await expect(result).toHaveCount(0)
  await page.locator('[name=end_v]').fill('150')
  await page.getByRole('button', { name: 'Calculate device parameters', exact: true }).click()
  await expect(result).toContainText('9.625e-8')
  await expect(result.locator('canvas')).toHaveCount(2)
  const downloadEvent = page.waitForEvent('download')
  await page.getByRole('button', { name: 'Download reproducible result', exact: true }).click()
  const download = await downloadEvent
  expect(download.suggestedFilename()).toBe('device-parameters-capacitance-integration.json')
  await download.saveAs(test.info().outputPath(download.suggestedFilename()))
  await page.screenshot({ path: test.info().outputPath('device-parameter-capacitance.png'), fullPage: true })
  await page.locator('[name=end_v]').fill('201')
  await expect(result).toHaveCount(0)
  await page.getByRole('button', { name: 'Calculate device parameters', exact: true }).click()
  await expect(page.getByRole('alert')).toContainText('supports 0 to 200')
  expect(errors).toEqual([])
})

test('real signed slew requires selected estimate; known and unknown table conditions differ', async ({ page }) => {
  await openParameters(page)
  const select = page.getByLabel('Parameter calculation')
  const calculate = page.getByRole('button', { name: 'Calculate device parameters', exact: true })
  await select.selectOption('gate-slew-estimate')
  await expect(calculate).toBeDisabled()
  await page.getByRole('checkbox', { name: 'Use the signed gate current and slew estimate with its stated assumptions', exact: true }).check()
  await page.locator('[name=v_drive_v]').fill('-5')
  await calculate.click()
  await expect(page.getByLabel('Device parameter results')).toContainText('-11000000000')
  await expect(page.getByLabel('Device parameter results')).toContainText('22000000000')
  await select.selectOption('measured-table')
  const unknown = page.getByRole('checkbox', { name: 'Requested conditions: Gate resistance (Ω) unknown', exact: true })
  await unknown.uncheck()
  await calculate.click()
  await expect(page.getByRole('alert')).toContainText('do not match')
  await unknown.check()
  await calculate.click()
  await expect(page.getByLabel('Device parameter results')).toContainText('0.0015')
})

test('rapidly replacing charts and leaving the panel cancels pending chart animation', async ({ page }) => {
  const errors = []
  page.on('pageerror', error => errors.push(error.message))
  await openParameters(page)
  const select = page.getByLabel('Parameter calculation')
  for (let i = 0; i < 3; i++) {
    await select.selectOption('capacitance-integration')
    await page.getByRole('button', { name: 'Calculate device parameters', exact: true }).click()
    await expect(page.getByLabel('Device parameter results').locator('canvas')).toHaveCount(2)
    await select.selectOption('linear-fit')
  }
  await page.getByRole('button', { name: 'Calculate device parameters', exact: true }).click()
  await expect(page.getByLabel('Device parameter results').locator('canvas')).toHaveCount(1)
  await page.getByRole('button', { name: 'Gate Charge Timing Model', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Gate charge timing estimate', exact: true })).toBeVisible()
  await page.waitForTimeout(1100)
  expect(errors).toEqual([])
})
