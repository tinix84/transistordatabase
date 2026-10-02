import { readFile } from 'node:fs/promises'
import { test, expect } from '@playwright/test'

async function setParameter(page, name, value) {
  const field = page.locator('label.param-item').filter({ hasText: name }).locator('input')
  await field.fill(String(value))
}

async function openTopology(page, topology = 'Boost') {
  await page.goto('/')
  await page.getByRole('button', { name: 'Topology', exact: true }).click()
  await page.getByRole('button', { name: new RegExp(topology) }).click()
}

test.describe('topology migration browser workflow', () => {
  test('calculates a boost point through FastAPI and invalidates changed results', async ({ page }) => {
    await openTopology(page)
    await setParameter(page, 'Input voltage', 100)
    await setParameter(page, 'Output voltage magnitude', 200)
    await setParameter(page, 'Output power', 1000)
    await setParameter(page, 'Switching frequency', 20)
    await setParameter(page, 'Inductance', 1000)
    await setParameter(page, 'Output capacitance', 1000)

    await page.getByRole('button', { name: 'Calculate', exact: true }).click()
    const results = page.locator('.results-panel').first()
    await expect(results).toBeVisible()
    await expect(page.locator('.model-label')).toHaveText('Ideal CCM operating-point estimate')
    await expect(results).toContainText('10.000 A')
    await expect(results).toContainText('5.000 A')
    await expect(results).toContainText('0.5000')
    await expect(page.locator('.charts-section canvas')).toHaveCount(1)

    const downloadPromise = page.waitForEvent('download')
    await page.getByRole('button', { name: 'Download JSON' }).click()
    const download = await downloadPromise
    expect(download.suggestedFilename()).toMatch(/^topology-boost-\d{4}-\d{2}-\d{2}\.json$/)
    const downloadPath = await download.path()
    const exported = JSON.parse(await readFile(downloadPath, 'utf8'))
    expect(exported).toMatchObject({
      topology: 'boost',
      result: { model: { mode: 'estimate' } },
    })
    expect(exported.request).toBeDefined()

    await setParameter(page, 'Input voltage', 110)
    await expect(page.locator('.results-panel')).toHaveCount(0)
    await expect(page.getByRole('button', { name: 'Download JSON' })).toBeDisabled()
  })

  test('reports a DCM boundary error in the browser instead of inventing a point', async ({ page }) => {
    await openTopology(page)
    await setParameter(page, 'Input voltage', 100)
    await setParameter(page, 'Output voltage magnitude', 200)
    await setParameter(page, 'Output power', 1000)
    await setParameter(page, 'Switching frequency', 20)
    await setParameter(page, 'Inductance', 1000)
    await page.locator('label.param-item').filter({ hasText: 'Conduction regime' }).locator('select').selectOption('DCM')

    await page.getByRole('button', { name: 'Calculate', exact: true }).click()
    await expect(page.getByRole('alert')).toContainText('at or above the CCM boundary')
    await expect(page.locator('.results-panel')).toHaveCount(0)
  })
})
