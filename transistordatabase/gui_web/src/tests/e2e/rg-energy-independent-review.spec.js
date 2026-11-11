import { expect, test } from '@playwright/test'
import { readFile } from 'node:fs/promises'

test('a fresh server result replaces stale inventory evidence at a tolerance boundary', async ({ page }) => {
  const base = process.env.RG_ENERGY_FIXTURE_URL || 'http://localhost:8003'
  const panel = page.getByTestId('rg-energy')
  const field = name => panel.getByTestId(`rg-${name}`)
  const oldHash = 'f'.repeat(64)
  // Change only the inventory. The calculation uses the fixture server and its current source.
  await page.route('**/api/transistors/*/rg-energy', async route => {
    if (route.request().method() !== 'GET') return route.continue()
    const response = await route.fetch()
    const inventory = await response.json()
    const snapshot = inventory.curves.find(row => row.branch === 'e_on' && row.representation === 'graph_r_e')
    snapshot.conditions.v_supply += 5e-10
    snapshot.snapshot_sha256 = oldHash
    snapshot.t_j = 105
    snapshot.preview.energy_j = snapshot.preview.energy_j.map(value => value * 2)
    await route.fulfill({ response, json: inventory })
  })
  await page.goto(`${base}/index.html?view=analytical&model=parameters#rg-energy`)
  await field('device').selectOption('Browser Rg energy')
  await field('context').selectOption({ index: 1 })
  const table = panel.getByRole('table')
  await expect(table).toContainText(oldHash)
  await field('current').fill('10')
  await field('temperature').fill('75')
  await field('query').fill('1.5')
  const waiting = page.waitForResponse(response =>
    response.url().endsWith('/rg-energy') && response.request().method() === 'POST')
  await field('calculate').click()
  const response = await waiting
  const body = await response.text()
  expect(response.status(), body).toBe(200)
  const proof = JSON.parse(body)
  const source = proof.sources.find(row => row.representation === 'graph_r_e')
  expect(source.conditions.v_supply).toBe(400)
  expect(proof.query.conditions.v_supply).toBe(400.0000000005)
  expect(proof.value).toBeCloseTo(.007, 12)
  await expect(table).toContainText(source.snapshot_sha256)
  await expect(table).not.toContainText(oldHash)
  const downloading = page.waitForEvent('download')
  await field('json').click()
  const file = await downloading
  expect(await readFile(await file.path(), 'utf8')).toBe(body)
})
