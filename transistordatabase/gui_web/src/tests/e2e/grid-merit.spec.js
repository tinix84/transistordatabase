import { test, expect } from '@playwright/test'
import fs from 'node:fs/promises'

const conditions = { t_j_c: 25, v_supply_v: 400 }
const omitted = { omit: true, omission_reason: 'Independent fixture has no turn-off model' }
function configuration(id, drop, energy) {
  return { id, source: `Independent ${id} hand fixture; no device measurement identity`,
    conduction: { positive: { mode: 'linear-drop-estimate', selected: true, source: `Explicit ${drop} V constant-drop estimate`,
      offset_v: drop, resistance_ohm: 0, conditions: { t_j_c: 25 } } },
    switching: { turn_on: { positive: { mode: 'recorded', source: `Independent constant ${energy} J energy fixture`,
      current_a: [1, 3], energy_j: [energy, energy], recorded_conditions: conditions, requested_conditions: conditions } }, turn_off: omitted } }
}
const a = configuration('A', 10, 4), b = configuration('B', 8, 8)
const excluded = { ...configuration('excluded', 9, 4), switching: { turn_on: omitted, turn_off: omitted } }
function point(id, current, weight) { return { id, source: `Independent ${current} A constant-current fixture`, weight,
  waveform: { time_s: [0, 1], current_a: [current, current], gate_v: [0, 1], bus_voltage_v: 400 },
  events: { mode: 'gate', threshold_v: .5 } } }
const objectives = [{ name: 'conduction_power_w', unit: 'W', direction: 'minimize', weight: 1 },
  { name: 'turn_on_power_w', unit: 'W', direction: 'minimize', weight: 3 }]
const payload = { configurations: [a, b, excluded], operating_points: [point('low', 1, 1), point('high', 3, 3), point('zero', -2, 0)],
  metric: { name: 'conduction_power_w', unit: 'W', direction: 'minimize' }, objectives }
async function calculate(page, request) {
  await page.getByLabel('Grid request JSON', { exact: true }).fill(JSON.stringify(request))
  const waiting = page.waitForResponse(value => value.url().includes('/api/analysis/grid') && value.request().method() === 'POST')
  await page.getByRole('button', { name: 'Calculate analysis grid', exact: true }).click()
  const response = await waiting; expect(response.status()).toBe(200); return response.json()
}
async function downloaded(page, name) {
  const waiting = page.waitForEvent('download'); await page.getByRole('button', { name, exact: true }).click()
  return fs.readFile(await (await waiting).path(), 'utf8')
}
async function capture(page, viewport) {
  await page.getByTestId('merit-evidence').evaluate(node => node.scrollIntoView({ block: 'start' }))
  await page.evaluate(() => window.scrollBy(0, -110))
  await page.screenshot({ path: test.info().outputPath(`merit-definition-${viewport.width}.png`) })
  await page.getByTestId('merit-ranking').evaluate(node => node.scrollIntoView({ block: 'start' }))
  await page.evaluate(() => window.scrollBy(0, -110))
  await page.screenshot({ path: test.info().outputPath(`merit-ranking-${viewport.width}.png`) })
  await page.getByTestId('merit-configurations').evaluate(node => {
    node.scrollIntoView({ block: 'start' }); node.scrollLeft = node.querySelectorAll('th')[2].offsetLeft
  })
  await page.evaluate(() => window.scrollBy(0, -110))
  await page.screenshot({ path: test.info().outputPath(`merit-values-${viewport.width}.png`) })
}

for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }]) {
  test(`compiled weighted merit uses one complete population at ${viewport.width}px`, async ({ page }) => {
    await page.setViewportSize(viewport)
    const errors = []; page.on('pageerror', error => errors.push(error.message))
    await page.goto('/index.html?view=analysis-grid')
    await expect(page.getByLabel('Compare weighted merit penalties', { exact: true })).not.toBeChecked()
    await page.getByText('Advanced canonical request, including explicit event masks', { exact: true }).click()
    await page.getByLabel('Use grid JSON as input', { exact: true }).check()
    const response = await calculate(page, payload), merit = response.merit
    expect(merit.status).toBe('computed')
    expect(merit.population.configuration_ids).toEqual(['A', 'B'])
    expect(merit.population.zero_weight_operating_point_ids).toEqual(['zero'])
    expect(merit.ranking.map(row => [row.configuration_id, row.score, row.rank])).toEqual([['A', .25, 1], ['B', .75, 2]])
    expect(merit.rows.find(row => row.configuration_id === 'A').objectives.map(row => row.full_mean)).toEqual([25, 4])
    expect(merit.rows.find(row => row.configuration_id === 'B').objectives.map(row => row.full_mean)).toEqual([20, 8])
    expect(merit.rows.find(row => row.configuration_id === 'excluded').status).toBe('excluded')
    await expect(page.getByTestId('merit-evidence')).toContainText('Lower dimensionless penalty')
    await expect(page.getByTestId('merit-ranking')).toContainText('Rank 1 · A · penalty 0.250000')
    await expect(page.getByTestId('merit-configurations')).toContainText('25.0000 W')
    for (const row of merit.ranking) {
      const ratio = await page.getByTestId('merit-ranking').locator(`[data-score="${row.score}"]`).evaluate(node => node.getBoundingClientRect().width / node.parentElement.getBoundingClientRect().width)
      expect(ratio).toBeCloseTo(row.score, 4)
    }
    await page.getByLabel('Filter pair diagnostics').selectOption('unsupported')
    const raw = await downloaded(page, 'Download merit JSON')
    expect(raw).toBe(merit.rows_json); expect(JSON.parse(raw)).toEqual(merit.export_rows)
    expect(await downloaded(page, 'Download merit CSV')).toBe(merit.rows_csv)
    await capture(page, viewport)
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
    const reverse = await calculate(page, { ...payload, objectives: [{ ...objectives[1], direction: 'maximize', weight: 1 }] })
    expect(reverse.merit.ranking.map(row => [row.configuration_id, row.score])).toEqual([['B', 0], ['A', 1]])
    const tied = await calculate(page, { ...payload, configurations: [a, b, { ...a, id: 'D', source: 'Independent equal-means fixture D' }, excluded] })
    expect(tied.merit.ranking.map(row => [row.configuration_id, row.rank])).toEqual([['A', 1], ['D', 1], ['B', 3]])
    expect(tied.merit.objectives[1].minimum_configuration_ids).toEqual(['A', 'D'])
    const changedSource = await calculate(page, { ...payload, configurations: [{ ...a, source: 'Changed independent source identity' }, b, excluded] })
    expect(changedSource.merit.grid_input_sha256).not.toBe(merit.grid_input_sha256)
    expect(changedSource.merit.population_sha256).not.toBe(merit.population_sha256)
    expect(changedSource.merit.ranking.map(row => row.score)).toEqual([.25, .75])
    const unavailable = await calculate(page, { ...payload, configurations: [excluded] })
    expect(unavailable.merit.status).toBe('unavailable'); expect(unavailable.merit.ranking).toEqual([])
    await expect(page.getByTestId('merit-evidence')).toContainText('Scores and ranks are unavailable')
    for (const raw of [JSON.stringify(payload).replace('"weight":1', '"weight":1e-999'), JSON.stringify(payload).replace('"weight":3}', '"weight":1e-999}')]) {
      await page.getByLabel('Grid request JSON', { exact: true }).fill(raw)
      const rejected = page.waitForResponse(value => value.url().includes('/api/analysis/grid') && value.request().method() === 'POST')
      await page.getByRole('button', { name: 'Calculate analysis grid', exact: true }).click()
      expect((await rejected).status()).toBe(422)
      await expect(page.getByTestId('merit-evidence')).toHaveCount(0)
    }
    await page.getByLabel('Grid request JSON', { exact: true }).fill(JSON.stringify({ ...payload, objectives: undefined }))
    await expect(page.getByTestId('merit-evidence')).toHaveCount(0)
    await expect(page.getByRole('button', { name: 'Download merit JSON', exact: true })).toHaveCount(0)
    const disabled = await calculate(page, { ...payload, objectives: undefined })
    expect(disabled.merit ?? null).toBe(null)
    expect(errors).toEqual([])
  })
}
