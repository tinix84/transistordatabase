import { expect, test } from '@playwright/test'
import { inspectRenderedChart } from '../fixtures/chartEvidence.js'

test('allocation preserves original decimal inputs and renders isolated sample markers', async ({ page }) => {
  await page.goto('/index.html?view=waveform')
  const panel = page.getByTestId('gate-allocation')
  const csv = page.getByTestId('waveform-csv')
  const field = name => panel.getByTestId(`allocation-${name}`)
  const requests = []
  page.on('request', request => {
    if (request.url().endsWith('/api/waveform/gate-allocation')) requests.push(request.postDataJSON())
  })

  await panel.locator('summary').first().click()
  await field('selected').check()
  await field('source').fill('Independent precision review')
  await csv.fill('time_s,current_a,gate_v\n-3,-4,0\n1,5,.99999999999999999')
  await field('calculate').click()
  await expect(panel.getByRole('alert')).toContainText('0/1')
  await expect(field('results')).toHaveCount(0)
  await csv.fill('time_s,current_a,gate_v\n-3,-4,-1\n1,5,0')
  await field('mode').selectOption('threshold')
  await field('threshold').fill('-1e-999')
  await field('calculate').click()
  await expect(panel.getByRole('alert')).toContainText('too small')
  await field('threshold').fill('-1')
  const temperature = panel.getByLabel('Junction temperature')
  await temperature.fill('1e-999')
  await field('calculate').click()
  await expect(panel.getByRole('alert')).toContainText('too small')
  expect(requests).toHaveLength(0)

  await temperature.fill('0')
  const pending = page.waitForResponse(response => response.url().endsWith('/api/waveform/gate-allocation'))
  await field('calculate').click()
  const envelope = await (await pending).json()
  expect(envelope.inputs.conditions).toEqual({ junction_temperature_c: 0, gate_on_voltage_v: null })
  expect(envelope.result.samples.map(row => [row.gate_on, row.gate_on_current_a, row.gate_off_current_a]))
    .toEqual([[false, 0, -4], [true, 5, 0]])
  await expect(field('results').locator('canvas')).toHaveCount(2)
  const plots = await field('results').locator('.curve-plotter').all()
  const charts = await Promise.all(plots.map(plot => plot.evaluate(inspectRenderedChart)))
  const lines = charts.flatMap(chart => chart.datasets.map(dataset => dataset.showLine))
  expect(lines).toEqual([false, false, false, false])

  await page.getByLabel('Waveform source (blank means unknown)').fill('Revised calibration context')
  await expect(field('json')).toHaveCount(0)
  await expect(field('csv')).toHaveCount(0)
})
