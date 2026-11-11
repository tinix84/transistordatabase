import { test, expect } from '@playwright/test'
import { readFile } from 'node:fs/promises'

const energy = { mode: 'recorded', current_a: [0, 3], energy_j: [.001, .002],
  source: 'Independent declared event model', recorded_conditions: {}, requested_conditions: {} }
const switching = { turn_on: { positive: energy }, turn_off: { positive: energy } }
function grid() {
  const configurations = [['A', [10, 15]], ['B', [20, 10]]].map(([id, drop_v]) => ({
    id, source: 'Independent constant-current fixture', switching,
    conduction: { positive: { mode: 'recorded', current_a: [1, 2], drop_v,
      source: 'Independent voltage drop', recorded_conditions: {}, requested_conditions: {} } },
  }))
  const operating_points = [1, 2].map((amps, index) => ({ id: 'I' + amps, weight: [1, 3][index],
    source: 'Independent held waveform', events: { mode: 'gate', threshold_v: .5 },
    waveform: { time_s: [0, 1], current_a: [amps, amps], gate_v: [0, 0] },
  }))
  return { configurations, operating_points, metric: { name: 'total_power_w', unit: 'W', direction: 'minimize' } }
}
async function calculate(page, route, name) {
  const [response] = await Promise.all([
    page.waitForResponse(r => r.url().endsWith('/api/' + route) && r.request().method() === 'POST'),
    page.getByRole('button', { name, exact: true }).click(),
  ])
  expect(response.status(), await response.text()).toBe(200)
  return response.json()
}
async function download(page, name) {
  const [file] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name, exact: true }).click()])
  return readFile(await file.path(), 'utf8')
}
async function prepareGrid(page, data) {
  await page.goto('/index.html?view=analysis-grid')
  await page.getByText('Advanced canonical request, including explicit event masks', { exact: true }).click()
  await page.getByLabel('Use grid JSON as input').check()
  await page.getByLabel('Grid request JSON').fill(JSON.stringify(data))
  return calculate(page, 'analysis/grid', 'Calculate analysis grid')
}
function canvas(page, title) {
  return page.locator('.curve-plotter').filter({ has: page.getByRole('heading', { name: title, exact: true }) }).locator('canvas')
}
async function traceSignature(chart) {
  return chart.evaluate(el => {
    const rgba = el.getContext('2d').getImageData(0, 0, el.width, el.height).data
    let count = 0, sum = 0
    for (let i = 0; i < rgba.length; i += 4) {
      // The first CurvePlotter dataset uses teal. Axes and body text do not.
      if (rgba[i + 3] > 100 && rgba[i + 1] - rgba[i] > 80 && Math.abs(rgba[i + 1] - rgba[i + 2]) < 12) {
        count++; sum += i / 4
      }
    }
    return { count, sum }
  })
}
async function cursorColumn(chart) {
  return chart.evaluate(el => {
    const { data } = el.getContext('2d').getImageData(0, 0, el.width, el.height)
    const columns = Array(el.width).fill(0)
    for (let i = 0; i < data.length; i += 4) {
      if (data[i + 3] > 50 && Math.abs(data[i] - 124) < 8 && Math.abs(data[i + 1] - 58) < 8
        && Math.abs(data[i + 2] - 237) < 8) columns[(i / 4) % el.width]++
    }
    const peak = Math.max(...columns)
    return peak > 10 ? columns.indexOf(peak) : -1
  })
}
async function stableTrace(chart) {
  let last = '', stable = 0
  await expect.poll(async () => {
    const snapshot = JSON.stringify(await traceSignature(chart))
    stable = snapshot === last ? stable + 1 : 0
    last = snapshot
    return stable
  }, { intervals: [150] }).toBeGreaterThanOrEqual(3)
  return JSON.parse(last)
}
async function spectrumPeaks(chart, lastFrequency) {
  return chart.evaluate((el, extent) => {
    const { data } = el.getContext('2d').getImageData(0, 0, el.width, el.height)
    const columns = Array(el.width).fill(Infinity)
    let bottom = 0
    for (let y = 0; y < el.height; y++) {
      for (let x = 0; x < el.width; x++) {
        const i = (y * el.width + x) * 4
        if (data[i + 3] < 100 || data[i + 1] - data[i] < 80 || Math.abs(data[i + 1] - data[i + 2]) > 12) continue
        columns[x] = Math.min(columns[x], y)
        bottom = Math.max(bottom, y)
      }
    }
    const left = columns.findIndex(Number.isFinite), right = columns.findLastIndex(Number.isFinite)
    const dcTop = Math.min(...columns)
    const firstInterior = left + Math.floor((right - left) / 9)
    const acTop = Math.min(...columns.slice(firstInterior))
    const position = columns.indexOf(acTop, firstInterior)
    return { frequency: extent * (position - left) / (right - left), normalizedAmplitude: (bottom - acTop) / (bottom - dcTop) }
  }, lastFrequency)
}
async function stage(page, network = 'forward', partial = false) {
  await page.getByText('Send exact waveform heat to the thermal workbench', { exact: true }).click()
  for (const [direction, owner] of [['positive', network], ['negative', 'forward'], ['zero', 'forward']]) {
    await page.getByLabel(direction + '-current heat owner', { exact: true }).selectOption(owner)
  }
  await page.getByLabel('Handoff thermal network').selectOption(network)
  if (partial) await page.getByLabel(/I accept known components only/).check()
  const handoff = await calculate(page, 'analysis/thermal-handoff', 'Prepare thermal handoff')
  await expect(page.getByTestId('staged-thermal-handoff')).toContainText(handoff.source_input_sha256)
  await expect(page.getByTestId('staged-thermal-handoff')).toContainText(handoff.input_sha256)
  return handoff
}
async function solveThermal(page) {
  await page.getByLabel('Reference temperature').fill('25')
  await page.getByLabel('Foster poles', { exact: true }).fill('2,1')
  await page.getByLabel(/I reviewed the thermal source/).check()
  return calculate(page, 'thermal/analyze', 'Calculate thermal result')
}

test('finite grid preserves failed rows, real means, ranking and selected heat', async ({ page }) => {
  const data = grid(), response = await prepareGrid(page, data)
  expect(response.ranking.map(r => r.value)).toEqual([20, 25])
  const panel = page.getByTestId('grid-results')
  await expect(panel.locator('.bars li')).toHaveText([/B.*20.*W/, /A.*25.*W/])
  const widths = await panel.locator('.bar-track > div').evaluateAll(els => els.map(el => el.getBoundingClientRect().width))
  expect(widths[0] / widths[1]).toBeCloseTo(.8, 2)
  data.configurations[0].conduction.positive.current_a = [.5, 1]
  data.configurations[0].conduction.positive.drop_v = [8, 10]
  data.summary_mode = 'successful-subset'
  await page.getByLabel('Grid request JSON').fill(JSON.stringify(data))
  await expect(panel).toHaveCount(0)
  const partial = await calculate(page, 'analysis/grid', 'Calculate analysis grid')
  expect(partial.rows.map(r => r.status)).toEqual(['computed', 'unsupported', 'computed', 'computed'])
  await expect(panel.locator('tr').filter({ hasText: /0\.250000/ })).toContainText('1 / 4')
  await page.getByLabel('Filter pair diagnostics').selectOption('unsupported')
  await expect(panel.locator('.pair-list button')).toHaveText(['A × I2 · unsupported'])
  await panel.locator('.pair-list button').click()
  await expect(page.getByTestId('selected-grid-pair').getByRole('alert')).toContainText(/supports/)
  expect(JSON.parse(await download(page, 'Download grid request + result JSON')).response).toEqual(partial)
  await page.getByLabel('Filter pair diagnostics').selectOption('all')
  await page.getByRole('button', { name: 'B × I1 · computed', exact: true }).click()
  const selected = partial.rows.find(r => r.configuration_id === 'B' && r.operating_point_id === 'I1')
  const handoff = await stage(page, 'reverse')
  expect(handoff.waveform_result.power_segments).toEqual(selected.result.power_segments)
  const thermal = await solveThermal(page)
  expect(thermal.request.segments).toEqual(handoff.thermal_input.segments)
  expect(thermal.result.final_temperature_c).toBeCloseTo(25 + 2 * 20 * (1 - Math.exp(-1)), 10)
  const exported = JSON.parse(await download(page, 'Export thermal request + result JSON'))
  expect(exported.waveform_handoff).toEqual(handoff)
})

test('coherent spectrum and linked time plots retain their independent axes at 390 px', async ({ page }, info) => {
  await page.setViewportSize({ width: 390, height: 844 })
  const data = grid(), p = data.operating_points[0], n = 32
  p.spectrum = { window: 'rectangular', detrend: 'none' }
  p.waveform = { time_s: Array.from({ length: n }, (_, k) => k / 128),
    current_a: Array.from({ length: n }, (_, k) => 1.5 + .25 * Math.sin(6 * Math.PI * k / n)),
    gate_v: Array(n).fill(0) }
  const electrical = (await prepareGrid(page, data)).rows[0].result
  expect(electrical.spectrum.frequency_hz[3]).toBe(12)
  expect(electrical.spectrum.amplitude_a[3]).toBeCloseTo(.25, 12)
  expect(electrical.spectrum.amplitude_a[0]).toBeCloseTo(1.5, 12)
  const current = canvas(page, 'Signed current'), fft = canvas(page, 'Uniform-sample current spectrum')
  await expect.poll(async () => (await traceSignature(fft)).count).toBeGreaterThan(100)
  await fft.scrollIntoViewIfNeeded()
  const before = await stableTrace(fft)
  await page.screenshot({ path: info.outputPath('390px-grid-spectrum.png') })
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390)
  const peaks = await spectrumPeaks(fft, 64)
  expect(Math.abs(peaks.frequency - 12)).toBeLessThan(.7)
  expect(Math.abs(peaks.normalizedAmplitude - .25 / 1.5)).toBeLessThan(.015)
  const gate = canvas(page, 'Gate event evidence')
  await page.getByTestId('time-cursor-sample').selectOption('8')
  await expect(page.getByTestId('time-cursor-value')).toHaveText('0.0625 s')
  await expect(page.getByLabel('Signed current cursor readout')).toContainText('1.25000 A')
  await expect.poll(() => cursorColumn(current)).toBeGreaterThan(0)
  await expect.poll(() => cursorColumn(gate)).toBeGreaterThan(0)
  const atEight = await Promise.all([cursorColumn(current), cursorColumn(gate)])
  await page.getByTestId('time-cursor-sample').selectOption('16')
  for (const [index, chart] of [current, gate].entries()) {
    await expect.poll(() => cursorColumn(chart)).toBeGreaterThan(atEight[index] + 10)
  }
  expect(await cursorColumn(fft)).toBe(-1)
  const timeBefore = await stableTrace(current)
  await page.getByTestId('range-start').fill('0.05')
  await page.getByTestId('range-end').fill('0.125')
  await expect.poll(() => traceSignature(current)).not.toEqual(timeBefore)
  expect(await stableTrace(fft)).toEqual(before)
})

test('waveform source sends exact partial cubics and impulses into the reviewed network', async ({ page }) => {
  await page.goto('/index.html?view=waveform')
  await page.getByText('Advanced: inspect or supply the canonical JSON request', { exact: true }).click()
  await page.getByLabel('Use JSON editor as calculation input').check()
  const source = {
    waveform: { time_s: [3, 4], current_a: [1, 2], gate_v: [0, 1] },
    events: { mode: 'gate', threshold_v: .5 },
    conduction: { duty: [0, 1], positive: { mode: 'linear-drop-estimate', selected: true,
      offset_v: 0, resistance_ohm: 2, conditions: {}, source: 'Independent 2 ohm' } },
    switching: { turn_on: { positive: { ...energy, energy_j: [.5, .5] } },
      turn_off: { omit: true, omission_reason: 'No turn-off energy characterization' } },
  }
  await page.getByLabel('Canonical request').fill(JSON.stringify(source))
  const first = await calculate(page, 'waveform/analyze', 'Calculate waveform losses')
  const stack = page.getByTestId('loss-component-stack')
  await expect(stack.locator('[data-component="turn_off"]')).toHaveCount(0)
  await expect(stack).toContainText(source.switching.turn_off.omission_reason)
  const bars = await stack.locator('rect[data-component]').evaluateAll(els => els.map(el => ({
    component: el.dataset.component, watts: +el.dataset.watts, width: +el.getAttribute('width'),
    paintedWidth: el.getBoundingClientRect().width,
  })))
  expect(bars.map(b => b.component)).toEqual(['conduction', 'turn_on'])
  expect(bars[0].watts).toBeCloseTo(17 / 6, 12)
  expect(bars[1].watts).toBe(.5)
  expect(bars[0].paintedWidth / bars[1].paintedWidth).toBeCloseTo(17 / 3, 4)
  const handoff = await stage(page, 'reverse', true)
  expect(handoff.omission_reasons.turn_off).toBe(source.switching.turn_off.omission_reason)
  expect(handoff.scope).toBe('known-components-only')
  expect(handoff.waveform_result.power_segments).toEqual(first.power_segments)
  expect(handoff.thermal_input.segments[0].power_coefficients_w).toEqual([0, 2, 4, 2])
  expect(handoff.thermal_input.events[0]).toMatchObject({ energy_j: .5, time_s: 3, heat_owner: 'reverse' })
  const [submitted, thermal] = await Promise.all([
    page.waitForRequest(r => r.url().endsWith('/api/thermal/analyze') && r.method() === 'POST'),
    solveThermal(page),
  ])
  expect(submitted.postDataJSON().events).toEqual(handoff.thermal_input.events)
  expect(submitted.postDataJSON().segments).toEqual(handoff.thermal_input.segments)
  expect(thermal.result.final_temperature_c).toBeCloseTo(25 + 13 / Math.E, 10)
  expect(thermal.result.conduction_energy_j).toBeCloseTo(17 / 6, 12)
})

test('map cells, source lines and missing and ambiguous evidence at 390 px', async ({ page }, info) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('/index.html?view=energy-map')
  await page.getByLabel(/I select bounded bilinear/).check()
  await page.getByLabel('Estimate current', { exact: true }).fill('5')
  await page.getByLabel('Estimate voltage', { exact: true }).fill('600')
  const result = await calculate(page, 'analysis/energy-map', 'Build energy map')
  expect(result.estimate.energy_j).toBeCloseTo(.00075, 12)
  const panel = page.getByTestId('energy-map-results')
  await expect(page.getByTestId('energy-estimate')).toContainText('0.000750000 J')
  const cells = panel.locator('.heatmap td')
  expect(await cells.first().evaluate(el => getComputedStyle(el).backgroundColor))
    .not.toEqual(await cells.last().evaluate(el => getComputedStyle(el).backgroundColor))
  const chart = canvas(page, 'Original source slices (all sources, including excluded slices)')
  await expect.poll(async () => (await traceSignature(chart)).count).toBeGreaterThan(50)
  await page.getByTestId('energy-estimate').scrollIntoViewIfNeeded()
  await page.screenshot({ path: info.outputPath('390px-energy-estimate.png') })
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390)
  const saved = JSON.parse(await download(page, 'Download map request + result JSON'))
  expect(saved.response).toEqual(result)
  const request = saved.request
  request.slices.push({ ...request.slices[0], id: 'other', source: 'Second source' })
  request.slices[1].conditions = { ...request.slices[1].conditions, commutation_source: 'Different source' }
  await page.getByText('Advanced canonical map request', { exact: true }).click()
  await page.getByLabel('Use map JSON as input').check()
  await page.getByLabel('Map request JSON').fill(JSON.stringify(request))
  await calculate(page, 'analysis/energy-map', 'Build energy map')
  await expect(cells.first()).toContainText('ambiguous')
  await expect(cells.last()).toContainText('missing')
  await cells.first().getByRole('button').click()
  await expect(panel).toContainText('Second source')
  await expect(page.getByTestId('energy-estimate')).toContainText('unsupported')
})
