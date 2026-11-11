import { expect, test } from '@playwright/test'
import { readFile } from 'node:fs/promises'
import { inspectRenderedChart } from '../fixtures/chartEvidence.js'

const field = (page, name) => page.getByTestId(`sharing-${name}`)
async function fixture(page) {
  await page.goto('/index.html?view=assemblies')
  await page.getByRole('link', { name: 'Go to reverse-current sharing', exact: true }).click()
  await expect(page.getByTestId('reverse-sharing').locator('summary').first()).toBeInViewport()
  await page.getByTestId('reverse-sharing').locator('summary').first().click()
  await field(page, 'fixture').click()
  await field(page, 'selected').check(); await field(page, 'compatible').check()
}
async function calculate(page) {
  const waiting = page.waitForResponse(response => response.url().includes('/api/assemblies/reverse-sharing'))
  await field(page, 'calculate').click()
  const response = await waiting, raw = await response.text()
  expect(response.status(), raw).toBe(200)
  await expect(field(page, 'results')).toBeVisible()
  return { data: JSON.parse(raw), raw }
}
async function download(page, name) {
  const waiting = page.waitForEvent('download'); await field(page, name).click()
  return readFile(await (await waiting).path(), 'utf8')
}
async function chart(page) {
  await page.waitForTimeout(1000)
  return field(page, 'results').locator('.curve-plotter').evaluate(inspectRenderedChart)
}
for (const width of [1440, 390]) {
  test(`static reverse split, bounded support and exact portable evidence at ${width}px`, async ({ page }, info) => {
    const errors = []; page.on('pageerror', error => errors.push(error.message)); page.on('console', message => { if (message.type() === 'error') errors.push(message.text()) })
    await page.setViewportSize({ width, height: 900 }); await fixture(page)
    const { data, raw } = await calculate(page), result = data.result
    expect(result.status).toBe('supported'); expect(result.support).toBe('recorded')
    expect(result.channel_current_a).toBeCloseTo(-20 / 3, 10); expect(result.diode_current_a).toBeCloseTo(-10 / 3, 10)
    expect(result.voltage_v).toBeCloseTo(-2 / 3, 10); expect(result.channel_static_power_w).toBeCloseTo(40 / 9, 10); expect(result.diode_static_power_w).toBeCloseTo(20 / 9, 10)
    await expect(field(page, 'results')).toContainText('not a measured current split')
    expect(await download(page, 'json')).toBe(raw)
    const csv = await download(page, 'csv')
    expect(csv).toContain('conditions_and_sources'); expect(csv).toContain('isolation_evidence'); expect(csv).toContain('root_evidence'); expect(csv).toContain(data.evidence.input_sha256)
    let plot = await chart(page)
    expect(plot.datasets).toHaveLength(4); expect(plot.xMin).toBeLessThan(0); expect(plot.xMax).toBeGreaterThan(10); expect(plot.yMin).toBeLessThan(0); expect(plot.yMax).toBeGreaterThan(1)
    for (const dataset of plot.datasets) for (const point of dataset.points) {
      expect(point.value).toBeCloseTo(point.input.y, 8); expect(point.input.x).toBeGreaterThan(plot.xMin); expect(point.input.x).toBeLessThan(plot.xMax); expect(point.input.y).toBeGreaterThan(plot.yMin); expect(point.input.y).toBeLessThan(plot.yMax)
      const radius = point.radius + point.borderWidth / 2
      expect(point.pixel.x - radius).toBeGreaterThan(plot.chartArea.left); expect(point.pixel.x + radius).toBeLessThan(plot.chartArea.right)
      expect(point.pixel.y - radius).toBeGreaterThan(plot.chartArea.top); expect(point.pixel.y + radius).toBeLessThan(plot.chartArea.bottom)
    }
    await field(page, 'results').screenshot({ path: info.outputPath(`reverse-sharing-${width}.png`) })
    const renderedPlot = field(page, 'results').locator('.curve-plotter')
    await renderedPlot.evaluate(node => node.scrollIntoView({ block: 'center' }))
    await renderedPlot.screenshot({ path: info.outputPath(`reverse-sharing-plot-${width}.png`) })
    await field(page, 'current').fill('-4'); await expect(field(page, 'results')).toHaveCount(0); await expect(field(page, 'json')).toHaveCount(0)
    const strict = (await calculate(page)).data.result
    expect(strict.status).toBe('unsupported'); expect(strict.channel_current_a).toBe(null); expect(strict.diode_current_a).toBe(null); expect(strict.voltage_v).toBe(null)
    plot = await chart(page); expect(plot.datasets).toHaveLength(2)
    await field(page, 'mode').selectOption('ideal-diode-blocking'); const estimated = (await calculate(page)).data.result
    expect(estimated.support).toBe('estimate'); expect(estimated.channel_current_a).toBe(-4); expect(estimated.diode_current_a).toBe(0); expect(estimated.voltage_v).toBeCloseTo(-.4, 12); expect(estimated.blocking_margin_v).toBeCloseTo(.2, 12)
    plot = await chart(page); expect(plot.datasets.at(-1).label).toContain('off source curve'); expect(plot.datasets.at(-1).points[0].input.y).toBeCloseTo(.4, 12)
    await field(page, 'current').fill('0'); const zero = (await calculate(page)).data.result
    expect(zero.status).toBe('supported'); expect(zero.channel_current_a).toBe(0); expect(zero.diode_current_a).toBe(0); expect(zero.total_static_power_w).toBe(0)
    await field(page, 'diode-recorded-gate').fill(''); await field(page, 'diode-requested-gate').fill('')
    const unknown = await calculate(page)
    expect(unknown.data.inputs.diode.recorded_conditions.gate_bias_v).toBe(null); expect(unknown.data.inputs.diode.requested_conditions.gate_bias_v).toBe(null)
    expect(unknown.data.inputs.channel.recorded_conditions.gate_bias_v).toBe(15); expect(await download(page, 'json')).toBe(unknown.raw)
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(width + 1)
    expect(errors).toEqual([])
  })
}
test('source acknowledgments, raw decimals, condition errors and stale responses', async ({ page }) => {
  await fixture(page); await field(page, 'selected').uncheck(); await field(page, 'calculate').click()
  await expect(page.getByTestId('reverse-sharing').getByRole('alert')).toContainText('Acknowledge')
  await field(page, 'selected').check(); await field(page, 'current').fill('-1e-999'); await field(page, 'calculate').click()
  await expect(page.getByTestId('reverse-sharing').getByRole('alert')).toContainText('too small')
  await field(page, 'current').fill(''); await field(page, 'calculate').click()
  await expect(page.getByTestId('reverse-sharing').getByRole('alert')).toContainText('finite decimal')
  await field(page, 'current').fill('-10'); await field(page, 'channel-recorded-gate').fill('1e-999'); await field(page, 'calculate').click()
  await expect(page.getByTestId('reverse-sharing').getByRole('alert')).toContainText('too small')
  await field(page, 'channel-recorded-gate').fill('15'); await field(page, 'diode-requested-gate').fill('2')
  const failure = page.waitForResponse(response => response.url().includes('/api/assemblies/reverse-sharing'))
  await field(page, 'calculate').click(); expect((await failure).status()).toBe(422)
  await expect(page.getByTestId('reverse-sharing').getByRole('alert')).toBeVisible()
  await field(page, 'diode-requested-gate').fill('0')
  let release
  const gate = new Promise(resolve => { release = resolve })
  await page.route('**/api/assemblies/reverse-sharing', async route => { const response = await route.fetch(); await gate; await route.fulfill({ response }) })
  const sent = page.waitForRequest(request => request.url().includes('/api/assemblies/reverse-sharing'))
  await field(page, 'calculate').click(); await sent; await field(page, 'channel-source').fill('Changed while pending'); release()
  await page.waitForTimeout(300); await expect(field(page, 'results')).toHaveCount(0); await expect(field(page, 'json')).toHaveCount(0)
})
