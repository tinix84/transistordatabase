import { test, expect } from '@playwright/test'
import { readFile } from 'node:fs/promises'
import { createHash } from 'node:crypto'
const root2 = Math.SQRT2
const values = [4, 1 + root2, 0, 1 - root2, 0, 1 - root2, 0, 1 + root2]
const omission = { omit: true, omission_reason: 'Independent spectrum-only numerical fixture' }
function fixture(samples, sampleRate = samples.length) {
  return { waveform: { time_s: samples.map((_, n) => n / sampleRate), current_a: samples }, conduction: omission,
    events: { mode: 'omitted', omission_reason: 'Spectrum-only fixture; no gate event inference' },
    switching: { turn_on: omission, turn_off: omission }, spectrum: { window: 'rectangular', detrend: 'none' },
    provenance: { source: 'Independent exact-bin spectrum browser oracle', calibration_notes: 'SI A and s; declared uniform sampling' } }
}
const sha = text => createHash('sha256').update(text).digest('hex')
async function attach(info, name, value) { await info.attach(name, { body: JSON.stringify(value, null, 2), contentType: 'application/json' }) }
async function calculate(page) {
  const ready = page.waitForResponse(r => r.url().endsWith('/api/waveform/analyze') && r.request().method() === 'POST')
  await page.getByRole('button', { name: 'Calculate waveform losses', exact: true }).click()
  const response = await ready
  return { status: response.status(), body: await response.json(), raw_request: response.request().postData() }
}
async function advanced(page, payload) {
  const checkbox = page.getByLabel('Use JSON editor as calculation input', { exact: true })
  if (!(await checkbox.isVisible())) await page.getByText('Advanced: inspect or supply the canonical JSON request', { exact: true }).click()
  await checkbox.check()
  await page.getByLabel('Canonical request', { exact: true }).fill(typeof payload === 'string' ? payload : JSON.stringify(payload))
  return calculate(page)
}
async function download(page, button, info, name) {
  const ready = page.waitForEvent('download'); await button.click(); const item = await ready
  const path = info.outputPath(name); await item.saveAs(path); return readFile(path, 'utf8')
}
test.beforeEach(async ({ page }, info) => {
  const capture = { console: [], pageerrors: [], failed_requests: [], http5xx: [] }
  page.on('console', item => capture.console.push({ type: item.type(), text: item.text() }))
  page.on('pageerror', error => capture.pageerrors.push(error.message))
  page.on('requestfailed', item => capture.failed_requests.push({ url: item.url(), failure: item.failure() }))
  page.on('response', item => { if (item.status() >= 500) capture.http5xx.push({ url: item.url(), status: item.status() }) })
  info.capture = capture
  await page.goto('/index.html?view=waveform')
  await expect(page.getByRole('heading', { name: 'Waveform workbench', exact: true })).toBeVisible()
})
test.afterEach(async ({}, info) => {
  await attach(info, 'browser-console-network', info.capture)
  expect(info.capture.pageerrors).toEqual([]); expect(info.capture.failed_requests).toEqual([]); expect(info.capture.http5xx).toEqual([])
})

test('spectrum acceptance actual controls resolve independent peak mask and raw band RMS', async ({ page }, info) => {
  await page.getByTestId('waveform-csv').fill('time_s,current_a\n' + values.map((v, n) => `${n / 8},${v}`).join('\n'))
  await page.getByLabel('Explicitly omit conduction', { exact: true }).check()
  await page.getByLabel('Reason for omitting conduction', { exact: true }).fill('Independent spectrum-only fixture')
  await page.getByLabel('Event policy', { exact: true }).selectOption('omitted')
  await page.getByLabel('Reason for omitting event detection', { exact: true }).fill('No switching model inferred')
  await page.getByLabel('Compute uniform-sample FFT (reject irregular sampling)', { exact: true }).check()
  await page.getByTestId('spectrum-mask-enabled').check()
  await page.getByTestId('spectrum-mask-source').fill('Independent 1.5 A peak limit at 1 Hz; exact 1 A at 2 Hz')
  await page.getByTestId('spectrum-mask-csv').fill('frequency_hz,limit_peak\n0,10\n1,1.5\n2,1\n3,10\n4,10')
  await page.getByTestId('spectrum-bands-enabled').check()
  await page.getByTestId('spectrum-bands-csv').fill('name,lower_hz,upper_hz,endpoints\ntones,1,2,closed\nwhole,0,4,closed')
  const result = await calculate(page); expect(result.status).toBe(200)
  const spectrum = result.body.spectrum, screen = spectrum.mask_screening
  expect(screen.violating_bin_indices).toEqual([1]); expect(screen.coverage_complete).toBe(true)
  expect(screen.bins[1].excess_peak).toBeCloseTo(.5, 14)
  expect(screen.bins[2].amplitude_peak).toBe(1); expect(screen.bins[2].status).toBe('pass')
  expect(spectrum.band_rms[0].rms).toBeCloseTo(Math.sqrt(2.5), 14)
  expect(spectrum.band_rms[1].rms).toBeCloseTo(Math.sqrt(3.5), 14)
  expect(screen.bins.map(r => r.amplitude_peak)).toEqual(spectrum.amplitude_a)
  await expect(page.getByTestId('spectrum-band-results')).toContainText('Raw windowed detrended discrete Parseval RMS; no power-gain correction')
  const maskCsv = await download(page, page.getByTestId('spectrum-mask-download'), info, 'mask.csv')
  expect(maskCsv.trim().split(/\r?\n/)).toHaveLength(6); expect(maskCsv).toContain('quantity_basis'); expect(maskCsv).not.toContain('"bins"')
  const bandCsv = await download(page, page.getByTestId('spectrum-bands-download'), info, 'bands.csv')
  expect(bandCsv.trim().split(/\r?\n/)).toHaveLength(3); expect(bandCsv).toContain('Raw windowed')
  const exported = JSON.parse(await download(page, page.getByTestId('spectrum-analysis-download'), info, 'spectrum.json'))
  expect(exported).toEqual(spectrum)
  await attach(info, 'real-api-and-identity', { ...result, response_sha256: sha(JSON.stringify(result.body)) })
  await page.screenshot({ path: info.outputPath('mask-band-oracle.png'), fullPage: true })
  await page.getByTestId('spectrum-mask-source').fill('Changed source invalidates previous result')
  await expect(page.getByTestId('spectrum-analysis-results')).toHaveCount(0)
  await expect(page.getByRole('status').filter({ hasText: 'Inputs changed. Calculate again' })).toBeVisible()
})

test('spectrum acceptance all bins remain independent of display selection gaps and exact canonical equality', async ({ page }, info) => {
  const data = fixture([1, ...Array(63).fill(0)]); data.spectrum.relative_threshold = 1
  data.spectrum.mask = { unit: 'A_peak', frequency_hz: Array.from({ length: 33 }, (_, i) => i),
    limit_peak: Array.from({ length: 33 }, (_, i) => i === 0 || i === 32 || i === 20 ? 1 / 64 : 1 / 32), source: 'Independent impulse coefficients', interpolation: 'exact' }
  let result = await advanced(page, data); expect(result.status).toBe(200)
  let spectrum = result.body.spectrum
  expect(spectrum.selected_bin_indices).toEqual([]); expect(spectrum.mask_screening.observed_bin_count).toBe(33)
  expect(spectrum.mask_screening.violating_bin_indices).toEqual([20])
  expect(spectrum.mask_screening.bins[20].excess_peak).toBe(1 / 64)
  expect(spectrum.mask_screening.bins.map(r => r.amplitude_peak)).toEqual(spectrum.amplitude_a)
  data.spectrum.mask = { unit: 'A_peak', frequency_hz: [0, 1, 2, 4, 8, 16, 32], limit_peak: [null, 1 / 32, null, 1 / 32, 1 / 32, 1 / 32, 1 / 32], source: 'Explicit null gaps with log-frequency segments', interpolation: 'log-frequency' }
  result = await advanced(page, data); expect(result.status).toBe(200); spectrum = result.body.spectrum
  expect(spectrum.mask_screening.status).toBe('incomplete'); expect(spectrum.mask_screening.uncovered_bin_indices).toEqual([0, 2, 3])
  expect(spectrum.mask_screening.bins[6].coverage).toBe('interpolated'); expect(spectrum.mask_screening.bins[6].status).toBe('pass')
  await expect(page.getByTestId('spectrum-mask-results')).toContainText('Uncovered bins are unknown')
  await page.screenshot({ path: info.outputPath('partial-log-coverage.png'), fullPage: true })
  await attach(info, 'partial-real-api', result)
  data.spectrum.mask = { unit: 'A_peak', frequency_hz: [0, 32], limit_peak: [null, null], source: 'Explicitly unavailable profile', interpolation: 'linear-frequency' }
  result = await advanced(page, data); expect(result.status).toBe(200)
  expect(result.body.spectrum.mask_screening.status).toBe('uncovered'); expect(result.body.spectrum.mask_screening.covered_bin_count).toBe(0)
  await expect(page.getByTestId('spectrum-mask-results').getByRole('heading', { name: 'Declared mask: uncovered', exact: true })).toBeVisible()
})

test('spectrum acceptance advanced raw JSON preserves rejected coordinate underflow and hysteresis lexemes', async ({ page }, info) => {
  const data = fixture(values)
  data.spectrum.mask = { unit: 'A_peak', frequency_hz: [1], limit_peak: [1.5], source: 'Raw coordinate precision oracle' }
  const variants = []
  variants.push(JSON.stringify(data).replace('"frequency_hz":[1]', '"frequency_hz":[1.0000000000000000001]'))
  variants.push(JSON.stringify(data).replace('"limit_peak":[1.5]', '"limit_peak":[-1e-400]'))
  const gate = fixture([1, 1, 1, 1]); delete gate.spectrum; gate.waveform.gate_v = [0, 10, 0, 10]
  gate.events = { mode: 'hysteresis', digitizer: { thresholds: { scale: 'relative', low_fraction: .25, high_fraction: 1, minimum_v: 0, maximum_v: 10 } } }
  variants.push(JSON.stringify(gate).replace('"high_fraction":1', '"high_fraction":1.0000000000000000001'))
  for (const [i, raw] of variants.entries()) {
    const result = await advanced(page, raw); expect(result.status).toBe(422)
    expect(result.raw_request).toBe(raw)
    await expect(page.locator('form').first().getByRole('alert')).toBeVisible()
    await attach(info, `raw-lexeme-${i}`, { ...result, request_sha256: sha(raw) })
  }
})

test('spectrum acceptance tiny raw band RMS remains nonzero while mask-only peak underflow rejects', async ({ page }, info) => {
  const data = fixture(Array.from({ length: 8 }, (_, i) => i % 2 ? -1e-200 : 1e-200))
  data.spectrum.mask = { unit: 'A_peak', frequency_hz: [4], limit_peak: [1e-200], source: 'Tiny SI Nyquist equality oracle' }
  data.spectrum.bands = [{ name: 'nyquist-only', lower_hz: 4, upper_hz: 4, endpoints: 'closed' }, { name: 'empty-open', lower_hz: 1.1, upper_hz: 1.2, endpoints: 'open' }]
  let result = await advanced(page, data); expect(result.status).toBe(200)
  expect(result.body.spectrum.band_rms[0].rms).toBeGreaterThan(0)
  expect(result.body.spectrum.band_rms[0].rms / 1e-200).toBeCloseTo(1, 14)
  expect(result.body.spectrum.band_rms[1].rms).toBeNull(); expect(result.body.spectrum.band_rms[1].status).toBe('no-bins')
  expect(result.body.spectrum.mask_screening.bins[4].status).toBe('pass')
  await page.screenshot({ path: info.outputPath('tiny-band-rms.png'), fullPage: true })
  await attach(info, 'tiny-real-api', result)
  const small = fixture([5e-324, 0, 0, 0]); small.spectrum.mask = { unit: 'A_peak', frequency_hz: [0, 1, 2], limit_peak: [0, 0, 0], source: 'Unrepresentable peak must reject' }
  result = await advanced(page, small); expect(result.status).toBe(422)
  expect(result.body.detail).toContain('nonzero FFT coefficient')
  await attach(info, 'mask-only-underflow-rejection', result)
})
