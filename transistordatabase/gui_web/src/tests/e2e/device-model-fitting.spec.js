import { expect, test } from '@playwright/test'
import { readFile } from 'node:fs/promises'
import { createHash } from 'node:crypto'
const field = (page, name) => page.getByTestId(`fit-${name}`)
const errors = new WeakMap()
test.beforeEach(async ({ page }) => { const records = []; errors.set(page, records); page.on('pageerror', error => records.push(error.message)) })
test.afterEach(async ({ page }) => expect(errors.get(page)).toEqual([]))
async function fixture(page, family = 'ron') {
  await page.goto('/index.html?view=analytical&model=device-fitting')
  await expect(page.getByTestId('model-fitting')).toBeVisible()
  await field(page, 'family').selectOption(family); await field(page, 'example').click()
}
async function post(page, name, endpoint) {
  const started = Date.now()
  const pending = page.waitForResponse(response => response.url().endsWith(`/api/models/${endpoint}`) && response.request().method() === 'POST')
  await field(page, name).click(); const response = await pending
  const raw = await response.text(); expect(response.status(), raw).toBe(200)
  return { raw, data: JSON.parse(raw), request: response.request().postDataJSON(), elapsed_ms: Date.now() - started }
}
async function download(page, name) {
  const pending = page.waitForEvent('download'); await field(page, name).click()
  return readFile(await (await pending).path(), 'utf8')
}

for (const width of [1440, 390]) {
  test(`compiled model fitting and prediction preserve independent diagnostics and exact exports at ${width}px`, async ({ page }, info) => {
    await page.setViewportSize({ width, height: 900 }); await fixture(page)
    const trained = await post(page, 'train', 'train'), held = trained.data.diagnostics[0].holdout
    expect(held.count).toBe(1); expect(held.rmse).toBeCloseTo(.00025, 12); expect(held.r2).toBeNull()
    expect(held.rows[0]).toMatchObject({ id: 'rh', status: 'predicted', observed: .0115 })
    expect(held.rows[0].predicted).toBeCloseTo(.01125, 12); expect(held.rows[0].residual).toBeCloseTo(.00025, 12)
    expect(trained.data.model.training_observations).toHaveLength(3)
    await expect(field(page, 'metrics-holdout')).toContainText('Unavailable')
    const predicted = await post(page, 'predict', 'evaluate')
    expect(predicted.data.result.predictions[0].value).toBeCloseTo(.01125, 12)
    await expect(page.locator('svg [data-kind=train]')).toHaveCount(3)
    await expect(page.locator('svg [data-kind=holdout]')).toHaveCount(1)
    await expect(page.locator('svg [data-kind=query]')).toHaveCount(1)
    const plot = page.getByTestId('fit-plot'); await expect(plot).toBeVisible()
    const chart = await plot.boundingBox(); expect(chart.width).toBeGreaterThan(width === 390 ? 250 : 400); expect(chart.height).toBeGreaterThan(120)
    expect(await download(page, 'json')).toBe(trained.raw)
    const modelJson = await download(page, 'model-json')
    expect(JSON.parse(modelJson)).toEqual(trained.data.model); expect(modelJson).toBe(trained.data.model_json)
    expect(createHash('sha256').update(modelJson).digest('hex')).toBe(trained.data.evidence.model_sha256)
    expect(await download(page, 'prediction-json')).toBe(predicted.raw)
    const csv = await download(page, 'csv'); expect(csv).toContain('holdout'); expect(csv).toContain('Independent hand fixture'); expect(csv).toContain(trained.data.evidence.model_sha256)
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(width + 1)
    await info.attach('model-fitting-timings.json', { body: JSON.stringify({ width, train_ms: trained.elapsed_ms, predict_ms: predicted.elapsed_ms, plots: trained.data.plots.length, rows: trained.request.observations.length }), contentType: 'application/json' })
    await field(page, 'results').scrollIntoViewIfNeeded(); await page.screenshot({ path: info.outputPath(`model-fitting-${width}.png`), fullPage: true })
    const figure = page.locator('.model-fit-plot').first()
    await figure.evaluate(element => element.scrollIntoView({ block: 'center', inline: 'center' }))
    const rendered = await plot.evaluate(svg => {
      const viewport = svg.getBoundingClientRect()
      return [...svg.querySelectorAll('[data-kind="train"], [data-kind="holdout"], [data-kind="query"]')].map(marker => {
        const box = marker.getBoundingClientRect(), style = getComputedStyle(marker)
        return { kind: marker.getAttribute('data-kind'), width: box.width, height: box.height,
          inside: box.left >= viewport.left && box.right <= viewport.right && box.top >= viewport.top && box.bottom <= viewport.bottom,
          visible: style.display !== 'none' && style.visibility !== 'hidden' && Number(style.opacity) > 0 }
      })
    })
    expect(rendered).toHaveLength(5)
    for (const marker of rendered) { expect(marker.width).toBeGreaterThan(0); expect(marker.height).toBeGreaterThan(0); expect(marker.inside).toBe(true); expect(marker.visible).toBe(true) }
    await info.attach('model-fit-rendered-markers.json', { body: JSON.stringify(rendered), contentType: 'application/json' })
    await figure.screenshot({ path: info.outputPath(`model-fit-figure-${width}.png`) })
    await plot.screenshot({ path: info.outputPath(`model-fit-plot-${width}.png`) })
    await field(page, 'query-t_j_c').fill('150'); const rejected = await post(page, 'predict', 'evaluate')
    expect(rejected.data.result.predictions[0]).toMatchObject({ status: 'rejected', value: null })
    await expect(field(page, 'prediction')).toContainText('Unavailable'); await expect(page.locator('svg [data-kind=query]')).toHaveCount(0)
    await field(page, 'source').fill('Changed provenance'); await expect(field(page, 'results')).toHaveCount(0)
  })
}

test('forward and inverse diode queries retain exact temperature and unknown versus zero gates', async ({ page }) => {
  await fixture(page, 'diode'); await post(page, 'train', 'train')
  let result = await post(page, 'predict', 'evaluate'); expect(result.data.result.predictions[0].value).toBeCloseTo(.60, 12)
  await field(page, 'query-output').selectOption('current'); result = await post(page, 'predict', 'evaluate')
  expect(result.data.result.predictions[0].value).toBeCloseTo(4, 12)
  await field(page, 'query-v_g_v').fill('0'); result = await post(page, 'predict', 'evaluate')
  expect(result.request.queries[0].coordinates.v_g_v).toBe(0); expect(result.data.result.predictions[0]).toMatchObject({ value: null, status: 'rejected' })
  await field(page, 'query-v_g_v').fill(''); await field(page, 'query-t_j_c').fill('26'); result = await post(page, 'predict', 'evaluate')
  expect(result.request.queries[0].coordinates.v_g_v).toBeNull(); expect(result.data.result.predictions[0]).toMatchObject({ value: null, status: 'rejected' })
})

test('separate energy events expose isolated fit support and a real mixed-coordinate holdout', async ({ page }, info) => {
  await fixture(page, 'energy'); let trained = await post(page, 'train', 'train')
  expect(trained.request.anchor).toEqual({ current_a: 20, voltage_v: 400, t_j_c: 25, r_g_ohm: 4 })
  expect(trained.data.plots).toHaveLength(4); expect(trained.data.plots.filter(plot => plot.normalization !== null)).toHaveLength(3)
  const held = trained.data.diagnostics.find(item => item.output === 'energy').holdout
  expect(held.rows[0].predicted).toBeCloseTo(.002578125, 12); expect(held.rows[0].residual).toBeCloseTo(.0001, 12)
  const prediction = await post(page, 'predict', 'evaluate')
  expect(prediction.data.result.predictions[0].value).toBeCloseTo(.002578125, 12)
  await expect(page.getByTestId('fit-plot')).toHaveCount(4)
  await field(page, 'results').scrollIntoViewIfNeeded(); await page.screenshot({ path: info.outputPath('energy-fit-support.png'), fullPage: true })
  await field(page, 'event').selectOption('e_off'); await expect(field(page, 'results')).toHaveCount(0)
  trained = await post(page, 'train', 'train'); expect(trained.data.model.event).toBe('e_off')
  expect(trained.data.model.training_observations.every(row => row.split === 'train')).toBe(true)
})

test('raw malformed decimals cannot become zero and stale fit or query responses do not reappear', async ({ page }) => {
  await fixture(page); let trainPosts = 0
  page.on('request', request => { if (request.url().endsWith('/api/models/train')) trainPosts++ })
  await field(page, 'condition-v_g_v').fill('0x10'); await field(page, 'train').click()
  await expect(page.getByRole('alert')).toContainText('finite decimal'); expect(trainPosts).toBe(0)
  await field(page, 'condition-v_g_v').fill(''); await post(page, 'train', 'train')
  let release, received
  const intercepted = new Promise(resolve => { received = resolve })
  await page.route('**/api/models/evaluate', async route => {
    const response = await route.fetch(); received(); await new Promise(resolve => { release = resolve }); await route.fulfill({ response })
  })
  await field(page, 'predict').click(); await intercepted
  await field(page, 'query-t_j_c').fill('60'); release()
  await expect(field(page, 'predictions')).toHaveCount(0)
  await page.unroute('**/api/models/evaluate')
  const fitIntercepted = new Promise(resolve => { received = resolve })
  await page.route('**/api/models/train', async route => {
    const response = await route.fetch(); received(); await new Promise(resolve => { release = resolve }); await route.fulfill({ response })
  })
  await field(page, 'train').click(); await fitIntercepted
  await field(page, 'degree').selectOption('2'); release()
  await expect(field(page, 'results')).toHaveCount(0); await expect(field(page, 'stale')).toBeVisible()
})

test('saved model JSON validates in Python and raw underflow tokens cannot become valid zero coefficients', async ({ page }) => {
  await fixture(page); const trained = await post(page, 'train', 'train')
  await page.getByText('Reuse a saved device model JSON', { exact: true }).click()
  await field(page, 'import-json').fill(trained.data.model_json); await expect(field(page, 'results')).toHaveCount(0)
  await field(page, 'import').click(); const predicted = await post(page, 'predict', 'evaluate')
  expect(predicted.data.result.predictions[0].value).toBeCloseTo(.01125, 12)
  await expect(page.getByTestId('fit-plot')).toHaveCount(0); await expect(field(page, 'metrics-train')).toHaveCount(0)
  const tampered = trained.data.model_json.replace(/("coefficients":\[)[^,\]]+/, (_, start) => `${start}1e-999`)
  expect(tampered).toContain('1e-999'); await field(page, 'import-json').fill(tampered); await field(page, 'import').click()
  const pending = page.waitForResponse(response => response.url().endsWith('/api/models/evaluate') && response.request().method() === 'POST')
  await field(page, 'predict').click(); const response = await pending
  expect(response.status()).toBe(422); expect(response.request().postData()).toContain('1e-999')
  await expect(page.getByRole('alert')).toBeVisible(); await expect(field(page, 'predictions')).toHaveCount(0)
  await field(page, 'import-json').fill(trained.data.model_json); await field(page, 'import').click()
  const preciseTamper = trained.data.model_json.replace(/("support":\[)25(?:\.0)?(?=,)/, (_, start) => `${start}25.0000000000000001`)
  expect(preciseTamper).not.toBe(trained.data.model_json); expect(preciseTamper).toContain('25.0000000000000001')
  await field(page, 'import-json').fill(preciseTamper); await field(page, 'import').click()
  const precisePending = page.waitForResponse(result => result.url().endsWith('/api/models/evaluate') && result.request().method() === 'POST')
  await field(page, 'predict').click(); const preciseResponse = await precisePending
  expect(preciseResponse.status()).toBe(422); expect(preciseResponse.request().postData()).toContain('25.0000000000000001')
  await expect(page.getByRole('alert')).toBeVisible(); await expect(field(page, 'predictions')).toHaveCount(0)
})

test('unknown diode temperature is explicit and matches no known temperature slice', async ({ page }) => {
  await fixture(page, 'diode')
  const header = 'id,source,series,split,current_a,t_j_c,v_g_v,value'
  await field(page, 'observations').fill(`${header}\nd1,,,train,1,null,null,.55\nd2,,,train,3,,,.65\nd3,,,train,5,null,null,.75\ndh,,,holdout,2,null,null,.61`)
  await post(page, 'train', 'train'); await field(page, 'query-t_j_c').fill('')
  let predicted = await post(page, 'predict', 'evaluate')
  expect(predicted.request.queries[0].coordinates.t_j_c).toBeNull(); expect(predicted.data.result.predictions[0].value).toBeCloseTo(.60, 12)
  await field(page, 'query-t_j_c').fill('0'); predicted = await post(page, 'predict', 'evaluate')
  expect(predicted.request.queries[0].coordinates.t_j_c).toBe(0); expect(predicted.data.result.predictions[0]).toMatchObject({ value: null, status: 'rejected' })
})
