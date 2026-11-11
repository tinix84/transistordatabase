import { test, expect } from '@playwright/test'
import fs from 'node:fs/promises'
import { createHash } from 'node:crypto'

const origin = 'http://localhost:8002'
async function imageBytes(page, color = 'white') {
  const data = await page.evaluate(color => {
    const canvas = document.createElement('canvas'); canvas.width = 1000; canvas.height = 600
    const ctx = canvas.getContext('2d'); ctx.fillStyle = color; ctx.fillRect(0, 0, 1000, 600)
    ctx.strokeStyle = '#394c65'; ctx.lineWidth = 2
    ctx.beginPath(); ctx.moveTo(100, 100); ctx.lineTo(100, 500); ctx.lineTo(900, 500); ctx.stroke()
    ctx.fillStyle = '#b42343'; ctx.beginPath(); ctx.arc(500, 300, 4, 0, Math.PI * 2); ctx.fill()
    return canvas.toDataURL('image/png').split(',')[1]
  }, color)
  return Buffer.from(data, 'base64')
}
async function upload(page, bytes, name = 'calibrated-figure.png') {
  const pending = page.waitForResponse(r => r.url().endsWith('/api/digitizer/image'))
  await page.getByTestId('digitizer-file').setInputFiles({ name, mimeType: 'image/png', buffer: bytes })
  const reply = await pending; expect(reply.status()).toBe(200)
  await expect(page.getByTestId('digitizer-image')).toBeVisible()
  return reply.json()
}
async function calibrate(page) {
  const values = { 'x-pixel-1': 100, 'x-pixel-2': 900, 'x-value-1': 1, 'x-value-2': 100,
    'y-pixel-1': 500, 'y-pixel-2': 100, 'y-value-1': 0, 'y-value-2': 20, 'x-unit': 'A', 'y-unit': 'V', source: 'Independent fixture, Figure 1. Test temperature and gate voltage unknown.' }
  for (const [key, value] of Object.entries(values)) await page.getByTestId(`digitizer-${key}`).fill(String(value))
  await page.getByTestId('digitizer-x-scale').selectOption('log')
}
async function clickPixel(page, x, y) {
  const svg = page.getByTestId('digitizer-image')
  await svg.scrollIntoViewIfNeeded()
  const screen = await svg.evaluate((svg, { x, y }) => {
    const result = new DOMPoint(x, y).matrixTransform(svg.getScreenCTM())
    return { x: result.x, y: result.y }
  }, { x, y })
  await page.mouse.click(screen.x, screen.y)
}
async function map(page, status = 200) {
  const pending = page.waitForResponse(r => r.url().endsWith('/api/digitizer/transform'))
  await page.getByTestId('digitizer-map').click()
  const reply = await pending; expect(reply.status()).toBe(status)
  return reply.json()
}
async function download(page, format) {
  const pending = page.waitForEvent('download')
  await page.getByTestId(`digitizer-${format}`).click()
  return fs.readFile(await (await pending).path(), 'utf8')
}

test('known natural pixels retain Python values through responsive scale, DPR, letterboxing and exports', async ({ browser }, testInfo) => {
  const context = await browser.newContext({ viewport: { width: 1280, height: 1000 }, deviceScaleFactor: 2 })
  const page = await context.newPage(), errors = []
  page.on('pageerror', error => errors.push(error.message))
  await page.goto(`${origin}/index.html?view=digitizer`)
  const bytes = await imageBytes(page)
  const metadata = await upload(page, bytes)
  expect((metadata.image ?? metadata).sha256).toBe(createHash('sha256').update(bytes).digest('hex'))
  await calibrate(page)
  // Select an axis anchor through the same natural-coordinate click path.
  await page.getByTestId('digitizer-x-anchor-1').click(); await clickPixel(page, 100, 250)
  // Browser pointer and transform matrices retain small binary roundoff.
  // This tolerance is less than 0.001 natural image pixel.
  expect(Number(await page.getByTestId('digitizer-x-pixel-1').inputValue())).toBeCloseTo(100, 3)
  await page.getByTestId('digitizer-point-mode').click()
  await clickPixel(page, 500, 300)
  const first = await map(page)
  expect(first.points[0].pixel_x).toBeCloseTo(500, 3); expect(first.points[0].pixel_y).toBeCloseTo(300, 3)
  expect(first.points[0].x).toBeCloseTo(10, 4); expect(first.points[0].y).toBeCloseTo(10, 4)
  expect(await download(page, 'csv')).toBe(first.csv)
  expect(JSON.parse(await download(page, 'json'))).toEqual(first)
  await page.screenshot({ path: testInfo.outputPath('digitizer-desktop.png'), fullPage: true })
  await page.setViewportSize({ width: 390, height: 900 })
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390)
  // Force letterboxing and a CSS transform. Inverse SVG CTM must still return natural pixels.
  await page.getByTestId('digitizer-image').evaluate(svg => {
    svg.style.height = '300px'; svg.style.aspectRatio = 'auto'; svg.style.transform = 'scale(.8)'; svg.style.transformOrigin = 'center'
  })
  await clickPixel(page, 500, 300)
  const resized = await map(page)
  expect(resized.points).toHaveLength(2)
  expect(resized.points[1].pixel_x).toBeCloseTo(500, 3); expect(resized.points[1].pixel_y).toBeCloseTo(300, 3)
  expect(resized.points[1].x).toBeCloseTo(10, 4); expect(resized.points[1].y).toBeCloseTo(10, 4)
  const csv = await download(page, 'csv'), json = JSON.parse(await download(page, 'json'))
  expect(csv).toBe(resized.csv); expect(json).toEqual(resized)
  await page.screenshot({ path: testInfo.outputPath('digitizer-mobile-scaled.png'), fullPage: true })
  // Change X only. Python must preserve Y and reject stale exports until recalculation.
  await page.getByTestId('digitizer-x-value-2').fill('10000')
  await expect(page.getByTestId('digitizer-csv')).toBeDisabled()
  const changed = await map(page)
  expect(changed.points[0].x).toBeCloseTo(100, 3)
  expect(changed.points[0].y).toBe(first.points[0].y)
  expect(errors).toEqual([])
  await context.close()
})

test('reversed/log axes, explicit extrapolation, point undo and image identities remain visible', async ({ page }) => {
  await page.goto(`${origin}/index.html?view=digitizer`)
  const bytes = await imageBytes(page)
  await upload(page, bytes); await calibrate(page)
  await page.getByTestId('digitizer-y-scale').selectOption('log')
  await page.getByTestId('digitizer-y-value-1').fill('1')
  await page.getByTestId('digitizer-y-value-2').fill('100')
  await clickPixel(page, 500, 300)
  let result = await map(page)
  expect(result.points[0].x).toBeCloseTo(10, 9); expect(result.points[0].y).toBeCloseTo(10, 9)
  await page.getByTestId('digitizer-x-scale').selectOption('linear')
  await page.getByTestId('digitizer-x-value-1').fill('100')
  await page.getByTestId('digitizer-x-value-2').fill('0')
  result = await map(page); expect(result.points[0].x).toBeCloseTo(50, 9)
  await page.getByTestId('digitizer-y-scale').selectOption('linear')
  await page.getByTestId('digitizer-y-value-1').fill('0')
  await page.getByTestId('digitizer-y-value-2').fill('20')
  result = await map(page); expect(result.points[0].x).toBeCloseTo(50, 9); expect(result.points[0].y).toBeCloseTo(10, 9)
  await page.getByRole('spinbutton', { name: 'Point 1 X pixel', exact: true }).fill('50')
  await page.getByRole('spinbutton', { name: 'Point 1 X pixel', exact: true }).blur()
  await map(page, 422); await expect(page.getByRole('alert')).toBeVisible()
  await page.getByTestId('digitizer-extrapolate').check()
  result = await map(page); expect(result.points[0].x).toBeCloseTo(106.25, 9)
  await page.getByTestId('digitizer-undo').click()
  expect(Number(await page.getByRole('spinbutton', { name: 'Point 1 X pixel', exact: true }).inputValue())).toBeCloseTo(500, 8)
  await page.getByRole('button', { name: 'Remove point 1', exact: true }).click()
  await expect(page.getByTestId('digitizer-point-row')).toHaveCount(0)
  await page.getByTestId('digitizer-undo').click()
  await upload(page, bytes, 'identical-renamed.png')
  await expect(page.getByTestId('digitizer-point-row')).toHaveCount(1)
  await expect(page.getByText(/image bytes are identical/)).toBeVisible()
  const changed = await imageBytes(page, '#dddddd')
  await upload(page, changed, 'identical-renamed.png')
  await expect(page.getByTestId('digitizer-point-row')).toHaveCount(0)
  await expect(page.getByTestId('digitizer-x-pixel-1')).toHaveValue('')
  await expect(page.getByTestId('digitizer-csv')).toBeDisabled()
})

test('held Python and image responses cannot restore an earlier selection', async ({ page }) => {
  await page.goto(`${origin}/index.html?view=digitizer`)
  const original = await imageBytes(page), replacement = await imageBytes(page, '#eeeeee')
  await upload(page, original); await calibrate(page); await clickPixel(page, 500, 300)
  let releaseTransform, transformStarted
  const heldTransform = new Promise(resolve => { releaseTransform = resolve })
  const startedTransform = new Promise(resolve => { transformStarted = resolve })
  await page.route('**/api/digitizer/transform', async route => {
    const response = await route.fetch(); transformStarted(); await heldTransform; await route.fulfill({ response })
  })
  await page.getByTestId('digitizer-map').click(); await startedTransform
  await page.getByTestId('digitizer-source').fill('Changed source, conditions unknown')
  const oldTransform = page.waitForResponse(r => r.url().endsWith('/api/digitizer/transform'))
  releaseTransform(); await (await oldTransform).finished()
  await expect(page.getByTestId('digitizer-results')).toHaveCount(0)
  await expect(page.getByTestId('digitizer-csv')).toBeDisabled()
  await page.unroute('**/api/digitizer/transform')
  let releaseImage, imageStarted, count = 0
  const heldImage = new Promise(resolve => { releaseImage = resolve })
  const startedImage = new Promise(resolve => { imageStarted = resolve })
  await page.route('**/api/digitizer/image', async route => {
    const response = await route.fetch()
    if (++count === 1) { imageStarted(); await heldImage }
    await route.fulfill({ response })
  })
  await page.getByTestId('digitizer-file').setInputFiles({ name: 'old-held.png', mimeType: 'image/png', buffer: original })
  await startedImage
  await upload(page, replacement, 'replacement.png')
  const expected = createHash('sha256').update(replacement).digest('hex')
  await expect(page.getByTestId('digitizer-hash')).toHaveText(expected)
  const oldReply = page.waitForResponse(r => r.url().endsWith('/api/digitizer/image') && r.request().postDataJSON().filename === 'old-held.png')
  releaseImage(); await (await oldReply).finished()
  await expect(page.getByTestId('digitizer-hash')).toHaveText(expected)
  await expect(page.getByTestId('digitizer-point-row')).toHaveCount(0)
  await expect(page.getByTestId('digitizer-x-pixel-1')).toHaveValue('')
  await expect(page.getByTestId('digitizer-csv')).toBeDisabled()
})
