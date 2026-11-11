import { createHash } from 'node:crypto'
import { readFile } from 'node:fs/promises'
import { test, expect } from '@playwright/test'

const start = 'http://localhost:8002/index.html?view=digitizer'
const field = (page, name) => page.getByTestId(`digitizer-${name}`)

async function figure(page, color) {
  const encoded = await page.evaluate(fill => {
    const canvas = document.createElement('canvas')
    canvas.width = 1000
    canvas.height = 600
    const paint = canvas.getContext('2d')
    paint.fillStyle = fill
    paint.fillRect(0, 0, 1000, 600)
    paint.strokeStyle = '#19435c'
    paint.strokeRect(100, 100, 800, 400)
    paint.fillRect(496, 296, 8, 8)
    paint.fillStyle = '#e33e22'
    paint.fillRect(497, 297, 6, 6)
    return canvas.toDataURL('image/png').split(',')[1]
  }, color)
  return Buffer.from(encoded, 'base64')
}

async function upload(page, bytes, name = 'independent.png') {
  const hash = createHash('sha256').update(bytes).digest('hex')
  const received = page.waitForResponse(r => r.url().endsWith('/api/digitizer/image'))
  await field(page, 'file').setInputFiles({ name, mimeType: 'image/png', buffer: bytes })
  const reply = await received
  expect(reply.status()).toBe(200)
  expect(await reply.json()).toMatchObject({ sha256: hash, width: 1000, height: 600, byte_size: bytes.length })
  await expect(field(page, 'hash')).toHaveText(hash)
  return hash
}

async function calibrate(page) {
  for (const [name, value] of Object.entries({
    'x-pixel-1': 100, 'x-value-1': 1, 'x-pixel-2': 900, 'x-value-2': 100, 'x-unit': 'nF',
    'y-pixel-1': 500, 'y-value-1': 0, 'y-pixel-2': 100, 'y-value-2': 20, 'y-unit': 'V',
    source: 'Independent synthetic raster. Temperature unknown.',
  })) await field(page, name).fill(String(value))
  await field(page, 'x-scale').selectOption('log')
}

async function mapped(page, status = 200) {
  const received = page.waitForResponse(r => r.url().endsWith('/api/digitizer/transform'))
  await field(page, 'map').click()
  const reply = await received
  expect(reply.status()).toBe(status)
  return reply.json()
}

async function exported(page, kind) {
  const event = page.waitForEvent('download')
  await field(page, kind).click()
  return readFile(await (await event).path(), 'utf8')
}

async function placeFigure(page, width, height) {
  const svg = field(page, 'image')
  await svg.evaluate((element, size) => {
    // Offset by a fraction so integer MouseEvent coordinates cannot pass this check.
    Object.assign(element.style, {
      width: size.width + 'px', height: size.height + 'px', aspectRatio: 'auto',
      transform: 'scale(.8)', transformOrigin: 'top left',
    })
    element.scrollIntoView({ block: 'center' })
    const box = element.getBoundingClientRect()
    element.style.transform = `translate(${Math.round(box.left) - box.left + .375}px, ${Math.round(box.top) - box.top + .625}px) scale(.8)`
  }, { width, height })
  return svg.boundingBox()
}

for (const width of [390, 1280]) {
  test(`maps hardware mouse coordinates at viewport ${width} and DPR 2`, async ({ browser }, info) => {
    const context = await browser.newContext({ viewport: { width, height: 900 }, deviceScaleFactor: 2, hasTouch: true })
    const page = await context.newPage()
    await page.goto(start)
    const bytes = await figure(page, '#eeeeee')
    const hash = await upload(page, bytes)
    await calibrate(page)
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(width)
    const box = await placeFigure(page, width === 390 ? 250 : 750, width === 390 ? 250 : 500)
    // Derive positions from the visible CSS rectangle and the aspect ratio alone.
    const scale = Math.min(box.width / 1000, box.height / 600)
    const left = box.x + (box.width - scale * 1000) / 2
    const top = box.y + (box.height - scale * 600) / 2
    await page.mouse.click(left + 500 * scale, box.y + 2)
    await expect(field(page, 'point-row')).toHaveCount(0)
    const center = { x: left + 500 * scale, y: top + 300 * scale }
    await page.mouse.move(center.x - 50, center.y)
    await page.mouse.down()
    await page.mouse.move(center.x + 50, center.y, { steps: 5 })
    await page.mouse.up()
    await expect(field(page, 'point-row')).toHaveCount(0)
    const touch = await context.newCDPSession(page)
    await touch.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [center] })
    await touch.send('Input.dispatchTouchEvent', { type: 'touchCancel', touchPoints: [] })
    await expect(field(page, 'point-row')).toHaveCount(0)
    for (const [x, y] of [[500, 300], [700, 200]]) {
      await page.mouse.click(left + x * scale, top + y * scale)
    }
    const result = await mapped(page)
    expect(result.calibration.image).toEqual({ sha256: hash, width: 1000, height: 600 })
    expect(result.points).toHaveLength(2)
    for (const [i, [px, py, x, y]] of [[500, 300, 10, 10], [700, 200, 10 ** 1.5, 15]].entries()) {
      expect(result.points[i].pixel_x).toBeCloseTo(px, 4)
      expect(result.points[i].pixel_y).toBeCloseTo(py, 4)
      expect(result.points[i].x).toBeCloseTo(x, 4)
      expect(result.points[i].y).toBeCloseTo(y, 4)
    }
    expect(result.units).toEqual({ x: 'nF', y: 'V' })
    expect(await exported(page, 'csv')).toBe(result.csv)
    expect(JSON.parse(await exported(page, 'json'))).toEqual(result)
    await page.screenshot({ path: info.outputPath(`digitizer-${width}.png`), fullPage: true })
    await field(page, 'image').scrollIntoViewIfNeeded()
    const svgBox = await field(page, 'image').boundingBox()
    const finger = { x: svgBox.x + svgBox.width / 2, y: svgBox.y + svgBox.height / 2 }
    await touch.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [finger] })
    await touch.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [{ x: finger.x, y: finger.y - 80 }] })
    await touch.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] })
    await expect(field(page, 'point-row')).toHaveCount(2)
    await context.close()
  })
}

test('keeps ordered point edits and rejects stale file and transform replies', async ({ page }) => {
  await page.goto(start)
  const first = await figure(page, 'white')
  await upload(page, first)
  await calibrate(page)
  for (const [x, y] of [[500, 300], [900, 100]]) {
    await field(page, 'manual-x').fill(String(x))
    await field(page, 'manual-y').fill(String(y))
    await page.getByRole('button', { name: 'Add pixel point' }).click()
  }
  const xInput = page.getByRole('spinbutton', { name: 'Point 1 X pixel', exact: true })
  await xInput.fill('50')
  await xInput.blur()
  await mapped(page, 422)
  await expect(page.getByRole('alert')).toContainText('extrapolation')
  await field(page, 'extrapolate').check()
  const outside = await mapped(page)
  expect(outside.points[0].x).toBeCloseTo(10 ** -.125, 10)
  await field(page, 'undo').click()
  await expect(xInput).toHaveValue('500')
  await page.getByRole('button', { name: 'Remove point 1' }).click()
  await expect(xInput).toHaveValue('900')
  await field(page, 'undo').click()
  await field(page, 'x-value-1').fill('100')
  await field(page, 'x-value-2').fill('1')
  const inverted = await mapped(page)
  expect(inverted.points.map(p => [p.x, p.y])).toEqual([[10, 10], [1, 20]])
  await upload(page, first, 'same.png')
  await expect(field(page, 'point-row')).toHaveCount(2)
  await expect(field(page, 'csv')).toBeDisabled()
  await expect(page.getByText(/image bytes are identical/i)).toBeVisible()
  const release = Promise.withResolvers()
  const held = Promise.withResolvers()
  await page.route('**/api/digitizer/transform', async route => {
    const reply = await route.fetch()
    expect(reply.status()).toBe(200)
    held.resolve()
    await release.promise
    await route.fulfill({ response: reply })
  }, { times: 1 })
  await field(page, 'map').click()
  await held.promise
  const second = await figure(page, '#222222')
  await upload(page, second, 'same.png')
  const late = page.waitForResponse(r => r.url().endsWith('/api/digitizer/transform'))
  release.resolve()
  await late
  await expect(field(page, 'point-row')).toHaveCount(0)
  await expect(field(page, 'x-pixel-1')).toHaveValue('')
  await expect(field(page, 'csv')).toBeDisabled()
  const releaseImage = Promise.withResolvers()
  const heldImage = Promise.withResolvers()
  await page.route('**/api/digitizer/image', async route => {
    const reply = await route.fetch()
    heldImage.resolve()
    await releaseImage.promise
    await route.fulfill({ response: reply })
  }, { times: 1 })
  await field(page, 'file').setInputFiles({ name: 'slow.png', mimeType: 'image/png', buffer: first })
  await heldImage.promise
  const finalHash = await upload(page, second, 'current.png')
  const delayed = page.waitForResponse(r => r.url().endsWith('/api/digitizer/image'))
  releaseImage.resolve()
  await delayed
  await expect(field(page, 'hash')).toHaveText(finalHash)
  await expect(page.getByText(/current\.png · 1000/)).toBeVisible()
})
