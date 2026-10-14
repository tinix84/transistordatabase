import { test, expect } from '@playwright/test'

const DEVICE_NAME = 'E2E Curve #Exact ?'
const CURVES_PATH = `/api/transistors/${encodeURIComponent(DEVICE_NAME)}/curves`
const CHANNEL_CURVE_PATH = `${CURVES_PATH}/switch/channel/0`

function fixtureDevice() {
  return {
    metadata: {
      name: DEVICE_NAME,
      type: 'MOSFET',
      manufacturer: 'Fixture Semiconductor',
      housing_type: 'TO263',
      author: 'E2E Fixture',
      comment: 'Isolated curve workflow fixture'
    },
    electrical: { v_abs_max: 650, i_abs_max: 120, i_cont: 60, t_j_max: 175 },
    thermal: {
      r_th_cs: 0.24,
      r_th_switch_cs: 0.12,
      r_th_diode_cs: 0.18,
      housing_area: 0.0012,
      cooling_area: 0.0008
    }
  }
}

function curveResponse() {
  return {
    switch: {
      channel_data: [{ _edit_token: 'a'.repeat(64), t_j: 25, v_g: 15, graph_v_i: [[0, 10, 20], [0, 2, 4]] }],
      e_on_data: [{
        dataset_type: 'graph_i_e', t_j: 25, v_supply: 400, v_g: 15, r_g: 2.5,
        graph_i_e: [[0, 10, 20], [0, 0.001, 0.002]], graph_r_e: null
      }],
      e_off_data: [],
      gate_charge_curves: [{
        v_supply: 400, t_j: 25, i_channel: 10, i_g: null,
        graph_q_v: [[0, 25e-9, 50e-9], [0, 8, 15]]
      }],
      soa: [{ t_c: 25, time_pulse: 0.01, graph_i_v: [[0, 100, 600], [100, 20, 1]] }]
    },
    diode: {
      channel_data: [],
      e_rr_data: []
    },
    capacitances: {
      c_oss: [{ t_j: 25, graph_v_c: [[0, 50, 100], [0.00001, 0.000002, 0.000001]] }],
      c_iss: [],
      c_rss: []
    }
  }
}

test('prefills, updates, views, closes, and deletes the exact device curve', async ({ page }) => {
  const curves = curveResponse()
  const requests = []
  const unmatchedRequests = []
  const expectedDeletePath = CHANNEL_CURVE_PATH
  const expectedUpdatePath = `${CURVES_PATH}/switch/channel/0`

  await page.route('**/api/**', async route => {
    const request = route.request()
    const url = new URL(request.url())
    const method = request.method()
    const path = url.pathname
    requests.push({ method, path })

    if (method === 'GET' && path === '/api/transistors') {
      await route.fulfill({ status: 200, json: [fixtureDevice()] })
      return
    }

    if (method === 'GET' && path === CURVES_PATH) {
      await route.fulfill({ status: 200, json: curves })
      return
    }

    if (method === 'PUT' && path === expectedUpdatePath) {
      const body = request.postDataJSON()
      requests.at(-1).body = body
      curves.switch.channel_data[0] = {
        ...curves.switch.channel_data[0],
        _edit_token: 'b'.repeat(64),
        graph_v_i: [[0, 10, 20], [0, 3, 4]]
      }
      await route.fulfill({ status: 200, json: { message: 'Curve updated', updated_index: 0, curve_count: 1 } })
      return
    }

    if (method === 'DELETE' && path === expectedDeletePath) {
      curves.switch.channel_data.splice(0, 1)
      await route.fulfill({ status: 200, json: { message: 'Curve deleted' } })
      return
    }

    unmatchedRequests.push(`${method} ${path}`)
    await route.fulfill({ status: 501, json: { detail: `Unmocked API request: ${method} ${path}` } })
  })

  await page.goto('/')
  const row = page.getByTestId('transistor-row').filter({ hasText: DEVICE_NAME })
  await row.getByRole('button', { name: 'Edit' }).click()
  await page.getByRole('button', { name: 'Channel Curves' }).click()

  const curveCard = page.locator('.curve-tab-content:visible .curve-card').filter({ hasText: 'Curve #1' })
  await curveCard.locator('[title="Edit"]').click()
  const editor = page.locator('.curve-editor')
  await expect(editor.getByRole('heading', { name: 'Edit Channel Characteristic' })).toBeVisible()
  const secondPoint = editor.locator('.manual-entry-container .data-point-row').nth(1)
  await expect(secondPoint.locator('input').nth(1)).toHaveValue('2')
  await secondPoint.locator('input').nth(1).fill('3')
  await editor.getByRole('button', { name: 'Update Curve' }).click()
  await expect(editor).toHaveCount(0)
  await expect(page.locator('.curve-tab-content:visible .curve-card')).toContainText('Curve #1')

  const updateRequest = requests.find(request => request.method === 'PUT')
  expect(updateRequest).toEqual({
    method: 'PUT',
    path: expectedUpdatePath,
    body: {
      expected_count: 1,
      expected_token: 'a'.repeat(64),
      curve: { graph_v_i: [[0, 10, 20], [0, 3, 4]] }
    }
  })

  await page.locator('.curve-tab-content:visible .curve-card').locator('[title="Edit"]').click()
  const refreshedEditor = page.locator('.curve-editor')
  await expect(refreshedEditor.locator('.manual-entry-container .data-point-row').nth(1).locator('input').nth(1)).toHaveValue('3')
  await refreshedEditor.getByRole('button', { name: 'Cancel' }).click()

  const refreshedCurveCard = page.locator('.curve-tab-content:visible .curve-card').filter({ hasText: 'Curve #1' })
  await refreshedCurveCard.locator('[title="View Curve"]').click()
  const viewer = page.getByTestId('curve-viewer')
  await expect(viewer).toContainText('Switch channel curve #1')
  await expect(viewer.locator('canvas')).toBeVisible()

  await viewer.getByRole('button', { name: 'Close curve view' }).click()
  await expect(viewer).toHaveCount(0)

  await refreshedCurveCard.locator('[title="Delete"]').click()
  await page.locator('.modal-actions').getByRole('button', { name: 'Delete' }).click()
  await expect(page.locator('.curve-tab-content:visible .curve-card')).toHaveCount(0)

  expect(requests.filter(request => request.path === CURVES_PATH)).toHaveLength(3)
  expect(requests.filter(request => request.method === 'DELETE')).toEqual([
    { method: 'DELETE', path: expectedDeletePath }
  ])
  expect(unmatchedRequests).toEqual([])
})

test('keeps the editor open and does not refresh after a stale edit response', async ({ page }) => {
  const curves = curveResponse()
  const requests = []
  await page.route('**/api/**', async route => {
    const request = route.request()
    const url = new URL(request.url())
    requests.push({ method: request.method(), path: url.pathname })

    if (request.method() === 'GET' && url.pathname === '/api/transistors') {
      await route.fulfill({ status: 200, json: [fixtureDevice()] })
      return
    }
    if (request.method() === 'GET' && url.pathname === CURVES_PATH) {
      await route.fulfill({ status: 200, json: curves })
      return
    }
    if (request.method() === 'PUT' && url.pathname === CHANNEL_CURVE_PATH) {
      await route.fulfill({ status: 409, json: { detail: 'This curve changed since it was loaded.' } })
      return
    }
    await route.fulfill({ status: 501, json: { detail: `Unmocked API request: ${request.method()} ${url.pathname}` } })
  })

  await page.goto('/')
  const row = page.getByTestId('transistor-row').filter({ hasText: DEVICE_NAME })
  await row.getByRole('button', { name: 'Edit' }).click()
  await page.getByRole('button', { name: 'Channel Curves' }).click()
  await page.locator('.curve-tab-content:visible .curve-card [title="Edit"]').click()
  const editor = page.locator('.curve-editor')
  await editor.locator('.manual-entry-container .data-point-row').nth(1).locator('input').nth(1).fill('3')
  await editor.getByRole('button', { name: 'Update Curve' }).click()

  await expect(editor).toBeVisible()
  await expect(page.locator('.curve-notification.error')).toContainText('This curve changed since it was loaded.')
  expect(requests.filter(request => request.method === 'GET' && request.path === CURVES_PATH)).toHaveLength(1)
})

test('switches the open editor to the selected curve and uses its index and token', async ({ page }) => {
  const curves = curveResponse()
  curves.switch.channel_data = [
    { _edit_token: 'a'.repeat(64), t_j: 25, v_g: 15, graph_v_i: [[0, 10, 20], [0, 2, 4]] },
    { _edit_token: 'd'.repeat(64), t_j: 125, v_g: 10, graph_v_i: [[0, 15, 30], [0, 7, 12]] }
  ]
  const requests = []
  const updatePath = `${CURVES_PATH}/switch/channel/1`

  await page.route('**/api/**', async route => {
    const request = route.request()
    const url = new URL(request.url())
    requests.push({ method: request.method(), path: url.pathname })

    if (request.method() === 'GET' && url.pathname === '/api/transistors') {
      await route.fulfill({ status: 200, json: [fixtureDevice()] })
      return
    }
    if (request.method() === 'GET' && url.pathname === CURVES_PATH) {
      await route.fulfill({ status: 200, json: curves })
      return
    }
    if (request.method() === 'PUT' && url.pathname === updatePath) {
      requests.at(-1).body = request.postDataJSON()
      await route.fulfill({ status: 200, json: { message: 'Curve updated', updated_index: 1, curve_count: 2 } })
      return
    }
    await route.fulfill({ status: 501, json: { detail: `Unmocked API request: ${request.method()} ${url.pathname}` } })
  })

  await page.goto('/')
  const row = page.getByTestId('transistor-row').filter({ hasText: DEVICE_NAME })
  await row.getByRole('button', { name: 'Edit' }).click()
  await page.getByRole('button', { name: 'Channel Curves' }).click()
  const cards = page.locator('.curve-tab-content:visible .curve-card')
  const editor = page.locator('.curve-editor')

  await cards.nth(0).locator('[title="Edit"]').click()
  await expect(editor.locator('input').nth(0)).toHaveValue('25')
  await expect(editor.locator('.manual-entry-container .data-point-row').nth(1).locator('input').nth(1)).toHaveValue('2')

  await cards.nth(1).locator('[title="Edit"]').click()
  await expect(editor.locator('input').nth(0)).toHaveValue('125')
  const secondPoint = editor.locator('.manual-entry-container .data-point-row').nth(1)
  await expect(secondPoint.locator('input').nth(1)).toHaveValue('7')
  await secondPoint.locator('input').nth(1).fill('8')
  await editor.getByRole('button', { name: 'Update Curve' }).click()

  expect(requests.filter(request => request.method === 'PUT')).toEqual([{
    method: 'PUT',
    path: updatePath,
    body: {
      expected_count: 2,
      expected_token: 'd'.repeat(64),
      curve: { graph_v_i: [[0, 15, 30], [0, 8, 12]] }
    }
  }])
})

test('keeps the edited route when a selector changes and resets edit state for Add', async ({ page }) => {
  const curves = curveResponse()
  const sameChannelCurve = {
    _edit_token: 'e'.repeat(64), t_j: 125, v_g: 15, graph_v_i: [[0, 10, 20], [0, 2, 4]]
  }
  curves.switch.channel_data = [{ ...sameChannelCurve }]
  curves.diode.channel_data = [{ ...sameChannelCurve }]
  const requests = []
  const expectedUpdatePath = `${CURVES_PATH}/switch/channel/0`

  await page.route('**/api/**', async route => {
    const request = route.request()
    const url = new URL(request.url())
    requests.push({ method: request.method(), path: url.pathname })

    if (request.method() === 'GET' && url.pathname === '/api/transistors') {
      await route.fulfill({ status: 200, json: [fixtureDevice()] })
      return
    }
    if (request.method() === 'GET' && url.pathname === CURVES_PATH) {
      await route.fulfill({ status: 200, json: curves })
      return
    }
    if (request.method() === 'PUT' && url.pathname === expectedUpdatePath) {
      requests.at(-1).body = request.postDataJSON()
      await route.fulfill({ status: 200, json: { message: 'Curve updated', updated_index: 0, curve_count: 1 } })
      return
    }
    await route.fulfill({ status: 501, json: { detail: `Unmocked API request: ${request.method()} ${url.pathname}` } })
  })

  await page.goto('/')
  const row = page.getByTestId('transistor-row').filter({ hasText: DEVICE_NAME })
  await row.getByRole('button', { name: 'Edit' }).click()
  await page.getByRole('button', { name: 'Channel Curves' }).click()
  const editor = page.locator('.curve-editor')
  await page.locator('.curve-tab-content:visible .curve-card [title="Edit"]').click()

  await page.getByRole('button', { name: 'Diode' }).click()
  await expect(editor).toBeVisible()
  await expect(editor.locator('input').nth(0)).toHaveValue('125')
  await editor.getByRole('button', { name: 'Update Curve' }).click()
  expect(requests.filter(request => request.method === 'PUT')).toEqual([{
    method: 'PUT',
    path: expectedUpdatePath,
    body: {
      expected_count: 1,
      expected_token: 'e'.repeat(64),
      curve: { graph_v_i: [[0, 10, 20], [0, 2, 4]] }
    }
  }])

  await page.getByRole('button', { name: 'Switch', exact: true }).click()
  await page.locator('.curve-tab-content:visible .curve-card [title="Edit"]').click()
  await expect(editor.locator('input').nth(0)).toHaveValue('125')
  await page.locator('.curve-tab-content:visible .list-header button').click()
  await expect(editor.getByRole('heading', { name: 'Add Channel Characteristic' })).toBeVisible()
  await expect(editor.locator('input').nth(0)).toHaveValue('25')
  await expect(editor.locator('.upload-container')).toBeVisible()
  await expect(editor.locator('.manual-entry-container')).toHaveCount(0)
})
