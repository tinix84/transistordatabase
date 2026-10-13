import { test, expect } from '@playwright/test'

const DEVICE_NAME = 'E2E Curve #Exact ?'
const CURVES_PATH = `/api/transistors/${encodeURIComponent(DEVICE_NAME)}/curves`
const CHANNEL_DELETE_PATH = `${CURVES_PATH}/switch/channel/0`

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
      channel_data: [{ t_j: 25, v_g: 15, graph_v_i: [[0, 10, 20], [0, 2, 4]] }],
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

test('loads, views, closes, and deletes the exact device curve', async ({ page }) => {
  const curves = curveResponse()
  const requests = []
  const unmatchedRequests = []
  const expectedDeletePath = CHANNEL_DELETE_PATH

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
  await expect(curveCard.locator('[title="Edit"]')).toHaveCount(0)
  await curveCard.locator('[title="View Curve"]').click()
  const viewer = page.getByTestId('curve-viewer')
  await expect(viewer).toContainText('Switch channel curve #1')
  await expect(viewer.locator('canvas')).toBeVisible()

  await viewer.getByRole('button', { name: 'Close curve view' }).click()
  await expect(viewer).toHaveCount(0)

  await curveCard.locator('[title="Delete"]').click()
  await page.locator('.modal-actions').getByRole('button', { name: 'Delete' }).click()
  await expect(page.locator('.curve-tab-content:visible .curve-card')).toHaveCount(0)

  expect(requests.filter(request => request.path === CURVES_PATH)).toHaveLength(2)
  expect(requests.filter(request => request.method === 'DELETE')).toEqual([
    { method: 'DELETE', path: expectedDeletePath }
  ])
  expect(unmatchedRequests).toEqual([])
})
