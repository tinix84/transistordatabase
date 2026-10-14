import { test, expect } from '@playwright/test'

const DEVICE_NAME = 'E2E DPT Energy Device'
const DPT_DATA_PATH = `/api/transistors/${encodeURIComponent(DEVICE_NAME)}/dpt_data`
const PREVIEW_PATH = `/api/transistors/${encodeURIComponent(DEVICE_NAME)}/dpt_energy/preview`
const SAVE_PATH = `/api/transistors/${encodeURIComponent(DEVICE_NAME)}/dpt_energy/save`

function fixtureDevice() {
  return {
    metadata: {
      name: DEVICE_NAME,
      type: 'MOSFET',
      manufacturer: 'Fixture Semiconductor',
      author: 'DPT energy browser fixture'
    }
  }
}

function dptData() {
  return {
    has_e_on: false,
    has_e_off: false,
    has_e_rr: false,
    metadata: null,
    raw_measurement_data: [{
      metadata: {
        dataset_type: 'dpt_u_i',
        comment: 'Two stored turn-on acquisitions',
        t_j: 25,
        v_supply: 400,
        v_g: 15,
        v_g_off: -5,
        r_g: 10,
        r_g_off: 12
      },
      dpt_on_vds: [
        [[0, 100], [1e-6, 80], [2e-6, 60]],
        [[0, 110], [1e-6, 90], [2e-6, 70]]
      ],
      dpt_on_id: [
        [[0, 0], [1e-6, 10], [2e-6, 20]],
        [[0, 0], [1e-6, 20], [2e-6, 40]]
      ],
      dpt_off_vds: [],
      dpt_off_id: []
    }]
  }
}

test('reviews, explicitly confirms, and saves an exact-window E_on curve', async ({ page }) => {
  const requests = []
  let dptReadCount = 0
  await page.route('**/api/**', async route => {
    const request = route.request()
    const url = new URL(request.url())
    const path = url.pathname
    requests.push({ method: request.method(), path })

    if (request.method() === 'GET' && path === '/api/transistors') {
      await route.fulfill({ status: 200, json: [fixtureDevice()] })
      return
    }
    if (request.method() === 'GET' && path === DPT_DATA_PATH) {
      dptReadCount += 1
      await route.fulfill({ status: 200, json: dptData() })
      return
    }
    if (request.method() === 'POST' && path === PREVIEW_PATH) {
      requests.at(-1).body = request.postDataJSON()
      await route.fulfill({
        status: 200,
        json: {
          loss_type: 'e_on',
          event: 'turn_on',
          points: [
            {
              measurement_index: 0, acquisition_index: 0, i_x: 10,
              energy_j: 0.0014, start_s: 0, end_s: 2e-6,
              sample_count: 3, vds_sign: 1, id_sign: -1
            },
            {
              measurement_index: 0, acquisition_index: 1, i_x: 20,
              energy_j: 0.0028, start_s: 0, end_s: 2e-6,
              sample_count: 3, vds_sign: -1, id_sign: 1
            }
          ],
          graph_i_e: [[10, 20], [0.0014, 0.0028]],
          conditions: {
            t_j: 25, v_supply: 400, v_g: 15, v_g_off: -5,
            r_g: 10, r_g_off: 12, load_inductance: 0.00000075,
            commutation_inductance: 0.000000125,
            commutation_device: 'Fixture diode',
            measurement_testbench: 'Fixture bench'
          },
          integration: {
            quantity: 'raw-window switching energy',
            formula: 'E = sum(0.5 * (p[n] + p[n+1]) * (t[n+1] - t[n]))',
            power_formula: 'p(t) = (Vds_sign * Vds(t)) * (Id_sign * Id(t))',
            interval: 'exact stored first-to-last event sample time',
            boundary_detection: 'none; user-selected imported windows are used as stored'
          },
          source_token: 'a'.repeat(64),
          request_token: 'b'.repeat(64),
          save_action: 'add',
          existing_curve_index: null,
          target_curve_count: 0,
          target_edit_token: null
        }
      })
      return
    }
    if (request.method() === 'POST' && path === SAVE_PATH) {
      requests.at(-1).body = request.postDataJSON()
      await route.fulfill({
        status: 200,
        json: {
          success: true, loss_type: 'e_on', action: 'added',
          curve_index: 0, curve_count: 1, point_count: 2,
          graph_i_e: [[10, 20], [0.0014, 0.0028]]
        }
      })
      return
    }

    await route.fulfill({
      status: 501,
      json: { detail: `Unmocked API request: ${request.method()} ${path}` }
    })
  })

  await page.goto('/')
  await page.getByRole('button', { name: 'More Tools' }).click()
  await page.getByRole('button', { name: 'DPT Manager' }).click()
  await expect(page.getByRole('heading', { name: 'Double Pulse Test (DPT) Manager' })).toBeVisible()
  await page.locator('.transistor-selector select').selectOption(DEVICE_NAME)

  const energy = page.getByTestId('dpt-energy-derivation')
  for (const index of [0, 1]) {
    await energy.getByTestId(`dpt-energy-record-${index}`).selectOption('0')
    await energy.getByTestId(`dpt-energy-acquisition-${index}`).selectOption(String(index))
    await energy.getByTestId(`dpt-energy-current-${index}`).fill(String((index + 1) * 10))
  }
  await energy.getByTestId('dpt-energy-vds-sign-0').selectOption('1')
  await energy.getByTestId('dpt-energy-id-sign-0').selectOption('-1')
  await energy.getByTestId('dpt-energy-vds-sign-1').selectOption('-1')
  await energy.getByTestId('dpt-energy-id-sign-1').selectOption('1')
  await energy.getByTestId('dpt-energy-preview').click()

  await expect(energy.getByTestId('dpt-energy-preview-result')).toContainText('Create a new curve')
  await expect(energy.getByTestId('dpt-energy-preview-result')).toContainText('0.0014')
  await expect(energy.getByTestId('dpt-energy-preview-result')).toContainText('[0, 0.000002]')
  await expect(energy.getByTestId('dpt-energy-save')).toBeDisabled()
  await energy.getByTestId('dpt-energy-confirm').check()
  await energy.getByTestId('dpt-energy-save').click()

  await expect(energy).toContainText('Added E_on curve #1 with 2 measured points.')
  expect(dptReadCount).toBe(2)
  expect(requests.filter(request => request.method === 'POST')).toEqual([
    {
      method: 'POST',
      path: PREVIEW_PATH,
      body: {
        loss_type: 'e_on',
        points: [
          { measurement_index: 0, acquisition_index: 0, i_x: 10, vds_sign: 1, id_sign: -1 },
          { measurement_index: 0, acquisition_index: 1, i_x: 20, vds_sign: -1, id_sign: 1 }
        ]
      }
    },
    {
      method: 'POST',
      path: SAVE_PATH,
      body: {
        loss_type: 'e_on',
        points: [
          { measurement_index: 0, acquisition_index: 0, i_x: 10, vds_sign: 1, id_sign: -1 },
          { measurement_index: 0, acquisition_index: 1, i_x: 20, vds_sign: -1, id_sign: 1 }
        ],
        source_token: 'a'.repeat(64),
        request_token: 'b'.repeat(64),
        save_action: 'add',
        existing_curve_index: null,
        target_curve_count: 0,
        target_edit_token: null
      }
    }
  ])
})

test('stale-target save conflict clears the reviewed preview and keeps the conflict visible', async ({ page }) => {
  await page.route('**/api/**', async route => {
    const request = route.request()
    const url = new URL(request.url())
    const path = url.pathname

    if (request.method() === 'GET' && path === '/api/transistors') {
      await route.fulfill({ status: 200, json: [fixtureDevice()] })
      return
    }
    if (request.method() === 'GET' && path === DPT_DATA_PATH) {
      await route.fulfill({ status: 200, json: dptData() })
      return
    }
    if (request.method() === 'POST' && path === PREVIEW_PATH) {
      await route.fulfill({
        status: 200,
        json: {
          loss_type: 'e_on',
          event: 'turn_on',
          points: [
            {
              measurement_index: 0, acquisition_index: 0, i_x: 10,
              energy_j: 0.0014, start_s: 0, end_s: 2e-6,
              sample_count: 3, vds_sign: 1, id_sign: 1
            },
            {
              measurement_index: 0, acquisition_index: 1, i_x: 20,
              energy_j: 0.0028, start_s: 0, end_s: 2e-6,
              sample_count: 3, vds_sign: 1, id_sign: 1
            }
          ],
          graph_i_e: [[10, 20], [0.0014, 0.0028]],
          conditions: { t_j: 25, v_supply: 400, v_g: 15, v_g_off: -5, r_g: 10 },
          integration: {
            quantity: 'raw-window switching energy',
            formula: 'E = sum of signed trapezoids',
            power_formula: 'p(t) = Vds(t) * Id(t)',
            interval: 'exact stored first-to-last event sample time',
            boundary_detection: 'none; user-selected imported windows are used as stored'
          },
          source_token: 'a'.repeat(64),
          request_token: 'b'.repeat(64),
          save_action: 'add',
          existing_curve_index: null,
          target_curve_count: 0,
          target_edit_token: null
        }
      })
      return
    }
    if (request.method() === 'POST' && path === SAVE_PATH) {
      await route.fulfill({
        status: 409,
        json: { detail: 'The switching curve target changed after preview.' }
      })
      return
    }

    await route.fulfill({ status: 501, json: { detail: `Unmocked API request: ${path}` } })
  })

  await page.goto('/')
  await page.getByRole('button', { name: 'More Tools' }).click()
  await page.getByRole('button', { name: 'DPT Manager' }).click()
  await page.locator('.transistor-selector select').selectOption(DEVICE_NAME)

  const energy = page.getByTestId('dpt-energy-derivation')
  for (const index of [0, 1]) {
    await energy.getByTestId(`dpt-energy-record-${index}`).selectOption('0')
    await energy.getByTestId(`dpt-energy-acquisition-${index}`).selectOption(String(index))
    await energy.getByTestId(`dpt-energy-current-${index}`).fill(String((index + 1) * 10))
    await energy.getByTestId(`dpt-energy-vds-sign-${index}`).selectOption('1')
    await energy.getByTestId(`dpt-energy-id-sign-${index}`).selectOption('1')
  }
  await energy.getByTestId('dpt-energy-preview').click()
  await energy.getByTestId('dpt-energy-confirm').check()
  await energy.getByTestId('dpt-energy-save').click()

  await expect(energy.getByTestId('dpt-energy-preview-result')).toHaveCount(0)
  await expect(energy.getByTestId('dpt-energy-save')).toHaveCount(0)
  await expect(energy.getByRole('alert')).toContainText(
    'The switching curve target changed after preview.'
  )
})
