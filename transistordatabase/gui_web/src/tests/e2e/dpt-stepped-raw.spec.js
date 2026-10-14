import { test, expect } from '@playwright/test'

const DEVICE_NAME = 'E2E Stepped DPT Device'
const DEVICE_PATH = `/api/transistors/${encodeURIComponent(DEVICE_NAME)}`

function fixtureDevice() {
  return {
    metadata: {
      name: DEVICE_NAME,
      type: 'MOSFET',
      manufacturer: 'Fixture Semiconductor',
      author: 'Stepped DPT browser fixture'
    }
  }
}

function dptData(rawMeasurementData = []) {
  return {
    has_e_on: false,
    has_e_off: false,
    has_e_rr: false,
    metadata: null,
    raw_measurement_data: rawMeasurementData
  }
}

function selectionFromMultipart(request) {
  const body = request.postData() || ''
  const match = body.match(/name="selection_json"\r\n\r\n([^\r\n]+)/)
  return match ? JSON.parse(match[1]) : null
}

test('selects stepped RAW plot and steps, invalidates stale preview, and appends explicitly', async ({ page }) => {
  const traceRequests = []
  let importSelection = null
  let persistedMeasurement = null
  await page.route('**/api/**', async route => {
    const request = route.request()
    const url = new URL(request.url())
    const path = url.pathname

    if (request.method() === 'GET' && path === '/api/transistors') {
      await route.fulfill({ status: 200, json: [fixtureDevice()] })
      return
    }
    if (request.method() === 'GET' && path === `${DEVICE_PATH}/dpt_data`) {
      await route.fulfill({
        status: 200,
        json: dptData(persistedMeasurement ? [persistedMeasurement] : [])
      })
      return
    }
    if (request.method() === 'POST' && path === `${DEVICE_PATH}/dpt_raw_import/preview`) {
      await route.fulfill({
        status: 200,
        json: {
          filename: 'stepped.raw',
          plots: [
            {
              plot_index: 0, plot_name: 'AC Analysis', sample_count: 4,
              trace_names: ['frequency', 'V(out)'], is_transient: false,
              is_stepped: false, steps: []
            },
            {
              plot_index: 1, plot_name: 'Transient Analysis', sample_count: 8,
              trace_names: ['time', 'V(ds)', 'I(M1)'], is_transient: true,
              is_stepped: true,
              steps: [
                { step_index: 0, sample_count: 3, time_range_s: [0, 2e-6] },
                { step_index: 1, sample_count: 5, time_range_s: [0, 4e-6] }
              ]
            }
          ]
        }
      })
      return
    }
    if (request.method() === 'POST' && path === `${DEVICE_PATH}/dpt_raw_import/trace_preview`) {
      const selection = selectionFromMultipart(request)
      traceRequests.push(selection)
      const hasReviewedContext = Boolean(
        selection.same_context_confirmed && selection.metadata &&
        Array.isArray(selection.step_selections) &&
        selection.step_selections.length === selection.step_indices.length
      )
      const steps = selection.step_indices.map(stepIndex => ({
        step_index: stepIndex,
        sample_count: stepIndex === 0 ? 3 : 5,
        time_range_s: [0, stepIndex === 0 ? 2e-6 : 4e-6],
        preview_sample_count: stepIndex === 0 ? 3 : 5,
        traces: {
          'V(ds)': stepIndex === 0
            ? [[0, 400], [1e-6, 200], [2e-6, 100]]
            : [[0, 500], [2e-6, 250], [4e-6, 100]],
          'I(M1)': stepIndex === 0
            ? [[0, 0], [1e-6, 10], [2e-6, 20]]
            : [[0, 0], [2e-6, 20], [4e-6, 40]]
        },
        non_finite_preview_sample_counts: { 'V(ds)': 0, 'I(M1)': 0 }
      }))
      await route.fulfill({
        status: 200,
        json: {
          filename: 'stepped.raw',
          time_trace: 'time',
          vds_trace: 'V(ds)',
          id_trace: 'I(M1)',
          plot_index: selection.plot_index,
          step_indices: selection.step_indices,
          sample_count: steps.reduce((sum, item) => sum + item.sample_count, 0),
          preview_sample_count: steps.reduce((sum, item) => sum + item.preview_sample_count, 0),
          time_range_s: null,
          traces: {},
          steps,
          review_required: hasReviewedContext,
          review_token: hasReviewedContext ? 'server-issued-review-token' : null
        }
      })
      return
    }
    if (request.method() === 'POST' && path === `${DEVICE_PATH}/dpt_raw_import`) {
      importSelection = selectionFromMultipart(request)
      persistedMeasurement = {
        metadata: {
          dataset_type: 'dpt_u_i',
          comment: 'Imported stepped RAW capture',
          t_j: 125,
          v_supply: 400,
          v_g: 15,
          v_g_off: -5,
          r_g: 10,
          r_g_off: 12,
          load_inductance: 0.75,
          commutation_inductance: 0.125,
          measurement_testbench: 'Fixture bench',
          commutation_device: 'Fixture diode'
        },
        dpt_on_vds: [
          [[0, 400], [1e-6, 200], [2e-6, 100]],
          [[0, 500], [1e-6, 300], [2e-6, 100]]
        ],
        dpt_on_id: [
          [[0, 0], [1e-6, 10], [2e-6, 20]],
          [[0, 0], [1e-6, 20], [2e-6, 40]]
        ],
        dpt_off_vds: [
          [[0, 100], [1e-6, 300], [2e-6, 500]],
          [[0, 100], [1e-6, 350], [2e-6, 550]]
        ],
        dpt_off_id: [
          [[0, 20], [1e-6, 10], [2e-6, 0]],
          [[0, 40], [1e-6, 20], [2e-6, 0]]
        ],
        import_provenance: {
          source_plot_index: importSelection.plot_index,
          source_step_indices: importSelection.step_indices,
          step_parameters_imported: false
        }
      }
      await route.fulfill({
        status: 200,
        json: {
          success: true,
          acquisition_count: 2,
          on_sample_count: 4,
          off_sample_count: 4
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
  await page.locator('.transistor-selector select').selectOption(DEVICE_NAME)

  const importer = page.getByTestId('dpt-raw-import')
  await importer.getByTestId('dpt-raw-file').setInputFiles({
    name: 'stepped.raw',
    mimeType: 'application/octet-stream',
    buffer: Buffer.from('deterministic raw fixture')
  })
  await importer.getByTestId('dpt-raw-preview-file').click()
  await importer.getByTestId('dpt-raw-plot').selectOption('1')
  await importer.getByTestId('dpt-raw-step-select').locator('input').nth(0).check()
  await importer.getByTestId('dpt-raw-vds-trace').selectOption('V(ds)')
  await importer.getByTestId('dpt-raw-id-trace').selectOption('I(M1)')
  await importer.getByTestId('dpt-raw-preview-traces').click()
  await expect(importer.getByTestId('dpt-raw-waveform-preview')).toBeVisible()

  await importer.getByTestId('dpt-raw-step-select').locator('input').nth(1).check()
  await expect(importer.getByTestId('dpt-raw-waveform-preview')).toHaveCount(0)
  await importer.getByTestId('dpt-raw-preview-traces').click()
  await expect(importer.getByTestId('dpt-raw-step-windows')).toBeVisible()
  expect(traceRequests.at(-1)).toEqual({
    plot_index: 1,
    step_indices: [0, 1],
    time_trace: 'time',
    vds_trace: 'V(ds)',
    id_trace: 'I(M1)'
  })

  for (const [step, current] of [[0, '10'], [1, '20']]) {
    await importer.getByTestId(`dpt-raw-step-${step}-current`).fill(current)
    await importer.getByTestId(`dpt-raw-step-${step}-on-start`).fill('0')
    await importer.getByTestId(`dpt-raw-step-${step}-on-end`).fill('0.000002')
    await importer.getByTestId(`dpt-raw-step-${step}-off-start`).fill('0')
    await importer.getByTestId(`dpt-raw-step-${step}-off-end`).fill('0.000002')
  }
  await importer.getByTestId('dpt-raw-metadata-tj').fill('125')
  await importer.getByLabel('Supply voltage [V]').fill('400')
  await importer.getByLabel('Turn-on gate voltage [V]').fill('15')
  await importer.getByLabel('Turn-off gate voltage [V]').fill('-5')
  await importer.getByLabel('Turn-on gate resistance [Ω]').fill('10')
  await importer.getByLabel('Turn-off gate resistance [Ω]').fill('12')
  await importer.getByLabel('Load inductance [µH]').fill('0.75')
  await importer.getByLabel('Commutation inductance [µH]').fill('0.125')
  await importer.getByLabel('Test bench').fill('Fixture bench')
  await importer.getByLabel('Commutation device').fill('Fixture diode')
  await importer.getByTestId('dpt-raw-same-context').check()

  const save = importer.getByTestId('dpt-raw-import-submit')
  await expect(save).toBeDisabled()
  await importer.getByTestId('dpt-raw-preview-traces').click()
  await expect(save).toBeEnabled()
  expect(traceRequests.at(-1)).toMatchObject({
    plot_index: 1,
    step_indices: [0, 1],
    step_selections: [
      {
        step_index: 0,
        i_x: 10,
        turn_on: { start_s: 0, end_s: 2e-6 },
        turn_off: { start_s: 0, end_s: 2e-6 }
      },
      {
        step_index: 1,
        i_x: 20,
        turn_on: { start_s: 0, end_s: 2e-6 },
        turn_off: { start_s: 0, end_s: 2e-6 }
      }
    ],
    same_context_confirmed: true,
    metadata: {
      t_j: 125,
      v_supply: 400,
      v_g: 15,
      v_g_off: -5,
      r_g: 10,
      r_g_off: 12,
      load_inductance: 0.75,
      commutation_inductance: 0.125,
      measurement_testbench: 'Fixture bench',
      commutation_device: 'Fixture diode'
    }
  })
  await save.click()
  await expect(importer).toContainText('Added one DPT record with 2 selected step acquisitions')
  const measurements = page.getByTestId('dpt-measurement-select')
  await expect(measurements.locator('option')).toContainText([
    'Measurement 1 · dpt_u_i · Imported stepped RAW capture'
  ])
  await expect(page.getByTestId('dpt-acquisition-select').locator('option'))
    .toHaveCount(2)
  await expect(page.getByTestId('dpt-waveform-on-vds')).toContainText('Turn-on Vds')
  await expect(page.getByText('Imported stepped RAW capture', { exact: true })).toBeVisible()
  expect(importSelection).toEqual({
    plot_index: 1,
    review_token: 'server-issued-review-token',
    step_indices: [0, 1],
    time_trace: 'time',
    vds_trace: 'V(ds)',
    id_trace: 'I(M1)',
    step_selections: [
      {
        step_index: 0,
        i_x: 10,
        turn_on: { start_s: 0, end_s: 2e-6 },
        turn_off: { start_s: 0, end_s: 2e-6 }
      },
      {
        step_index: 1,
        i_x: 20,
        turn_on: { start_s: 0, end_s: 2e-6 },
        turn_off: { start_s: 0, end_s: 2e-6 }
      }
    ],
    same_context_confirmed: true,
    metadata: {
      t_j: 125,
      v_supply: 400,
      v_g: 15,
      v_g_off: -5,
      r_g: 10,
      r_g_off: 12,
      load_inductance: 0.75,
      commutation_inductance: 0.125,
      measurement_testbench: 'Fixture bench',
      commutation_device: 'Fixture diode'
    }
  })
})
