import { readFile } from 'node:fs/promises'
import { test, expect } from '@playwright/test'

async function setParameter(page, name, value) {
  const field = page.locator('label.param-item').filter({ hasText: name }).locator('input')
  await field.fill(String(value))
}

async function openTopology(page, topology = 'Boost') {
  await page.goto('/')
  await page.getByRole('button', { name: 'Topology', exact: true }).click()
  await page.getByRole('button', { name: new RegExp(topology) }).click()
}

function topologyResult(sweep, mode = 'estimate') {
  return {
    topology: 'buck',
    model: {
      mode,
      conduction_regime: 'CCM',
      output_polarity: 1,
      assumptions: ['Browser fixture for the topology result view.']
    },
    calculated: {
      duty_cycle: 0.5,
      i_in_avg_a: 5,
      i_out_avg_a: 10,
      i_l_avg_a: 10,
      i_l_peak_a: 16.25,
      i_l_ripple_pp_a: 12.5,
      v_out_ripple_pp_v: 0.15625,
      v_switch_block_v: 100,
      v_out_signed_v: 50
    },
    losses: mode === 'conduction-only' ? {
      conduction_switch_w: 1.2,
      conduction_diode_w: 0.8,
      switching_on_w: 0,
      switching_off_w: 0,
      total_w: 2,
      scope: 'included switch and diode channel conduction only'
    } : null,
    temperatures: null,
    performance: {
      p_out_w: 500,
      p_in_w: mode === 'conduction-only' ? 502 : null,
      efficiency_percent: mode === 'conduction-only' ? 99.6 : null,
      input_power_scope: mode === 'conduction-only' ? 'partial-loss estimate based on the ideal operating point' : null
    },
    waveforms: {
      time_s: [0, 0.00001, 0.00002, 0.00003],
      inductor_current_a: [8, 12, 8, 12]
    },
    ...(sweep ? { sweep } : {})
  }
}

async function mockTopologyCalculation(page, response) {
  let requestBody
  await page.route('**/api/topology/calculate', async route => {
    requestBody = route.request().postDataJSON()
    await route.fulfill({ json: response })
  })
  return () => requestBody
}

test.describe('topology migration browser workflow', () => {
  test('calculates a boost point through FastAPI and invalidates changed results', async ({ page }) => {
    await openTopology(page)
    await setParameter(page, 'Input voltage', 100)
    await setParameter(page, 'Output voltage magnitude', 200)
    await setParameter(page, 'Output power', 1000)
    await setParameter(page, 'Switching frequency', 20)
    await setParameter(page, 'Inductance', 1000)
    await setParameter(page, 'Output capacitance', 1000)

    await page.getByRole('button', { name: 'Calculate', exact: true }).click()
    const results = page.locator('.results-panel').first()
    await expect(results).toBeVisible()
    await expect(page.locator('.model-label')).toHaveText('Ideal CCM operating-point estimate')
    await expect(results).toContainText('10.000 A')
    await expect(results).toContainText('5.000 A')
    await expect(results).toContainText('0.5000')
    await expect(page.locator('.charts-section canvas')).toHaveCount(1)

    const downloadPromise = page.waitForEvent('download')
    await page.locator('.calc-header').getByRole('button', { name: 'Download JSON' }).click()
    const download = await downloadPromise
    expect(download.suggestedFilename()).toMatch(/^topology-boost-\d{4}-\d{2}-\d{2}\.json$/)
    const downloadPath = await download.path()
    const exported = JSON.parse(await readFile(downloadPath, 'utf8'))
    expect(exported).toMatchObject({
      topology: 'boost',
      result: { model: { mode: 'estimate' } },
    })
    expect(exported.request).toBeDefined()

    await setParameter(page, 'Input voltage', 110)
    await expect(page.locator('.results-panel')).toHaveCount(0)
    await expect(page.getByRole('button', { name: 'Download JSON' })).toBeDisabled()
  })

  test('reports a DCM boundary error in the browser instead of inventing a point', async ({ page }) => {
    await openTopology(page)
    await setParameter(page, 'Input voltage', 100)
    await setParameter(page, 'Output voltage magnitude', 200)
    await setParameter(page, 'Output power', 1000)
    await setParameter(page, 'Switching frequency', 20)
    await setParameter(page, 'Inductance', 1000)
    await page.locator('label.param-item').filter({ hasText: 'Conduction regime' }).locator('select').selectOption('DCM')

    await page.getByRole('button', { name: 'Calculate', exact: true }).click()
    await expect(page.getByRole('alert')).toContainText('at or above the CCM boundary')
    await expect(page.locator('.results-panel')).toHaveCount(0)
  })

  test('runs a DCM line sweep through FastAPI and exports its request and result', async ({ page }) => {
    await openTopology(page, 'Buck')
    await setParameter(page, 'Input voltage', 100)
    await setParameter(page, 'Output voltage magnitude', 50)
    await setParameter(page, 'Output power', 10)
    await setParameter(page, 'Switching frequency', 20)
    await setParameter(page, 'Inductance', 1000)
    await page.locator('label.param-item').filter({ hasText: 'Conduction regime' }).locator('select').selectOption('DCM')
    await page.locator('label.sweep-toggle input').check()
    await setParameter(page, 'X minimum', 10)
    await setParameter(page, 'X maximum', 15)
    await setParameter(page, 'X points', 3)

    const apiResponse = page.waitForResponse(response => (
      response.url().includes('/api/topology/calculate')
      && response.request().method() === 'POST'
    ))
    await page.getByRole('button', { name: 'Calculate', exact: true }).click()
    const response = await apiResponse
    expect(response.status()).toBe(200)
    const calculation = await response.json()
    expect(calculation.model).toMatchObject({ mode: 'estimate', conduction_regime: 'DCM' })
    expect(calculation.sweep).toMatchObject({ kind: 'line', x_axis: 'f_sw', x_unit: 'Hz' })
    expect(calculation.sweep.x_values).toEqual([10000, 12500, 15000])
    expect(calculation.sweep.values).toHaveLength(3)
    await expect(page.locator('.model-label')).toHaveText('Ideal DCM operating-point estimate')
    await expect(page.getByRole('img', { name: 'i_l_ripple_pp_a vs f_sw' })).toBeVisible()

    const downloadPromise = page.waitForEvent('download')
    await page.locator('.calc-header').getByRole('button', { name: 'Download JSON' }).click()
    const download = await downloadPromise
    const exported = JSON.parse(await readFile(await download.path(), 'utf8'))
    expect(exported.request).toMatchObject({
      conduction_regime: 'DCM',
      sweep: { x_axis: 'f_sw', x_min: 10000, x_max: 15000, x_points: 3 }
    })
    expect(exported.result.sweep.values).toEqual(calculation.sweep.values)
    expect(exported.result.model.assumptions).toEqual(calculation.model.assumptions)
  })

  test('submits and exports Boost DCM channel feedback with both residuals', async ({ page }) => {
    await page.route('**/api/transistors', route => route.fulfill({ json: [
      { metadata: { name: 'Boost DCM Switch' } },
      { metadata: { name: 'Boost DCM Diode' } }
    ] }))
    const response = topologyResult(null, 'loss-balanced')
    response.topology = 'boost'
    response.model.conduction_regime = 'DCM'
    response.calculated = {
      ...response.calculated,
      duty_cycle: 0.45,
      i_l_peak_a: 2.25,
      switch_conduction_fraction: 0.45,
      diode_conduction_fraction: 0.44,
      idle_fraction: 0.11,
      inductor_volt_second_residual_v: 1e-10,
      power_balance_residual_w: 2e-10
    }
    response.losses = {
      conduction_switch_w: 1,
      conduction_diode_w: 2,
      total_w: 3,
      scope: 'switch and diode channel conduction only',
      loss_scope: 'channel_conduction_only'
    }
    response.performance = {
      p_out_w: 100,
      p_in_w: 103,
      efficiency_percent: 97.09,
      power_balance_residual_w: 2e-10
    }
    const capturedRequest = await mockTopologyCalculation(page, response)

    await openTopology(page)
    await page.locator('label.param-item').filter({ hasText: 'Calculation mode' }).locator('select').selectOption('loss-balanced')
    await page.locator('label.param-item').filter({ hasText: 'Conduction regime' }).locator('select').selectOption('DCM')
    await page.locator('label.param-item').filter({ hasText: 'Switch transistor' }).locator('select').selectOption('Boost DCM Switch')
    await page.locator('label.param-item').filter({ hasText: 'Diode transistor' }).locator('select').selectOption('Boost DCM Diode')
    await setParameter(page, 'Output power', 100)
    await setParameter(page, 'Junction temperature', 25)
    await setParameter(page, 'Switch gate voltage', 15)
    await page.getByRole('button', { name: 'Calculate', exact: true }).click()

    await expect(page.locator('.model-label')).toHaveText('Boost DCM TDB channel loss-balanced averaged model')
    await expect(page.locator('.results-panel')).toContainText('Power-balance residual: 0.00000000 W')
    await expect(page.locator('.results-panel')).toContainText('Averaged inductor volt-second residual: 0.00000000 V')
    expect(capturedRequest()).toMatchObject({
      topology: 'boost', mode: 'loss-balanced', conduction_regime: 'DCM',
      switch_transistor_id: 'Boost DCM Switch', diode_transistor_id: 'Boost DCM Diode'
    })

    const downloadPromise = page.waitForEvent('download')
    await page.getByRole('button', { name: 'Download JSON' }).click()
    const exported = JSON.parse(await readFile(await (await downloadPromise).path(), 'utf8'))
    expect(exported.request).toMatchObject({ topology: 'boost', mode: 'loss-balanced', conduction_regime: 'DCM' })
    expect(exported.result.losses.loss_scope).toBe('channel_conduction_only')
    expect(exported.result.calculated.inductor_volt_second_residual_v).toBe(1e-10)
    expect(exported.result.calculated.power_balance_residual_w).toBe(2e-10)
  })

  test('submits and exports Buck DCM channel feedback with both residuals', async ({ page }) => {
    await page.route('**/api/transistors', route => route.fulfill({ json: [
      { metadata: { name: 'Buck DCM Switch' } },
      { metadata: { name: 'Buck DCM Diode' } }
    ] }))
    const response = topologyResult(null, 'loss-balanced')
    response.topology = 'buck'
    response.model.conduction_regime = 'DCM'
    response.calculated = {
      ...response.calculated,
      duty_cycle: 0.29,
      i_in_avg_a: 0.103,
      i_out_avg_a: 0.2,
      i_l_avg_a: 0.2,
      i_l_peak_a: 0.711,
      i_diode_avg_a: 0.097,
      switch_conduction_fraction: 0.29,
      diode_conduction_fraction: 0.273,
      idle_fraction: 0.437,
      inductor_volt_second_residual_v: 1e-10,
      power_balance_residual_w: 2e-10
    }
    response.losses = {
      conduction_switch_w: 0.103,
      conduction_diode_w: 0.194,
      total_w: 0.297,
      scope: 'switch and diode channel conduction only',
      loss_scope: 'channel_conduction_only'
    }
    response.performance = {
      p_out_w: 10,
      p_in_w: 10.3,
      efficiency_percent: 97.12,
      power_balance_residual_w: 2e-10
    }
    const capturedRequest = await mockTopologyCalculation(page, response)

    await openTopology(page, 'Buck')
    await page.locator('label.param-item').filter({ hasText: 'Calculation mode' }).locator('select').selectOption('loss-balanced')
    await page.locator('label.param-item').filter({ hasText: 'Conduction regime' }).locator('select').selectOption('DCM')
    await expect(page.getByRole('button', { name: 'Compare topologies' })).toBeDisabled()
    await page.locator('label.param-item').filter({ hasText: 'Switch transistor' }).locator('select').selectOption('Buck DCM Switch')
    await page.locator('label.param-item').filter({ hasText: 'Diode transistor' }).locator('select').selectOption('Buck DCM Diode')
    await setParameter(page, 'Input voltage', 100)
    await setParameter(page, 'Output voltage magnitude', 50)
    await setParameter(page, 'Output power', 10)
    await setParameter(page, 'Switching frequency', 20)
    await setParameter(page, 'Inductance', 1000)
    await setParameter(page, 'Junction temperature', 25)
    await setParameter(page, 'Switch gate voltage', 15)
    await page.getByRole('button', { name: 'Calculate', exact: true }).click()

    await expect(page.locator('.model-label')).toHaveText('Buck DCM TDB channel loss-balanced averaged model')
    await expect(page.locator('.results-panel')).toContainText('Power-balance residual: 0.00000000 W')
    await expect(page.locator('.results-panel')).toContainText('Averaged inductor volt-second residual: 0.00000000 V')
    expect(capturedRequest()).toMatchObject({
      topology: 'buck', mode: 'loss-balanced', conduction_regime: 'DCM',
      switch_transistor_id: 'Buck DCM Switch', diode_transistor_id: 'Buck DCM Diode'
    })

    const downloadPromise = page.waitForEvent('download')
    await page.getByRole('button', { name: 'Download JSON' }).click()
    const exported = JSON.parse(await readFile(await (await downloadPromise).path(), 'utf8'))
    expect(exported.request).toMatchObject({ topology: 'buck', mode: 'loss-balanced', conduction_regime: 'DCM' })
    expect(exported.result.losses.loss_scope).toBe('channel_conduction_only')
    expect(exported.result.calculated.inductor_volt_second_residual_v).toBe(1e-10)
    expect(exported.result.calculated.power_balance_residual_w).toBe(2e-10)
  })

  test('submits and exports inverting Buck-Boost DCM channel feedback', async ({ page }) => {
    await page.route('**/api/transistors', route => route.fulfill({ json: [
      { metadata: { name: 'Buck-Boost DCM Switch' } },
      { metadata: { name: 'Buck-Boost DCM Diode' } }
    ] }))
    const response = topologyResult(null, 'loss-balanced')
    response.topology = 'buck-boost'
    response.model.conduction_regime = 'DCM'
    response.model.output_polarity = -1
    response.calculated = {
      ...response.calculated,
      duty_cycle: 0.21,
      i_in_avg_a: 0.1,
      i_out_avg_a: 0.2,
      i_l_avg_a: 0.3,
      i_l_peak_a: 0.7,
      i_diode_avg_a: 0.2,
      switch_conduction_fraction: 0.21,
      diode_conduction_fraction: 0.4,
      idle_fraction: 0.39,
      inductor_volt_second_residual_v: 1e-10,
      power_balance_residual_w: 2e-10,
      v_out_signed_v: -50
    }
    response.losses = {
      conduction_switch_w: 0.1,
      conduction_diode_w: 0.4,
      total_w: 0.5,
      scope: 'included switch and diode channel conduction only',
      loss_scope: 'channel_conduction_only'
    }
    response.performance = {
      p_out_w: 10,
      p_in_w: 10.5,
      efficiency_percent: 95.24,
      power_balance_residual_w: 2e-10
    }
    const capturedRequest = await mockTopologyCalculation(page, response)

    await openTopology(page, 'Inverting buck-boost')
    await expect(page.getByText(
      'Scalar DCM feedback is available for Buck, Boost, inverting Buck-Boost, and Flyback.'
    )).toBeVisible()
    await page.locator('label.param-item').filter({ hasText: 'Calculation mode' }).locator('select').selectOption('loss-balanced')
    await page.locator('label.param-item').filter({ hasText: 'Conduction regime' }).locator('select').selectOption('DCM')
    await expect(page.locator('.sweep-toggle input')).toBeDisabled()
    await expect(page.getByRole('button', { name: 'Compare topologies' })).toBeDisabled()
    await page.locator('label.param-item').filter({ hasText: 'Switch transistor' }).locator('select').selectOption('Buck-Boost DCM Switch')
    await page.locator('label.param-item').filter({ hasText: 'Diode transistor' }).locator('select').selectOption('Buck-Boost DCM Diode')
    await setParameter(page, 'Input voltage', 100)
    await setParameter(page, 'Output voltage magnitude', 50)
    await setParameter(page, 'Output power', 10)
    await setParameter(page, 'Switching frequency', 20)
    await setParameter(page, 'Inductance', 1000)
    await setParameter(page, 'Junction temperature', 25)
    await setParameter(page, 'Switch gate voltage', 15)
    await page.getByRole('button', { name: 'Calculate', exact: true }).click()

    await expect(page.locator('.model-label')).toHaveText('Buck-Boost DCM TDB channel loss-balanced averaged model')
    await expect(page.locator('.results-panel')).toContainText('Inverting buck-boost output')
    await expect(page.locator('.results-panel')).toContainText('signed output voltage is -50.00 V')
    await expect(page.locator('.result-item').filter({ hasText: 'Average input current (loss-balanced)' })).toContainText('0.100 A')
    await expect(page.locator('.result-item').filter({ hasText: 'Average output current' })).toContainText('0.200 A')
    await expect(page.locator('.result-item').filter({ hasText: 'Average inductor current' })).toContainText('0.300 A')
    await expect(page.locator('.loss-item').filter({ hasText: 'Switch conduction' })).toContainText('0.100 W')
    await expect(page.locator('.loss-item').filter({ hasText: 'Diode conduction' })).toContainText('0.400 W')
    await expect(page.locator('.results-panel')).toContainText('Power-balance residual: 0.00000000 W')
    await expect(page.locator('.results-panel')).toContainText('Averaged inductor volt-second residual: 0.00000000 V')
    expect(capturedRequest()).toMatchObject({
      topology: 'buck-boost', mode: 'loss-balanced', conduction_regime: 'DCM',
      switch_transistor_id: 'Buck-Boost DCM Switch',
      diode_transistor_id: 'Buck-Boost DCM Diode'
    })

    const downloadPromise = page.waitForEvent('download')
    await page.getByRole('button', { name: 'Download JSON' }).click()
    const exported = JSON.parse(await readFile(await (await downloadPromise).path(), 'utf8'))
    expect(exported.request).toMatchObject({
      topology: 'buck-boost', mode: 'loss-balanced', conduction_regime: 'DCM'
    })
    expect(exported.result.model.output_polarity).toBe(-1)
    expect(exported.result.losses.loss_scope).toBe('channel_conduction_only')
  })

  test('submits and exports Flyback DCM with primary and secondary current labels', async ({ page }) => {
    await page.route('**/api/transistors', route => route.fulfill({ json: [
      { metadata: { name: 'Flyback DCM Switch' } },
      { metadata: { name: 'Flyback DCM Rectifier' } }
    ] }))
    const response = topologyResult(null, 'loss-balanced')
    response.topology = 'flyback'
    response.model.conduction_regime = 'DCM'
    response.calculated = {
      ...response.calculated,
      duty_cycle: 0.404,
      i_in_avg_a: 0.404,
      i_out_avg_a: 0.769,
      i_l_avg_a: 0.789,
      i_l_peak_a: 2,
      i_l_ripple_pp_a: 2,
      i_diode_avg_a: 0.769,
      i_diode_peak_a: 4,
      i_diode_rms_a: 1.431,
      diode_conduction_fraction: 0.385,
      idle_fraction: 0.211,
      inductor_volt_second_residual_v: 1e-10,
      power_balance_residual_w: 2e-10
    }
    response.losses = {
      conduction_switch_w: 0.404,
      conduction_diode_w: 1.538,
      total_w: 1.942,
      scope: 'included switch and diode channel conduction only',
      loss_scope: 'channel_conduction_only'
    }
    response.performance = {
      p_out_w: 38.462,
      p_in_w: 40.404,
      efficiency_percent: 95.2,
      power_balance_residual_w: 2e-10
    }
    const capturedRequest = await mockTopologyCalculation(page, response)

    await openTopology(page, 'Flyback')
    await page.locator('label.param-item').filter({ hasText: 'Calculation mode' }).locator('select').selectOption('loss-balanced')
    await page.locator('label.param-item').filter({ hasText: 'Conduction regime' }).locator('select').selectOption('DCM')
    await expect(page.locator('.sweep-toggle input')).toBeDisabled()
    await expect(page.getByRole('button', { name: 'Compare topologies' })).toBeDisabled()
    await page.locator('label.param-item').filter({ hasText: 'Switch transistor' }).locator('select').selectOption('Flyback DCM Switch')
    await page.locator('label.param-item').filter({ hasText: 'Diode transistor' }).locator('select').selectOption('Flyback DCM Rectifier')
    await setParameter(page, 'Flyback turns ratio (Np/Ns)', 2)
    await setParameter(page, 'Input voltage', 100)
    await setParameter(page, 'Output voltage magnitude', 50)
    await setParameter(page, 'Output power', 38.46)
    await setParameter(page, 'Switching frequency', 20)
    await setParameter(page, 'Inductance', 1000)
    await setParameter(page, 'Junction temperature', 25)
    await setParameter(page, 'Switch gate voltage', 15)
    await page.getByRole('button', { name: 'Calculate', exact: true }).click()

    await expect(page.locator('.model-label')).toHaveText('Flyback DCM TDB channel loss-balanced averaged model')
    await expect(page.locator('.result-item').filter({ hasText: 'Average primary magnetizing current' })).toContainText('0.789 A')
    await expect(page.locator('.result-item').filter({ hasText: 'Peak primary magnetizing current' })).toContainText('2.000 A')
    await expect(page.locator('.result-item').filter({ hasText: 'Average secondary diode current' })).toContainText('0.769 A')
    await expect(page.locator('.result-item').filter({ hasText: 'Peak secondary diode current' })).toContainText('4.000 A')
    await expect(page.locator('.result-item').filter({ hasText: 'RMS secondary diode current' })).toContainText('1.431 A')
    await expect(page.locator('.results-panel')).toContainText('Power-balance residual: 0.00000000 W')
    expect(capturedRequest()).toMatchObject({
      topology: 'flyback', mode: 'loss-balanced', conduction_regime: 'DCM',
      turns_ratio_nps: 2, switch_transistor_id: 'Flyback DCM Switch',
      diode_transistor_id: 'Flyback DCM Rectifier'
    })

    const downloadPromise = page.waitForEvent('download')
    await page.getByRole('button', { name: 'Download JSON' }).click()
    const exported = JSON.parse(await readFile(await (await downloadPromise).path(), 'utf8'))
    expect(exported.request).toMatchObject({
      topology: 'flyback', mode: 'loss-balanced', conduction_regime: 'DCM',
      turns_ratio_nps: 2
    })
    expect(exported.result.calculated.i_diode_peak_a).toBe(4)
    expect(exported.result.losses.loss_scope).toBe('channel_conduction_only')
  })

  test('sends separate switch and diode selections in a curve-backed browser request', async ({ page }) => {
    await page.route('**/api/transistors', route => route.fulfill({ json: [
      { metadata: { name: 'Browser Switch MOSFET' } },
      { metadata: { name: 'Browser Body Diode' } }
    ] }))
    const capturedRequest = await mockTopologyCalculation(page, topologyResult(null, 'conduction-only'))

    await openTopology(page, 'Buck')
    await page.locator('label.param-item').filter({ hasText: 'Calculation mode' }).locator('select').selectOption('conduction-only')
    await page.locator('label.param-item').filter({ hasText: 'Switch transistor' }).locator('select').selectOption('Browser Switch MOSFET')
    await page.locator('label.param-item').filter({ hasText: 'Diode transistor' }).locator('select').selectOption('Browser Body Diode')
    await page.getByRole('button', { name: 'Calculate', exact: true }).click()

    await expect(page.locator('.model-label')).toHaveText('TDB channel conduction-only partial model')
    await expect(page.locator('.results-panel')).toContainText('included switch and diode channel conduction only')
    expect(capturedRequest()).toMatchObject({
      mode: 'conduction-only',
      switch_transistor_id: 'Browser Switch MOSFET',
      diode_transistor_id: 'Browser Body Diode'
    })
  })

  test('renders a line sweep result as a plotted chart in the browser', async ({ page }) => {
    const sweep = {
      kind: 'line',
      x_axis: 'f_sw',
      x_unit: 'Hz',
      x_values: [50000, 100000, 150000],
      metric: 'i_l_ripple_pp_a',
      metric_unit: 'A',
      values: [5, 7.5, 10]
    }
    const capturedRequest = await mockTopologyCalculation(page, topologyResult(sweep))

    await openTopology(page, 'Buck')
    await page.locator('label.sweep-toggle input').check()
    await setParameter(page, 'X maximum', 150)
    await setParameter(page, 'X points', 3)
    await page.getByRole('button', { name: 'Calculate', exact: true }).click()

    const sweepSection = page.locator('.charts-section').filter({ hasText: 'i_l_ripple_pp_a vs f_sw' })
    const chart = sweepSection.getByRole('img', { name: 'i_l_ripple_pp_a vs f_sw' })
    await expect(chart).toBeVisible()
    const renderedPixels = await chart.evaluate(canvas => {
      const context = canvas.getContext('2d')
      const { data } = context.getImageData(0, 0, canvas.width, canvas.height)
      let opaquePixels = 0
      for (let index = 3; index < data.length; index += 4) {
        if (data[index] > 0) opaquePixels += 1
      }
      return { width: canvas.width, height: canvas.height, opaquePixels }
    })
    expect(renderedPixels.width).toBeGreaterThan(0)
    expect(renderedPixels.height).toBeGreaterThan(0)
    expect(renderedPixels.opaquePixels).toBeGreaterThan(100)
    expect(capturedRequest()).toMatchObject({
      sweep: { x_axis: 'f_sw', x_min: 50000, x_max: 150000, x_points: 3, metric: 'i_l_ripple_pp_a' }
    })
  })

  test('submits and exports a loss-balanced line sweep with its model limits', async ({ page }) => {
    const sweep = {
      kind: 'line', mode: 'loss-balanced', loss_scope: 'channel_conduction_only',
      voltage_law_note: 'Each point reports its averaged inductor volt-second residual.',
      voltage_law_residual_values_v: [0.01, 0.02],
      x_axis: 'f_sw', x_unit: 'Hz', x_values: [50000, 100000],
      metric: 'p_loss_total_w', metric_unit: 'W', values: [8, 10]
    }
    const capturedRequest = await mockTopologyCalculation(page, topologyResult(sweep, 'loss-balanced'))

    await openTopology(page)
    await page.locator('label.param-item').filter({ hasText: 'Calculation mode' }).locator('select').selectOption('loss-balanced')
    await page.locator('label.sweep-toggle input').check()
    await setParameter(page, 'X maximum', 100)
    await setParameter(page, 'X points', 2)
    await page.getByRole('button', { name: 'Calculate', exact: true }).click()

    const sweepSection = page.locator('.charts-section').filter({ hasText: 'p_loss_total_w vs f_sw' })
    await expect(sweepSection).toContainText('channel_conduction_only')
    await expect(sweepSection).toContainText('averaged inductor volt-second residual')
    await expect(sweepSection).toContainText('Hover a point to inspect its averaged volt-second residual')
    await expect(sweepSection.getByRole('img', { name: 'p_loss_total_w vs f_sw' })).toBeVisible()
    expect(capturedRequest()).toMatchObject({
      mode: 'loss-balanced',
      sweep: { x_axis: 'f_sw', x_min: 50000, x_max: 100000, x_points: 2 }
    })

    const downloadPromise = page.waitForEvent('download')
    await page.getByRole('button', { name: 'Download JSON' }).click()
    const download = await downloadPromise
    const exported = JSON.parse(await readFile(await download.path(), 'utf8'))
    expect(exported.request.mode).toBe('loss-balanced')
    expect(exported.request.sweep).toBeDefined()
    expect(exported.result.sweep.loss_scope).toBe('channel_conduction_only')
    expect(exported.result.sweep.voltage_law_residual_values_v).toEqual([0.01, 0.02])
    expect(exported.result.sweep.voltage_law_note).toContain('volt-second residual')
  })

  test('shows contour sweep values and coordinate tooltips in the browser', async ({ page }) => {
    const sweep = {
      kind: 'contour', mode: 'loss-balanced', loss_scope: 'channel_conduction_only',
      voltage_law_note: 'Each point reports its averaged inductor volt-second residual.',
      voltage_law_residual_values_v: [[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]],
      x_axis: 'f_sw',
      x_unit: 'Hz',
      x_values: [50000, 100000, 150000],
      y_axis: 'v_in',
      y_unit: 'V',
      y_values: [100, 200],
      metric: 'i_l_ripple_pp_a',
      metric_unit: 'A',
      values: [[5, 7.5, 10], [10, 15, 20]]
    }
    const capturedRequest = await mockTopologyCalculation(page, topologyResult(sweep, 'loss-balanced'))

    await openTopology(page, 'Buck')
    await page.locator('label.param-item').filter({ hasText: 'Calculation mode' }).locator('select').selectOption('loss-balanced')
    await page.locator('label.sweep-toggle input').check()
    await page.locator('label.param-item').filter({ hasText: 'Sweep type' }).locator('select').selectOption('contour')
    await setParameter(page, 'X maximum', 150)
    await setParameter(page, 'X points', 3)
    await setParameter(page, 'Y minimum', 100)
    await setParameter(page, 'Y maximum', 200)
    await setParameter(page, 'Y points', 2)
    await page.getByRole('button', { name: 'Calculate', exact: true }).click()

    await expect(page.getByRole('img', { name: 'i_l_ripple_pp_a contour plot' })).toBeVisible()
    const cells = page.locator('.contour-grid span')
    await expect(cells).toHaveCount(6)
    const firstCell = cells.first()
    await firstCell.hover()
    await expect(firstCell).toHaveAttribute('title', 'x=50000 Hz; y=100 V; i_l_ripple_pp_a=5 A; inductor_volt_second_residual=0.01 V')
    await expect(page.locator('.results-panel')).toContainText('X: f_sw (Hz); Y: v_in (V); color shows i_l_ripple_pp_a (A).')
    await expect(page.locator('.charts-section').filter({ hasText: 'i_l_ripple_pp_a contour' })).toContainText('channel_conduction_only')
    await expect(page.locator('.charts-section').filter({ hasText: 'i_l_ripple_pp_a contour' })).toContainText('volt-second residual')
    expect(capturedRequest()).toMatchObject({
      mode: 'loss-balanced',
      sweep: {
        x_axis: 'f_sw', x_min: 50000, x_max: 150000, x_points: 3,
        y_axis: 'v_in', y_min: 100, y_max: 200, y_points: 2,
        metric: 'i_l_ripple_pp_a'
      }
    })
  })
})
