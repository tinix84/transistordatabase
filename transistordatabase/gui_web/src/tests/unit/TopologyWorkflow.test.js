import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'

const { calculateTopology, compareTopologies } = vi.hoisted(() => ({
  calculateTopology: vi.fn(),
  compareTopologies: vi.fn()
}))

vi.mock('../../services/api.js', () => ({
  transistorApi: { calculateTopology, compareTopologies }
}))

import TopologyCalculator from '../../components/TopologyCalculator.vue'

const estimateResponse = {
  topology: 'buck',
  model: {
    mode: 'estimate',
    conduction_regime: 'CCM',
    output_polarity: 1,
    assumptions: ['Ideal averaged conversion ratio.']
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
  losses: null,
  temperatures: null,
  performance: { p_out_w: 500, p_in_w: null, efficiency_percent: null }
}

describe('TopologyCalculator API workflow', () => {
  let wrapper

  async function setSelect(labelText, value) {
    const label = wrapper.findAll('label').find(item => item.text().includes(labelText))
    await label.get('select').setValue(value)
  }

  async function setInput(labelText, value) {
    const label = wrapper.findAll('label.param-item').find(item => item.text().includes(labelText))
    await label.get('input').setValue(value)
  }

  beforeEach(() => {
    calculateTopology.mockReset()
    compareTopologies.mockReset()
    calculateTopology.mockResolvedValue(estimateResponse)
    wrapper = mount(TopologyCalculator, {
      props: { transistors: [{ metadata: { name: 'TDB switch' } }] }
    })
  })

  afterEach(() => {
    wrapper.unmount()
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('sends operating inputs in SI and renders backend results', async () => {
    await wrapper.get('button.btn-primary').trigger('click')
    await flushPromises()

    const request = calculateTopology.mock.calls[0][0]
    expect(request.mode).toBe('estimate')
    expect(request.topology).toBe('buck')
    expect(request.f_sw).toBe(100000)
    expect(request.l_value).toBeCloseTo(100e-6, 12)
    expect(request.c_out).toBeCloseTo(1000e-6, 12)
    expect(wrapper.text()).toContain('Ideal CCM operating-point estimate')
    expect(wrapper.text()).toContain('12.500 A')
    expect(wrapper.text()).toContain('Ideal averaged conversion ratio.')
  })

  it('sends the selected topology, detailed mode, devices, and a backend sweep', async () => {
    await setSelect('Calculation mode', 'detailed')
    await setSelect('Switch transistor', 'TDB switch')
    await setSelect('Diode transistor', 'TDB switch')
    const contourToggle = wrapper.findAll('input[type="checkbox"]')[0]
    await contourToggle.setValue(true)
    await wrapper.findAll('select').find(select => select.element.value === 'f_sw').setValue('f_sw')
    await wrapper.get('button.btn-primary').trigger('click')
    await flushPromises()

    const request = calculateTopology.mock.calls[0][0]
    expect(request.mode).toBe('detailed')
    expect(request.switch_transistor_id).toBe('TDB switch')
    expect(request.diode_transistor_id).toBe('TDB switch')
    expect(request.t_j).toBe(25)
    expect(request.sweep.x_min).toBe(50000)
    expect(request.sweep.x_max).toBe(200000)
  })

  it('sends loss-balanced mode and labels its input current correctly', async () => {
    calculateTopology.mockResolvedValueOnce({
      ...estimateResponse,
      topology: 'boost',
      calculated: { ...estimateResponse.calculated, inductor_volt_second_residual_v: 0 },
      model: { ...estimateResponse.model, mode: 'loss-balanced' },
      losses: {
        conduction_switch_w: 4,
        conduction_diode_w: 6,
        total_w: 10,
        scope: 'channel conduction only',
        loss_scope: 'channel_conduction_only'
      },
      performance: {
        p_out_w: 500,
        p_in_w: 510,
        efficiency_percent: 98.04,
        power_balance_residual_w: 0
      }
    })
    await wrapper.findAll('button.topology-card').find(button => button.text().includes('Boost')).trigger('click')
    await setSelect('Calculation mode', 'loss-balanced')
    await wrapper.get('button.btn-primary').trigger('click')
    await flushPromises()

    expect(calculateTopology.mock.calls[0][0]).toMatchObject({
      topology: 'boost', mode: 'loss-balanced', conduction_regime: 'CCM'
    })
    expect(wrapper.text()).toContain('CCM TDB channel loss-balanced averaged model')
    expect(wrapper.text()).toContain('Average input current (loss-balanced)')
    expect(wrapper.text()).toContain('Power-balance residual: 0.00000000 W')
    expect(wrapper.text()).toContain('Averaged inductor volt-second residual: 0.00000000 V')
    const comparisonButton = wrapper.findAll('button').find(button => button.text().includes('Compare topologies'))
    expect(comparisonButton.element.disabled).toBe(false)
  })

  it('sends loss-balanced sweep and renders its scope and voltage-law caveat', async () => {
    calculateTopology.mockResolvedValueOnce({
      ...estimateResponse,
      model: { ...estimateResponse.model, mode: 'loss-balanced' },
      sweep: {
        kind: 'line', x_axis: 'f_sw', x_unit: 'Hz', x_values: [50000, 100000],
        metric: 'p_loss_total_w', metric_unit: 'W', values: [8, 10],
        voltage_law_residual_values_v: [0.01, 0.02],
        loss_scope: 'channel_conduction_only',
        voltage_law_note: 'Each point reports its averaged inductor volt-second residual.'
      }
    })
    await setSelect('Calculation mode', 'loss-balanced')
    const sweepToggle = wrapper.findAll('input[type="checkbox"]')[0]
    expect(sweepToggle.element.disabled).toBe(false)
    await sweepToggle.setValue(true)
    await wrapper.get('button.btn-primary').trigger('click')
    await flushPromises()

    expect(calculateTopology.mock.calls[0][0]).toMatchObject({
      mode: 'loss-balanced',
      sweep: { x_axis: 'f_sw', metric: 'i_l_ripple_pp_a' }
    })
    expect(wrapper.text()).toContain('Sweep scope: channel_conduction_only')
    expect(wrapper.text()).toContain('Each point reports its averaged inductor volt-second residual.')
    expect(wrapper.text()).toContain('Hover a point to inspect its averaged volt-second residual')
  })

  it('runs scalar Boost DCM loss-balanced mode and displays both residuals', async () => {
    calculateTopology.mockResolvedValueOnce({
      ...estimateResponse,
      topology: 'boost',
      calculated: {
        ...estimateResponse.calculated,
        duty_cycle: 0.45,
        i_l_peak_a: 2.25,
        switch_conduction_fraction: 0.45,
        diode_conduction_fraction: 0.44,
        idle_fraction: 0.11,
        inductor_volt_second_residual_v: 1e-10,
        power_balance_residual_w: 2e-10
      },
      model: { ...estimateResponse.model, mode: 'loss-balanced', conduction_regime: 'DCM' },
      losses: {
        conduction_switch_w: 1,
        conduction_diode_w: 2,
        total_w: 3,
        scope: 'channel conduction only',
        loss_scope: 'channel_conduction_only'
      },
      performance: {
        p_out_w: 100,
        p_in_w: 103,
        efficiency_percent: 97.09,
        power_balance_residual_w: 2e-10
      }
    })
    await wrapper.findAll('button.topology-card').find(button => button.text().includes('Boost')).trigger('click')
    await setSelect('Calculation mode', 'loss-balanced')
    await setSelect('Conduction regime', 'DCM')
    await wrapper.get('button.btn-primary').trigger('click')
    await flushPromises()
    expect(calculateTopology.mock.calls[0][0]).toMatchObject({
      topology: 'boost', mode: 'loss-balanced', conduction_regime: 'DCM'
    })
    expect(wrapper.text()).toContain('Boost DCM TDB channel loss-balanced averaged model')
    expect(wrapper.text()).toContain('Power-balance residual: 0.00000000 W')
    expect(wrapper.text()).toContain('Averaged inductor volt-second residual: 0.00000000 V')
    const comparisonButton = wrapper.findAll('button').find(button => button.text().includes('Compare topologies'))
    expect(comparisonButton.element.disabled).toBe(true)
    expect(wrapper.text()).toContain('Only scalar Buck, Boost, and Buck-Boost DCM feedback is supported')
  })

  it('runs scalar Buck DCM loss-balanced mode and displays its solved metrics', async () => {
    calculateTopology.mockResolvedValueOnce({
      ...estimateResponse,
      topology: 'buck',
      calculated: {
        ...estimateResponse.calculated,
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
      },
      model: { ...estimateResponse.model, mode: 'loss-balanced', conduction_regime: 'DCM' },
      losses: {
        conduction_switch_w: 0.103,
        conduction_diode_w: 0.194,
        total_w: 0.297,
        scope: 'channel conduction only',
        loss_scope: 'channel_conduction_only'
      },
      performance: {
        p_out_w: 10,
        p_in_w: 10.3,
        efficiency_percent: 97.12,
        power_balance_residual_w: 2e-10
      }
    })
    await wrapper.findAll('button.topology-card').find(button => button.text().includes('Buck')).trigger('click')
    await setSelect('Calculation mode', 'loss-balanced')
    await setSelect('Conduction regime', 'DCM')
    await wrapper.get('button.btn-primary').trigger('click')
    await flushPromises()

    expect(calculateTopology.mock.calls[0][0]).toMatchObject({
      topology: 'buck', mode: 'loss-balanced', conduction_regime: 'DCM'
    })
    expect(wrapper.text()).toContain('Buck DCM TDB channel loss-balanced averaged model')
    expect(wrapper.text()).toContain('Average input current (loss-balanced)')
    expect(wrapper.text()).toContain('Power-balance residual: 0.00000000 W')
    expect(wrapper.text()).toContain('Averaged inductor volt-second residual: 0.00000000 V')
    const comparisonButton = wrapper.findAll('button').find(button => button.text().includes('Compare topologies'))
    expect(comparisonButton.element.disabled).toBe(true)
    expect(wrapper.text()).toContain('Only scalar Buck, Boost, and Buck-Boost DCM feedback is supported')
  })

  it('runs scalar inverting Buck-Boost DCM and disables sweep/comparison', async () => {
    calculateTopology.mockResolvedValueOnce({
      ...estimateResponse,
      topology: 'buck-boost',
      calculated: {
        ...estimateResponse.calculated,
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
      },
      model: {
        ...estimateResponse.model,
        mode: 'loss-balanced',
        conduction_regime: 'DCM',
        output_polarity: -1
      },
      losses: {
        conduction_switch_w: 0.1,
        conduction_diode_w: 0.4,
        total_w: 0.5,
        scope: 'switch and diode channel conduction only',
        loss_scope: 'channel_conduction_only'
      },
      performance: {
        p_out_w: 10,
        p_in_w: 10.5,
        efficiency_percent: 95.24,
        power_balance_residual_w: 2e-10
      }
    })
    await wrapper.findAll('button.topology-card').find(button => button.text().includes('Inverting buck-boost')).trigger('click')
    await setSelect('Calculation mode', 'loss-balanced')
    await setSelect('Conduction regime', 'DCM')
    expect(wrapper.text()).toContain(
      'Scalar DCM feedback is supported for Buck, Boost, and inverting Buck-Boost; Flyback DCM feedback is unsupported.'
    )

    const sweepToggle = wrapper.find('.sweep-toggle input')
    expect(sweepToggle.element.disabled).toBe(true)
    await wrapper.get('button.btn-primary').trigger('click')
    await flushPromises()

    expect(calculateTopology.mock.calls[0][0]).toMatchObject({
      topology: 'buck-boost', mode: 'loss-balanced', conduction_regime: 'DCM'
    })
    expect(wrapper.text()).toContain('Buck-Boost DCM TDB channel loss-balanced averaged model')
    expect(wrapper.text()).toContain('Inverting buck-boost output')
    expect(wrapper.text()).toContain('-50.00 V')
    expect(wrapper.text()).toContain('0.100 A')
    expect(wrapper.text()).toContain('0.200 A')
    expect(wrapper.text()).toContain('0.300 A')
    expect(wrapper.findAll('.loss-item').find(item => item.text().includes('Switch conduction')).text()).toContain('0.100 W')
    expect(wrapper.findAll('.loss-item').find(item => item.text().includes('Diode conduction')).text()).toContain('0.400 W')
    expect(wrapper.text()).toContain('Power-balance residual: 0.00000000 W')
    expect(wrapper.text()).toContain('Averaged inductor volt-second residual: 0.00000000 V')
    const comparisonButton = wrapper.findAll('button').find(button => button.text().includes('Compare topologies'))
    expect(comparisonButton.element.disabled).toBe(true)
  })

  it('shows backend validation errors instead of stale results', async () => {
    calculateTopology.mockRejectedValueOnce({
      response: { data: { detail: 'Boost requires v_out > v_in > 0' } }
    })
    await wrapper.get('button.btn-primary').trigger('click')
    await flushPromises()
    expect(wrapper.get('[role="alert"]').text()).toContain('Boost requires')
    expect(wrapper.text()).not.toContain('12.500 A')
  })

  it('sends explicit Flyback ratio and DCM regime', async () => {
    calculateTopology.mockResolvedValueOnce({
      ...estimateResponse,
      topology: 'flyback',
      model: { ...estimateResponse.model, conduction_regime: 'DCM' }
    })
    await wrapper.findAll('button.topology-card').find(button => button.text().includes('Flyback')).trigger('click')
    await setInput('Flyback turns ratio (Np/Ns)', '2')
    await setSelect('Conduction regime', 'DCM')
    await wrapper.get('button.btn-primary').trigger('click')
    await flushPromises()
    expect(calculateTopology.mock.calls[0][0]).toMatchObject({
      topology: 'flyback', turns_ratio_nps: 2, conduction_regime: 'DCM'
    })
    expect(wrapper.text()).toContain('Ideal DCM operating-point estimate')
  })

  it('calls the comparison endpoint with device ids separated from operating parameters', async () => {
    compareTopologies.mockResolvedValueOnce({ comparison: { ranking_available: false } })
    await wrapper.findAll('button').find(button => button.text().includes('Compare topologies')).trigger('click')
    await flushPromises()
    const request = compareTopologies.mock.calls[0][0]
    expect(request.parameters.mode).toBe('estimate')
    expect(request.parameters.topology).toBeUndefined()
    expect(request.switch_transistor_id).toBeUndefined()
  })

  it('renders CCM loss-balanced comparison residuals, thermal availability, and model limits safely', async () => {
    const assumptions = [
      'CCM duty is adjusted to balance the requested output power and included channel-conduction losses.',
      'Triangular inductor current in continuous conduction mode.',
      'Output ripple is a calculated peak-to-peak estimate; capacitor ESR and switching ripple are ignored.',
      'v_out and i_out are positive output magnitudes.',
      'Loss-balanced mode solves CCM duty/current so input power balances output power plus included channel losses.',
      'Inductor ripple is triangular and refined using measured switch-channel voltage at the selected exact curve conditions.',
      'CCM power and averaged volt-second residuals can differ because the CCM solver uses a triangular-ripple approximation with nonlinear channel curves.',
      'Only switch and diode channel conduction losses are included; switching, reverse recovery, gate drive, passive, and magnetic losses remain excluded.',
      'Thermal output is a one-pass estimate at the selected curve temperature, not an electrothermal iteration.',
      'Parallel devices are identical and share current evenly; each branch uses the original per-device TDB curves without modifying curve data.'
    ]
    const buckResult = {
      ...estimateResponse,
      topology: 'buck',
      model: {
        ...estimateResponse.model,
        mode: 'loss-balanced',
        conduction_regime: 'CCM',
        thermal_closure: 'unavailable',
        assumptions: [...assumptions, null, { ignored: 'non-string entry' }]
      },
      calculated: { ...estimateResponse.calculated, inductor_volt_second_residual_v: 0.125 },
      losses: { scope: 'included switch and diode channel conduction only' },
      performance: {
        p_out_w: 500,
        p_in_w: 510,
        efficiency_percent: 98.04,
        power_balance_residual_w: 0.001
      },
      temperatures: {
        t_case_c: 60,
        switch_c: null,
        diode_c: null,
        switch_unavailable_reason: 'switch junction-to-case data are missing',
        diode_unavailable_reason: 'diode case-to-sink data are missing',
        thermal_closure: 'unavailable',
        thermal_closure_reason: 'switch junction-to-case data are missing'
      }
    }
    const boostResult = {
      ...buckResult,
      topology: 'boost',
      model: {
        ...buckResult.model,
        thermal_closure: 'closed_at_selected_curve_temperature',
        assumptions: 'malformed assumption list'
      },
      calculated: { ...buckResult.calculated, inductor_volt_second_residual_v: null },
      performance: { ...buckResult.performance, power_balance_residual_w: null },
      temperatures: {
        t_case_c: 60,
        switch_c: 75,
        diode_c: 80,
        thermal_closure: 'closed_at_selected_curve_temperature'
      }
    }
    compareTopologies.mockResolvedValueOnce({
      comparison: { ranking_available: true, loss_scope: 'channel_conduction_only' },
      buck: buckResult,
      boost: boostResult,
      'buck-boost': { error: 'Buck-Boost channel curve is unavailable' },
      flyback: { supported: false, reason: 'Flyback operating point is unsupported' }
    })

    await setSelect('Calculation mode', 'loss-balanced')
    await wrapper.findAll('button').find(button => button.text().includes('Compare topologies')).trigger('click')
    await flushPromises()

    const text = wrapper.text()
    expect(text).toContain('Power-balance residual: 0.00100000 W')
    expect(text).toContain('Averaged inductor volt-second residual: 0.12500000 V')
    expect(text).toContain('Power-balance residual: — W')
    expect(text).toContain('Averaged inductor volt-second residual: — V')
    expect(text).toContain('Thermal status: unavailable')
    expect(text).toContain('Switch temperature unavailable: switch junction-to-case data are missing')
    expect(text).toContain('Diode temperature unavailable: diode case-to-sink data are missing')
    expect(text).toContain('Thermal status: closed_at_selected_curve_temperature')
    expect(text).toContain('Switch junction estimate: 75.0 °C')
    expect(text).toContain('Model assumptions and limitations')
    for (const assumption of assumptions) expect(text).toContain(assumption)
    expect(text).toContain('No model assumptions or limitations were reported in this response.')
    expect(text).not.toContain('[object Object]')
    expect(text).toContain('Buck-Boost channel curve is unavailable')
    expect(text).toContain('Flyback operating point is unsupported')
    expect(text).not.toMatch(/undefined|NaN/)
    expect(wrapper.findAll('.comparison-row')).toHaveLength(4)
  })

  it('keeps estimate comparison results clear and omits loss-balanced details', async () => {
    compareTopologies.mockResolvedValueOnce({
      comparison: {
        ranking_available: false,
        reason: 'Estimate mode does not calculate efficiency.'
      },
      buck: estimateResponse,
      boost: { ...estimateResponse, topology: 'boost' },
      'buck-boost': { ...estimateResponse, topology: 'buck-boost' },
      flyback: { ...estimateResponse, topology: 'flyback' }
    })

    await wrapper.findAll('button').find(button => button.text().includes('Compare topologies')).trigger('click')
    await flushPromises()

    expect(wrapper.text()).toContain('Estimate mode does not calculate efficiency.')
    expect(wrapper.text()).toContain('Estimate mode; efficiency is not calculated.')
    expect(wrapper.text()).not.toContain('Power-balance residual')
    expect(wrapper.text()).not.toContain('Thermal status:')
    expect(wrapper.text()).not.toContain('Model assumptions and limitations')
  })

  it('handles missing and malformed assumption lists in comparison results', async () => {
    compareTopologies.mockResolvedValueOnce({
      comparison: { ranking_available: false },
      buck: {
        ...estimateResponse,
        model: { mode: 'loss-balanced', conduction_regime: 'CCM' }
      },
      boost: {
        ...estimateResponse,
        topology: 'boost',
        model: {
          ...estimateResponse.model,
          mode: 'loss-balanced',
          assumptions: 'not an array'
        }
      },
      'buck-boost': { error: 'Buck-Boost result unavailable' },
      flyback: { supported: false, reason: 'Unsupported operating point.' }
    })

    await setSelect('Calculation mode', 'loss-balanced')
    await wrapper.findAll('button').find(button => button.text().includes('Compare topologies')).trigger('click')
    await flushPromises()

    expect(wrapper.text().match(/No model assumptions or limitations were reported in this response\./g)).toHaveLength(2)
    expect(wrapper.text()).not.toMatch(/undefined|NaN|\[object Object\]/)
  })

  it('downloads compare-only results with the exact comparison request and response', async () => {
    const comparisonResponse = {
      comparison: { ranking_available: false },
      buck: { topology: 'buck', supported: true },
      boost: { topology: 'boost', supported: true }
    }
    let downloadedBlob
    const createObjectURL = vi.fn(blob => {
      downloadedBlob = blob
      return 'blob:comparison-results'
    })
    vi.stubGlobal('URL', {
      createObjectURL,
      revokeObjectURL: vi.fn()
    })
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {})
    compareTopologies.mockResolvedValueOnce(comparisonResponse)

    await wrapper.findAll('button').find(button => button.text().includes('Compare topologies')).trigger('click')
    await flushPromises()

    const exportButton = wrapper.get('.header-actions button.btn-info')
    expect(exportButton.element.disabled).toBe(false)
    await exportButton.trigger('click')
    expect(createObjectURL).toHaveBeenCalledOnce()

    const exported = JSON.parse(await downloadedBlob.text())
    const comparisonRequest = JSON.parse(JSON.stringify(compareTopologies.mock.calls[0][0]))
    expect(exported.topology).toBeNull()
    expect(exported.request).toEqual(comparisonRequest)
    expect(exported.result).toBeNull()
    expect(exported.comparison).toEqual(comparisonResponse)
    expect(exported.comparison_request).toEqual(comparisonRequest)
  })

  it('does not restore comparison results when Reset is clicked during a pending request', async () => {
    let resolveComparison
    compareTopologies.mockReturnValueOnce(new Promise(resolve => { resolveComparison = resolve }))
    await wrapper.findAll('button').find(button => button.text().includes('Compare topologies')).trigger('click')
    expect(compareTopologies).toHaveBeenCalledOnce()

    await wrapper.find('.header-actions button').trigger('click')
    resolveComparison({ comparison: { ranking_available: false }, buck: { stale: true } })
    await flushPromises()

    expect(wrapper.findAll('.comparison-row')).toHaveLength(0)
    expect(wrapper.text()).not.toContain('Topology comparison')
    expect(wrapper.find('.header-actions button.btn-info').element.disabled).toBe(true)
  })

  it('sends blank optional detailed conditions as omitted values', async () => {
    await setSelect('Calculation mode', 'detailed')
    for (const labelText of [
      'Case / sink reference temperature', 'Turn-off gate voltage',
      'Diode channel gate voltage', 'Turn-on gate resistance', 'Turn-off gate resistance'
    ]) {
      const label = wrapper.findAll('label.param-item').find(item => item.text().includes(labelText))
      await label.get('input').setValue('')
    }
    await wrapper.get('button.btn-primary').trigger('click')
    await flushPromises()

    const request = calculateTopology.mock.calls[0][0]
    expect(request.t_case).toBeUndefined()
    expect(request.diode_v_g).toBeUndefined()
    expect(request.v_g_off).toBeUndefined()
    expect(request.r_g_on).toBeUndefined()
    expect(request.r_g_off).toBeUndefined()
  })

  it('sends conduction-only mode with exact channel conditions but no switching conditions', async () => {
    calculateTopology.mockResolvedValueOnce({
      ...estimateResponse,
      model: { ...estimateResponse.model, mode: 'conduction-only' },
      losses: {
        conduction_switch_w: 0.5,
        conduction_diode_w: 0.5,
        switching_on_w: 0,
        switching_off_w: 0,
        total_w: 1,
        scope: 'included switch and diode channel conduction only'
      },
      performance: {
        p_out_w: 500,
        p_in_w: 501,
        efficiency_percent: 99.8,
        input_power_scope: 'partial-loss estimate based on the ideal operating point'
      }
    })
    await setSelect('Calculation mode', 'conduction-only')
    await setInput('Diode channel gate voltage', '0')
    await wrapper.get('button.btn-primary').trigger('click')
    await flushPromises()

    const request = calculateTopology.mock.calls[0][0]
    expect(request).toMatchObject({
      mode: 'conduction-only', conduction_regime: 'CCM', t_j: 25, v_g: 15
    })
    expect(request.diode_v_g).toBe(0)
    expect(request.v_g_off).toBeUndefined()
    expect(request.r_g_on).toBeUndefined()
    expect(request.r_g_off).toBeUndefined()
    expect(wrapper.text()).toContain('Average input current (ideal lossless point)')
    expect(wrapper.text()).toContain('Partial input power estimate')
    expect(wrapper.text()).not.toContain('Switching on')
  })

  it('clears saved-looking output when an operating input or topology changes', async () => {
    await wrapper.get('button.btn-primary').trigger('click')
    await flushPromises()
    expect(wrapper.find('.results-panel').exists()).toBe(true)

    const inputVoltage = wrapper.findAll('label.param-item').find(item => item.text().includes('Input voltage'))
    await inputVoltage.get('input').setValue('420')
    await flushPromises()
    expect(wrapper.find('.results-panel').exists()).toBe(false)

    await wrapper.get('button.btn-primary').trigger('click')
    await flushPromises()
    await wrapper.findAll('button.topology-card').find(button => button.text().includes('Boost')).trigger('click')
    await flushPromises()
    expect(wrapper.find('.results-panel').exists()).toBe(false)
    expect(wrapper.find('.header-actions button.btn-info').element.disabled).toBe(true)
  })
})
