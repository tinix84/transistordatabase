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

  it('disables topology comparison for unsupported loss-balanced DCM', async () => {
    await setSelect('Calculation mode', 'loss-balanced')
    await setSelect('Conduction regime', 'DCM')
    const comparisonButton = wrapper.findAll('button').find(button => button.text().includes('Compare topologies'))
    expect(comparisonButton.element.disabled).toBe(true)
    expect(wrapper.text()).toContain('the solver supports CCM only')
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
