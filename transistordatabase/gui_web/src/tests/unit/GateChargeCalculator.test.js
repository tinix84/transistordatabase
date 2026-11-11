import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'

const { calculateGateCharge, downloadAnalysis } = vi.hoisted(() => ({ calculateGateCharge: vi.fn(), downloadAnalysis: vi.fn() }))
vi.mock('../../services/api', () => ({ transistorApi: { calculateGateCharge } }))
vi.mock('../../components/analysisFiles.js', async importOriginal => ({ ...await importOriginal(), downloadAnalysis }))
import GateChargeCalculator from '../../components/GateChargeCalculator.vue'

const trace = { time_s: [0, 2e-9], v_gate_v: [0, 5], i_gate_a: [2.5, 1.667],
  id_normalized: [0, 1], vds_normalized: [1, 0],
  phases: [{ name: 't_delay', start_s: 0, end_s: 2e-9, instantaneous: false }] }
const response = { estimate: { assumptions: ['An explicit test estimate'], turn_on: trace, turn_off: trace },
  turn_on_times: { t_total: 13.8655812973 }, turn_off_times: { t_total: 40.6090918152 } }

describe('gate timing workflow', () => {
  let wrapper
  beforeEach(() => {
    calculateGateCharge.mockReset().mockResolvedValue(response)
    wrapper = mount(GateChargeCalculator, { global: { stubs: { CurvePlotter: true } } })
  })
  afterEach(() => wrapper.unmount())
  async function submit() {
    await wrapper.get('input[type=checkbox]').setValue(true)
    await wrapper.get('button').trigger('click')
    await flushPromises()
  }
  it('requires selecting the estimate, sends SI and plots API traces in ns', async () => {
    expect(wrapper.get('button').attributes('disabled')).toBeDefined()
    await submit()
    expect(calculateGateCharge).toHaveBeenCalledWith({ q_gs: 10e-9, q_gd: 15e-9, q_g: 50e-9,
      v_th: 3, v_plateau: 5, r_g: 6, r_g_int: 0, v_driver: 15, v_off: 0 })
    expect(wrapper.text()).toContain('13.865581 ns')
    expect(wrapper.text()).toContain('40.609092 ns')
    const plots = wrapper.findAllComponents({ name: 'CurvePlotter' })
    expect(plots).toHaveLength(6)
    expect(plots[0].props('datasets')[0].data).toEqual([{ x: 0, y: 0 }, { x: 2, y: 5 }])
    expect(plots[2].props('title')).toContain('NORMALIZED')
  })
  it('adds a separate target phase and exports its complete relative-time evidence', async () => {
    downloadAnalysis.mockClear()
    const parameters = { q_gs: 4e-9, q_gd: 2e-9, q_g: 12e-9, v_th: 2, v_plateau: 4,
      r_g: 1.5, r_g_int: .5, v_driver: 10, v_off: 0 }
    for (const [name, value] of Object.entries(parameters)) await wrapper.get(`[name=${name}]`).setValue(value)
    await wrapper.get('[data-testid=gate-settling-enabled]').setValue(true)
    await wrapper.get('[data-testid=gate-settling-target]').setValue('7')
    const phase = { target_gate_v: 7, initial_gate_v: 4, duration_s: 2e-9 * Math.log(2),
      incremental_charge_c: 3e-9, post_plateau_capacitance_f: 1e-9, anchor_s: 1.688317914e-9,
      absolute_end_s: 3.074612275e-9, absolute_time_status: 'resolved', instantaneous: false, status: 'computed',
      relative_time_s: [0, 2e-9 * Math.log(2)], v_gate_v: [4, 7], i_gate_a: [3, 1.5],
      initial_gate_current_a: 3, final_gate_current_a: 1.5, method: 'constant-capacitance-post-miller-v1',
      scope: 'Gate charging only; no drain switching energy', charge_reference: 'Qg remains specified at Von',
      time_basis: 'Relative seconds after Miller end', assumption: 'Explicit constant-capacitance estimate' }
    const complete = { ...response, parameters: { ...parameters, settling_target_v: 7 }, estimate: { ...response.estimate, turn_on_settling: phase } }
    calculateGateCharge.mockResolvedValueOnce(complete)
    await submit()
    expect(calculateGateCharge.mock.calls.at(-1)[0]).toEqual({ ...parameters, settling_target_v: 7 })
    expect(wrapper.get('[data-testid=gate-settling-duration]').text()).toContain('1.38629e-9')
    const plots = wrapper.findAllComponents({ name: 'CurvePlotter' })
    expect(plots).toHaveLength(8)
    expect(plots[6].props('xLabel')).toContain('Relative time')
    expect(plots[6].props('datasets')[0].data[0]).toEqual({ x: 0, y: 4 })
    await wrapper.get('[data-testid=gate-settling-json]').trigger('click')
    expect(downloadAnalysis.mock.calls[0][0]).toEqual(complete)
    await wrapper.get('[data-testid=gate-settling-csv]').trigger('click')
    expect(downloadAnalysis.mock.calls[1][0]).toContain('relative_time_s')
    expect(downloadAnalysis.mock.calls[1][0]).toContain('anchor_s')
    expect(downloadAnalysis.mock.calls[1][0]).toContain('incremental_charge_c')
    await wrapper.get('[data-testid=gate-settling-target]').setValue('8')
    expect(wrapper.find('.gate-results').exists()).toBe(false)
    let resolve; calculateGateCharge.mockImplementationOnce(() => new Promise(done => { resolve = done }))
    await wrapper.get('button').trigger('click')
    await wrapper.get('[data-testid=gate-settling-enabled]').setValue(false)
    resolve(complete); await flushPromises()
    expect(wrapper.find('.gate-results').exists()).toBe(false)
  })
  it('does not draw a finite-time current pulse for an ideal zero-charge crossing', async () => {
    calculateGateCharge.mockResolvedValueOnce({ ...response, estimate: { ...response.estimate, turn_on_settling: {
      status: 'instantaneous-ideal-crossing', instantaneous: true, target_gate_v: 7, duration_s: 0,
      incremental_charge_c: 0, post_plateau_capacitance_f: 0, anchor_s: 1e-9, absolute_end_s: 1e-9,
      initial_gate_current_a: null, final_gate_current_a: null, relative_time_s: [0, 0], v_gate_v: [4, 7], i_gate_a: [null, null] } } })
    await submit()
    expect(wrapper.text()).toContain('No finite-time gate current pulse')
    expect(wrapper.findAllComponents({ name: 'CurvePlotter' })).toHaveLength(7)
    const plot = wrapper.findAllComponents({ name: 'CurvePlotter' }).at(-1)
    expect(plot.props('datasets')[0].showLine).toBe(false)
    expect(wrapper.get('[data-testid=gate-settling-results]').text()).toContain('Unavailable / Unavailable')
  })
  it('shows positive relative duration with an unavailable absolute timestamp', async () => {
    calculateGateCharge.mockResolvedValueOnce({ ...response, estimate: { ...response.estimate, turn_on_settling: {
      status: 'computed', instantaneous: false, target_gate_v: 4.000000000000001, duration_s: 2.96e-25,
      incremental_charge_c: 8.88e-25, post_plateau_capacitance_f: 1e-9, anchor_s: 1.688e-6,
      absolute_end_s: null, absolute_time_status: 'unresolved-use-relative-time', initial_gate_current_a: 3,
      final_gate_current_a: 2.9999999999999996, relative_time_s: [0, 2.96e-25], v_gate_v: [4, 4.000000000000001], i_gate_a: [3, 2.9999999999999996] } } })
    await submit()
    expect(wrapper.get('[data-testid=gate-settling-duration]').text()).toBe('2.96000e-25')
    expect(wrapper.text()).toContain('Unavailable — unresolved-use-relative-time')
  })
  it('plots supported large settling times in SI seconds without nanosecond overflow', async () => {
    calculateGateCharge.mockResolvedValueOnce({ ...response, estimate: { ...response.estimate, turn_on_settling: {
      status: 'computed', instantaneous: false, target_gate_v: 1.0000000000000002, duration_s: 6.93e300,
      incremental_charge_c: 2.22e283, post_plateau_capacitance_f: 1e299, anchor_s: 1e-6,
      absolute_end_s: 6.93e300, absolute_time_status: 'resolved', initial_gate_current_a: 4.44e-18,
      final_gate_current_a: 2.22e-18, relative_time_s: [0, 6.93e300], v_gate_v: [1, 1.0000000000000002], i_gate_a: [4.44e-18, 2.22e-18] } } })
    await submit()
    const plot = wrapper.findAllComponents({ name: 'CurvePlotter' })[6]
    expect(plot.props('xLabel')).toBe('Relative time after Miller end (s)')
    expect(plot.props('datasets')[0].data.at(-1).x).toBe(6.93e300)
    expect(plot.props('datasets')[0].data.every(row => Number.isFinite(row.x))).toBe(true)
  })
  it('rejects a blank or nonzero-underflow target before JSON transport', async () => {
    await wrapper.get('[data-testid=gate-settling-enabled]').setValue(true)
    await submit(); expect(calculateGateCharge).not.toHaveBeenCalled()
    expect(wrapper.get('[role=alert]').text()).toContain('finite decimal')
    await wrapper.get('[data-testid=gate-settling-target]').setValue('1e-400')
    await submit(); expect(calculateGateCharge).not.toHaveBeenCalled()
    expect(wrapper.get('[role=alert]').text()).toContain('too small')
  })
  it('invalidates completed results and errors when inputs change', async () => {
    await submit()
    await wrapper.get('[name=v_off]').setValue(-5)
    expect(wrapper.find('.gate-results').exists()).toBe(false)
    calculateGateCharge.mockRejectedValueOnce({ response: { data: { detail: 'Require Vplateau < Von' } } })
    await wrapper.get('button').trigger('click')
    await flushPromises()
    expect(wrapper.get('[role=alert]').text()).toContain('Require Vplateau < Von')
    await wrapper.get('[name=v_driver]').setValue(20)
    expect(wrapper.find('[role=alert]').exists()).toBe(false)
  })
  it('rejects nonfinite or missing fields before submission', async () => {
    await wrapper.get('[name=q_gs]').setValue('')
    await submit()
    expect(calculateGateCharge).not.toHaveBeenCalled()
    expect(wrapper.get('[role=alert]').text()).toContain('finite number')
  })
  it('does not restore a stale response after editing during calculation', async () => {
    let resolve
    calculateGateCharge.mockImplementationOnce(() => new Promise(done => { resolve = done }))
    await submit()
    await wrapper.get('[name=r_g]').setValue(12)
    resolve(response)
    await flushPromises()
    expect(wrapper.find('.gate-results').exists()).toBe(false)
    expect(wrapper.get('button').attributes('disabled')).toBeUndefined()
  })
  it('shows structured API validation errors and prevents stale error restoration', async () => {
    calculateGateCharge.mockRejectedValueOnce({ response: { data: {
      detail: [{ loc: ['body', 'q_gs'], msg: 'Input should be greater than 0' }] } } })
    await submit()
    expect(wrapper.get('[role=alert]').text()).toContain('q_gs: Input should be greater than 0')
    let reject
    calculateGateCharge.mockImplementationOnce(() => new Promise((resolve, fail) => { reject = fail }))
    await wrapper.get('button').trigger('click')
    await wrapper.get('[name=r_g]').setValue(12)
    reject(new Error('obsolete failure'))
    await flushPromises()
    expect(wrapper.find('[role=alert]').exists()).toBe(false)
  })
  it('hides results when estimate selection is withdrawn', async () => {
    await submit()
    await wrapper.get('input[type=checkbox]').setValue(false)
    expect(wrapper.find('.gate-results').exists()).toBe(false)
    expect(wrapper.get('button').attributes('disabled')).toBeDefined()
  })
})
