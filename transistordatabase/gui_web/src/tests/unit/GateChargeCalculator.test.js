import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'

const { calculateGateCharge } = vi.hoisted(() => ({ calculateGateCharge: vi.fn() }))
vi.mock('../../services/api', () => ({ transistorApi: { calculateGateCharge } }))
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
