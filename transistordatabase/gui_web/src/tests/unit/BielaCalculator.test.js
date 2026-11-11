import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
const { calculateBiela } = vi.hoisted(() => ({ calculateBiela: vi.fn() }))
vi.mock('../../services/api', () => ({ transistorApi: { calculateBiela } }))
import BielaCalculator from '../../components/BielaCalculator.vue'

const diagnostics = { turn_on: { iterations: 1, gm_relative_residual: 0, equation_residual_a: 0 },
  turn_off: { iterations: 1, gm_relative_residual: 0, equation_residual_a: 0 }, body_diode_energy_j: 0,
  turn_on_energy_partitions_j: { current_rise: 1e-6, voltage_fall: 6e-6 },
  turn_off_energy_partitions_j: { voltage_rise: 10e-6, current_fall: 1e-6 } }
const response = { results: { e_on: 7e-6, e_off: 11e-6, e_total: 18e-6, diagnostics },
  parameter_inputs_si: { source: 'fixture provenance' }, operating_point: { v_dc: 100, i_load: 10, t_j: 25 },
  assumptions: ['An explicit estimate'] }

describe('Biela actual-parameter workflow', () => {
  let wrapper
  beforeEach(() => {
    calculateBiela.mockReset().mockResolvedValue(response)
    wrapper = mount(BielaCalculator, { global: { stubs: { CurvePlotter: true, BielaEnergyBreakdown: true } } })
  })
  afterEach(() => wrapper.unmount())
  async function submit() {
    await wrapper.findAll('input[type=checkbox]')[1].setValue(true)
    await wrapper.get('button').trigger('click')
    await flushPromises()
  }
  it('requires an explicit estimate and sends the physical SI contract with provenance', async () => {
    expect(wrapper.get('button').attributes('disabled')).toBeDefined()
    await submit()
    const input = calculateBiela.mock.calls[0][0]
    expect(input.model_kind).toBe('christen_biela_estimate')
    expect(input.c_gs).toBe(1e-9)
    expect(input.q_oss).toBe(200e-9)
    expect(input.recovery_mode).toBe('omit')
    expect(input.parameter_t_j).toBe(25)
    expect(input.source).toContain('algebra')
    expect(input).not.toHaveProperty('q_g')
    expect(wrapper.get('[aria-label="Biela estimate results"]').text()).toContain('fixture provenance')
  })
  it('invalidates completed results when any model condition changes', async () => {
    await submit()
    await wrapper.get('[name=parameter_t_j]').setValue(100)
    expect(wrapper.find('[aria-label="Biela estimate results"]').exists()).toBe(false)
  })
  it('does not restore stale responses or errors after input edits', async () => {
    let resolve
    calculateBiela.mockImplementationOnce(() => new Promise(done => { resolve = done }))
    await submit()
    await wrapper.get('[name=r_g]').setValue(20)
    resolve(response)
    await flushPromises()
    expect(wrapper.find('[aria-label="Biela estimate results"]').exists()).toBe(false)
    let reject
    calculateBiela.mockImplementationOnce(() => new Promise((done, fail) => { reject = fail }))
    await wrapper.get('button').trigger('click')
    await wrapper.get('[name=r_g]').setValue(30)
    reject(new Error('stale failure'))
    await flushPromises()
    expect(wrapper.find('[role=alert]').exists()).toBe(false)
  })
  it('preserves successful sweep points and presents unsupported points separately', async () => {
    calculateBiela.mockResolvedValueOnce({ ...response, sweep: { supported_points: 1, requested_points: 2,
      points: [{ current_a: 0, status: 'unsupported', error: 'Zero-current regime unsupported' },
        { current_a: 10, status: 'supported', results: response.results }] } })
    await wrapper.findAll('input[type=checkbox]')[0].setValue(true)
    await submit()
    expect(calculateBiela.mock.calls[0][0].i_min).toBe(0)
    expect(wrapper.get('[aria-label="Biela sweep evidence"]').text()).toContain('1 / 2 supported')
    expect(wrapper.text()).toContain('Zero-current regime unsupported')
    const chart = wrapper.findAllComponents({ name: 'CurvePlotter' })[0]
    expect(chart.props('datasets')[0].data).toEqual([{ x: 0, y: null }, { x: 10, y: 7 }])
  })
  it('requires finite entries and shows structured backend errors', async () => {
    await wrapper.get('[name=q_oss]').setValue('')
    await submit()
    expect(calculateBiela).not.toHaveBeenCalled()
    expect(wrapper.get('[role=alert]').text()).toContain('finite')
    await wrapper.get('[name=q_oss]').setValue(200e-9)
    calculateBiela.mockRejectedValueOnce({ response: { data: { detail: [{ loc: ['body', 'q_oss'], msg: 'Wrong condition' }] } } })
    await wrapper.get('button').trigger('click')
    await flushPromises()
    expect(wrapper.get('[role=alert]').text()).toContain('q_oss: Wrong condition')
  })
  it('rejects a finite SI energy that overflows the separate µJ display conversion', async () => {
    calculateBiela.mockResolvedValueOnce({ ...response, results: { ...response.results, e_on: 1e308 } })
    await submit()
    expect(wrapper.find('[aria-label="Biela estimate results"]').exists()).toBe(false)
    expect(wrapper.get('[role=alert]').text()).toContain('finite µJ display range')
  })
})
