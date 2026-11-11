import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
const { post } = vi.hoisted(() => ({ post: vi.fn() }))
vi.mock('../../services/api', () => ({ default: { post } }))
import DeviceParameterCalculator from '../../components/DeviceParameterCalculator.vue'

const response = { data: { result: { v0_v: 1.2, resistance_ohm: 0.03, sample_count: 3 },
  condition_policy: 'Exact recorded conditions', units: 'SI', plots: [{ title: 'V-I fit',
    x_label: 'Current (A)', y_label: 'Voltage (V)', x: [5, 10], series: { Fitted: [1.35, 1.5] } }] } }
describe('device parameter workbench', () => {
  let wrapper
  beforeEach(() => { post.mockReset().mockResolvedValue(response); wrapper = mount(DeviceParameterCalculator, { global: { stubs: { CurvePlotter: true } } }) })
  afterEach(() => wrapper.unmount())
  async function submit() { await wrapper.get('button').trigger('click'); await flushPromises() }
  it('sends SI input with source conditions and plots backend result samples', async () => {
    await submit()
    expect(post).toHaveBeenCalledWith('/api/parameters/calculate', expect.objectContaining({ operation: 'linear-fit', lower_a: 0, upper_a: 21, current_a: [0, 5, 10, 20] }))
    expect(wrapper.get('[aria-label="Device parameter results"]').text()).toContain('0.03')
    expect(wrapper.findComponent({ name: 'CurvePlotter' }).props('datasets')[0].data).toEqual([{ x: 5, y: 1.35 }, { x: 10, y: 1.5 }])
    await wrapper.get('[name=upper_a]').setValue('19')
    expect(wrapper.find('[aria-label="Device parameter results"]').exists()).toBe(false)
  })
  it('requires explicit estimate selection and never reuses the previous operation result', async () => {
    await submit()
    await wrapper.get('select').setValue('gate-slew-estimate')
    expect(wrapper.get('button').attributes('disabled')).toBeDefined()
    expect(wrapper.find('[aria-label="Device parameter results"]').exists()).toBe(false)
    await wrapper.get('.estimate-choice [type=checkbox]').setValue(true)
    await submit()
    expect(post).toHaveBeenLastCalledWith('/api/parameters/calculate', expect.objectContaining({ operation: 'gate-slew-estimate', selected: true, c_res_f: 1e-10 }))
  })
  it('requires inverse estimate acknowledgment and defines selected values as minimum total resistance requirements', async () => {
    await wrapper.get('select').setValue('gate-resistance-estimate')
    expect(wrapper.get('button').attributes('disabled')).toBeDefined()
    await wrapper.get('[name=target_dv_dt_v_s]').setValue('20000000000')
    await wrapper.get('[name=c_rss_min_f]').setValue('1e-10')
    await wrapper.get('.estimate-choice [type=checkbox]').setValue(true)
    post.mockResolvedValue({ data: { operation: 'gate-resistance-estimate', context: { source: 'Independent algebra fixture' }, result: { r_on_ohm: 5.5, r_off_ohm: 6, assumptions: ['Constant capacitance'], model_source: 'SC mfiles/calcRg.m; sc-92afe9737e47b855' }, plots: [] } })
    await submit()
    expect(post.mock.lastCall[1]).toMatchObject({ operation: 'gate-resistance-estimate', selected: true, target_dv_dt_v_s: 20e9, c_rss_min_f: 1e-10 })
    const result = wrapper.get('[aria-label="Device parameter results"]')
    expect(result.text()).toContain('Minimum turn-on total series resistance (Ω)')
    expect(result.text()).toContain('minimum total series resistance')
    expect(result.text()).toContain('5.5')
    expect(result.text()).not.toContain('upper resistance bounds')
    expect(result.text()).toContain('sc-92afe9737e47b855')
  })
  it('rejects invalid JSON and displays structured server errors', async () => {
    await wrapper.get('[aria-label="Parameter input data"]').setValue('{')
    await submit()
    expect(post).not.toHaveBeenCalled()
    expect(wrapper.find('[role=alert]').exists()).toBe(true)
    await wrapper.get('select').setValue('measured-table')
    post.mockRejectedValueOnce({ response: { data: { detail: [{ loc: ['body', 'query'], msg: 'Outside domain' }] } } })
    await submit()
    expect(wrapper.get('[role=alert]').text()).toContain('query: Outside domain')
  })
  it('discards responses after editing or switching operations', async () => {
    let resolve
    post.mockImplementationOnce(() => new Promise(done => { resolve = done }))
    await submit()
    await wrapper.get('[name=lower_a]').setValue('2')
    resolve(response)
    await flushPromises()
    expect(wrapper.find('[aria-label="Device parameter results"]').exists()).toBe(false)
    expect(wrapper.get('button').attributes('disabled')).toBeUndefined()
  })
  it('accepts paired measurements and edits unknown conditions without JSON', async () => {
    await wrapper.get('[aria-label="Measured current samples (A)"]').setValue('2, 4, 6')
    await wrapper.get('[aria-label="Measured voltage samples (V)"]').setValue('1.26\n1.32\n1.38')
    await submit()
    expect(post).toHaveBeenLastCalledWith('/api/parameters/calculate', expect.objectContaining({ current_a: [2, 4, 6], voltage_v: [1.26, 1.32, 1.38] }))
    await wrapper.get('[aria-label="Source conditions: Junction temperature (°C) unknown"]').setValue(true)
    await submit()
    expect(post.mock.lastCall[1].conditions.t_j_c).toBe(null)
  })
  it('resets provenance when loading illustrative points for another measured quantity', async () => {
    await wrapper.get('select').setValue('measured-table')
    await wrapper.get('[name=source]').setValue('Real manufacturer measurement')
    await wrapper.get('[aria-label="Measured quantity"]').setValue('r_factor')
    expect(wrapper.get('[name=source]').element.value).toContain('User supplied example')
    await wrapper.get('[name=r_nom_ohm]').setValue('.04')
    await submit()
    expect(post.mock.lastCall[1]).toEqual(expect.objectContaining({ r_nom_ohm: .04, table: expect.objectContaining({ kind: 'r_factor', x: [25, 125] }) }))
  })
})
