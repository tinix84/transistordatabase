import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
const { analyze } = vi.hoisted(() => ({ analyze: vi.fn() }))
vi.mock('../../services/api.js', () => ({ assemblyApi: { analyze } }))
import SemiconductorAssemblyWorkbench from '../../components/SemiconductorAssemblyWorkbench.vue'

const response = { operation: 'parallel-bank', count: 3, device: { blocking_voltage_v: 1200, current_rating_a: 30, cost: 6, currency: 'CHF', area_m2: .03, weight_kg: null, internal_resistance_k_per_w: 1 / 3, shared_sink_resistance_k_per_w: .2, conduction: { x: [0, 30], y: [0, 2], source: 'Returned Python fixture' } }, assumptions: ['Ideal equal sharing'], request: { count: 3 } }
describe('semiconductor assembly browser adapter', () => {
  let wrapper
  beforeEach(() => { analyze.mockReset().mockResolvedValue(response); wrapper = mount(SemiconductorAssemblyWorkbench, { global: { stubs: { CurvePlotter: true } } }) })
  afterEach(() => wrapper.unmount())
  async function submit() { await wrapper.get('form').trigger('submit'); await flushPromises() }
  it('keeps the physical sharing form outside the original assembly form', () => {
    expect(wrapper.findAll('form')).toHaveLength(2)
    expect(wrapper.findAll('form form')).toHaveLength(0)
    expect(wrapper.find('[data-testid=reverse-sharing]').exists()).toBe(true)
    expect(wrapper.get('a[href="#reverse-current-sharing"]').text()).toBe('Go to reverse-current sharing')
    expect(wrapper.find('#reverse-current-sharing').exists()).toBe(true)
  })
  it('submits readable SI fields and plots exactly the Python result', async () => {
    await submit()
    expect(analyze.mock.calls[0][0].count).toBe(3)
    expect(analyze.mock.calls[0][0].device.turn_on.conditions.r_g_ohm).toBe(null)
    expect(wrapper.get('[data-testid=assembly-results]').text()).toContain('30.000000')
    expect(wrapper.get('[data-testid=assembly-results]').text()).toContain('Unknown')
    expect(wrapper.findComponent({ name: 'CurvePlotter' }).props('datasets')[0].data).toEqual([{ x: 0, y: 0 }, { x: 30, y: 2 }])
  })
  it('keeps estimate selection explicit and carries independent branch conditions', async () => {
    await wrapper.get('[aria-label="Assembly operation"]').setValue('signed-composite')
    await submit()
    expect(analyze.mock.calls.at(-1)[0].voltage_estimate).toBe(null)
    await wrapper.get('[aria-label="Select linear voltage estimate"]').setValue(true)
    await wrapper.get('[aria-label="Target supply voltage"]').setValue('200')
    await submit()
    expect(analyze.mock.calls.at(-1)[0].voltage_estimate).toEqual({ selected: true, target_voltage_v: 200 })
    expect(analyze.mock.calls.at(-1)[0].positive.turn_on.conditions.v_supply_v).toBe(400)
  })
  it('does not borrow a different condition when marked unknown', async () => {
    await wrapper.get('[aria-label="Single device Turn-on energy t_j_c unknown"]').setValue(true)
    await submit()
    expect(analyze.mock.calls[0][0].device.turn_on.conditions.t_j_c).toBe(null)
    expect(analyze.mock.calls[0][0].device.turn_off.conditions.t_j_c).toBe(25)
  })
  it('invalidates complete and pending results on any input change', async () => {
    await submit()
    await wrapper.get('[aria-label="Parallel device count"]').setValue(4)
    expect(wrapper.find('[data-testid=assembly-results]').exists()).toBe(false)
    let resolve
    analyze.mockImplementationOnce(() => new Promise(done => { resolve = done }))
    await wrapper.get('form').trigger('submit')
    await wrapper.get('[aria-label="Parallel device count"]').setValue(5)
    resolve(response); await flushPromises()
    expect(wrapper.find('[data-testid=assembly-results]').exists()).toBe(false)
  })
  it('blocks malformed sample data until the same field is repaired', async () => {
    await wrapper.get('[aria-label="Single device Conduction curve current A"]').setValue('0, invalid')
    expect(wrapper.get('button[type=submit]').attributes('disabled')).toBeDefined()
    await wrapper.get('[aria-label="Single device Conduction curve values"]').setValue('0, 2')
    expect(wrapper.get('button[type=submit]').attributes('disabled')).toBeDefined()
    await wrapper.get('[aria-label="Single device Conduction curve current A"]').setValue('0, 10')
    expect(wrapper.get('button[type=submit]').attributes('disabled')).toBeUndefined()
  })
  it('applies advanced input without replacing it with operation defaults', async () => {
    const data = { operation: 'reverse-envelope', selected: true, channel: { kind: 'channel', x: [0, 20], y: [0, 4], source: 'Actual channel', conditions: { t_j_c: 100 } }, diode: { kind: 'diode', x: [0, 20], y: [1, 2], source: 'Actual diode', conditions: { t_j_c: 100 } } }
    await wrapper.get('[aria-label="Assembly JSON"]').setValue(JSON.stringify(data))
    await wrapper.get('[aria-label="Assembly JSON"]').trigger('change')
    await submit()
    expect(analyze.mock.calls.at(-1)[0]).toEqual(data)
  })
  it('clears invalid sample state when a curve is explicitly unavailable', async () => {
    await wrapper.get('[aria-label="Single device Conduction curve current A"]').setValue('0, invalid')
    expect(wrapper.get('button[type=submit]').attributes('disabled')).toBeDefined()
    await wrapper.get('[aria-label="Single device include conduction"]').setValue(false)
    expect(wrapper.get('button[type=submit]').attributes('disabled')).toBeUndefined()
    await submit()
    expect(analyze.mock.calls.at(-1)[0].device.conduction).toBe(null)
  })
  it('shows structured API errors and clears them after edits', async () => {
    analyze.mockRejectedValueOnce({ response: { data: { detail: [{ loc: ['body', 'count'], msg: 'Invalid count' }] } } })
    await submit()
    expect(wrapper.get('[role=alert]').text()).toContain('Invalid count')
    await wrapper.get('[aria-label="Parallel device count"]').setValue(2)
    expect(wrapper.find('[role=alert]').exists()).toBe(false)
  })
  it('clears only the disabled Foster input flags while another curve remains invalid', async () => {
    await wrapper.get('[aria-label="Single device include Foster"]').setValue(true)
    await wrapper.get('[aria-label="Single device resistance_k_per_w"]').setValue('bad pole')
    expect(wrapper.get('[role=alert]').text()).toContain('finite numeric')
    await wrapper.get('[aria-label="Single device Conduction curve current A"]').setValue('bad current')
    await wrapper.get('[aria-label="Single device include Foster"]').setValue(false)
    expect(wrapper.get('button[type=submit]').attributes('disabled')).toBeDefined()
    await wrapper.get('[aria-label="Single device include conduction"]').setValue(false)
    expect(wrapper.get('button[type=submit]').attributes('disabled')).toBeUndefined()
    await submit()
    expect(analyze.mock.calls.at(-1)[0].device.foster).toBe(null)
  })
})
