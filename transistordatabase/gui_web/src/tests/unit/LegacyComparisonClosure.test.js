import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
const { getAll, compareAdvanced } = vi.hoisted(() => ({ getAll: vi.fn(), compareAdvanced: vi.fn() }))
vi.mock('../../services/api.js', () => ({ transistorApi: { getAll, compareAdvanced } }))
import Comparison from '../../components/TransistorComparisonEnhanced.vue'
import WaveformModelEditor from '../../components/WaveformModelEditor.vue'
import { curve, serializeCurve } from '../../components/converterForm.js'

const response = {
  observation_policy: 'All recorded slices shown at actual conditions',
  channel_characteristics: { title: 'Recorded channel slices', status: 'recorded-observations', curves: [{ name: 'first (t_j=125)', device: 'first', record_index: 0, v_data: [0, 2], i_data: [0, 10], conditions: { t_j: 125, v_g: 18 } }] },
  loss_breakdown: { title: 'Explicit loss estimate', rows: [{ name: 'first', status: 'computed', conduction: 5, switching_on: 100, switching_off: 200, total: 305 }, { name: 'second', status: 'unsupported', total: null, reason: 'Missing exact switching record' }] },
  efficiency: { title: 'Semiconductor-only efficiency estimate', scope: 'Excludes magnetics', rows: [{ name: 'first', status: 'unsupported', efficiency_percent: null, reason: 'Missing explicit output power' }] },
}
describe('honest legacy comparison', () => {
  let wrapper
  beforeEach(async () => {
    getAll.mockResolvedValue(['first', 'second', 'third'].map(name => ({ metadata: { name } })))
    compareAdvanced.mockReset().mockResolvedValue(response)
    wrapper = mount(Comparison, { global: { stubs: { CurvePlotter: true } } }); await flushPromises()
  })
  afterEach(() => wrapper.unmount())
  async function select() { await wrapper.get('[data-testid=comparison-device-1]').setValue('first'); await wrapper.get('[data-testid=comparison-device-3]').setValue('third') }
  async function run() { await wrapper.get('[data-testid=run-comparison]').trigger('click'); await flushPromises() }
  it('has observation-only defaults and preserves recorded conditions', async () => {
    await select(); await run()
    expect(compareAdvanced).toHaveBeenCalledWith(['first', 'third'], {})
    expect(wrapper.text()).toContain('t_j: 125; v_g: 18')
    const plot = wrapper.findComponent({ name: 'CurvePlotter' })
    expect(plot.props('datasets')[0].data).toEqual([{ x: 0, y: 0 }, { x: 2, y: 10 }])
    await wrapper.findAll('nav button').find(button => button.text() === 'Loss breakdown').trigger('click')
    expect(wrapper.text()).toContain('305.0000'); expect(wrapper.text()).toContain('Missing exact switching record'); expect(wrapper.text()).toContain('Unavailable')
    await wrapper.findAll('nav button').find(button => button.text() === 'Efficiency').trigger('click')
    expect(wrapper.text()).toContain('Excludes magnetics'); expect(wrapper.text()).toContain('Missing explicit output power')
  })
  it('rejects blank explicit inputs before transport', async () => {
    await select(); await wrapper.get('[data-testid=comparison-estimate]').setValue(true); await run()
    expect(compareAdvanced).not.toHaveBeenCalled(); expect(wrapper.get('[role=alert]').text()).toContain('finite number')
  })
  it('aligns configurations by occupied slots and transports explicit SI inputs', async () => {
    await select(); await wrapper.get('[data-testid=comparison-estimate]').setValue(true)
    for (const [name, value] of Object.entries({ t_j: 25, i_channel: 10, f_sw: 100000, duty: .25, output_power_w: 1000 })) await wrapper.get(`[data-testid=comparison-${name}]`).setValue(value)
    for (const slot of [1, 3]) {
      const fieldset = wrapper.findAll('fieldset')[slot === 1 ? 0 : 1]
      const inputs = fieldset.findAll('input')
      for (const [i, value] of [400, 15, 4, 7, 1].entries()) await inputs[i].setValue(value)
    }
    await wrapper.get('[data-testid=comparison-supply-3]').setValue(600)
    await run(); const sent = compareAdvanced.mock.calls[0][1]
    expect(sent.v_supply).toEqual([400, 600]); expect(sent.v_g_off).toEqual([null, null]); expect(sent.f_sw).toBe(100000); expect(sent.duty).toBe(.25)
  })
  it('suppresses a held response when settings change', async () => {
    let resolve; compareAdvanced.mockImplementation(() => new Promise(done => { resolve = done }))
    await select(); await wrapper.get('[data-testid=run-comparison]').trigger('click')
    await wrapper.get('[data-testid=comparison-device-3]').setValue('second')
    resolve(response); await flushPromises()
    expect(wrapper.find('[data-testid=comparison-results]').exists()).toBe(false); expect(wrapper.text()).toContain('Inputs changed')
  })
})

describe('known condition serialization', () => {
  it('keeps cleared known conditions invalid and preserves explicit unknown', async () => {
    const value = curve(); value.recorded.t_j_c = NaN
    expect(() => serializeCurve(value)).toThrow('Known condition t_j_c')
    const wrapper = mount(WaveformModelEditor, { props: { modelValue: value, title: 'Source model' } })
    expect(wrapper.text()).toContain('Known condition needs a finite number')
    expect(wrapper.get('input[aria-invalid=true]').exists()).toBe(true)
    wrapper.unmount(); value.recorded.t_j_c = null
    expect(serializeCurve(value).recorded_conditions.t_j_c).toBeNull()
    value.requested.r_g_ohm = NaN
    expect(() => serializeCurve(value)).toThrow('Known condition r_g_ohm')
  })
})
