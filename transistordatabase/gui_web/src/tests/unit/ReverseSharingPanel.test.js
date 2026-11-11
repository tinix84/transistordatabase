import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
const { reverseSharing } = vi.hoisted(() => ({ reverseSharing: vi.fn() }))
vi.mock('../../services/api.js', () => ({ assemblyApi: { reverseSharing } }))
import ReverseSharingPanel from '../../components/ReverseSharingPanel.vue'

const source = (kind, voltage_v) => ({ kind, current_a: [0, 10], voltage_v, source: 'Fixture', isolation_evidence: 'Constructed isolated function', recorded_conditions: { junction_temperature_c: 25, gate_bias_v: kind === 'channel-only' ? 15 : 0 }, requested_conditions: { junction_temperature_c: 25, gate_bias_v: kind === 'channel-only' ? 15 : 0 } })
const response = { inputs: { channel: source('channel-only', [0, 1]), diode: source('diode-only', [.6, .8]) },
  result: { status: 'supported', support: 'recorded', diagnostic: 'Unique root', channel_current_a: -20 / 3, diode_current_a: -10 / 3, current_sum_a: -10, current_residual_a: 0,
    voltage_v: -2 / 3, channel_static_power_w: 40 / 9, diode_static_power_w: 20 / 9, total_static_power_w: 20 / 3, blocking_margin_v: null,
    feasible_channel_current_magnitude_a: { lower_a: 0, upper_a: 10 }, root_evidence: { isolated_root_count: 1, flat_intervals_a: [], candidate_channel_current_magnitude_a: [20 / 3], evaluated_knot_count: 2, voltage_residual_v: 0 }, equations: ['Ic + Id = I'], scope: 'Static power only', units: { current: 'A' } },
  evidence: { input_sha256: 'input-hash', result_sha256: 'result-hash' } }

describe('physical reverse-current sharing child', () => {
  let wrapper
  const field = name => wrapper.get(`[data-testid=sharing-${name}]`)
  beforeEach(() => { reverseSharing.mockReset().mockResolvedValue({ data: response, rawJson: JSON.stringify(response) }); wrapper = mount(ReverseSharingPanel, { global: { stubs: { CurvePlotter: true } } }) })
  afterEach(() => wrapper.unmount())
  async function ready() { await field('fixture').trigger('click'); await field('selected').setValue(true); await field('compatible').setValue(true) }
  async function submit() { await wrapper.get('form').trigger('submit'); await flushPromises() }
  it('requires explicit source and operating-state acknowledgments, including after loading a fixture', async () => {
    await field('fixture').trigger('click'); await submit(); expect(reverseSharing).not.toHaveBeenCalled(); expect(wrapper.text()).toContain('Acknowledge')
    await field('selected').setValue(true); await submit(); expect(reverseSharing).not.toHaveBeenCalled(); expect(wrapper.text()).toContain('compatible operating state')
    await field('compatible').setValue(true); await field('channel-isolation').setValue(''); await submit(); expect(reverseSharing).not.toHaveBeenCalled(); expect(wrapper.text()).toContain('isolation evidence')
  })
  it('submits independent conditions and SI source-only curves and plots the returned common-voltage split', async () => {
    await ready(); await submit()
    const sent = reverseSharing.mock.calls[0][0]
    expect(sent.current_a).toBe(-10); expect(sent.mode).toBe('strict-recorded'); expect(sent.channel.kind).toBe('channel-only')
    expect(sent.channel.requested_conditions).toEqual({ junction_temperature_c: 25, gate_bias_v: 15 }); expect(sent.diode.requested_conditions.gate_bias_v).toBe(0)
    expect(sent.channel.current_a).toEqual([0, 10]); expect(sent.diode.voltage_v).toEqual([.6, .8])
    expect(field('results').text()).toContain('Calculated static split'); expect(field('results').text()).toContain('not a measured current split')
    const plot = wrapper.findComponent({ name: 'CurvePlotter' }), props = plot.props(), dots = props.datasets.filter(item => item.showLine === false)
    expect(dots.map(item => item.data[0])).toEqual([{ x: 20 / 3, y: 2 / 3 }, { x: 10 / 3, y: 2 / 3 }]); expect(props.allowZoom).toBe(false)
    expect(props.xMin).toBeLessThan(0); expect(props.xMax).toBeGreaterThan(10); expect(props.yMin).toBeLessThan(0); expect(props.yMax).toBeGreaterThan(1)
  })
  it('keeps unsupported values unavailable with no solved dots and labels explicit blocking estimates', async () => {
    await ready(); const unsupported = structuredClone(response); Object.assign(unsupported.result, { status: 'unsupported', support: 'none', channel_current_a: null, diode_current_a: null, voltage_v: null, channel_static_power_w: null })
    reverseSharing.mockResolvedValueOnce({ data: unsupported, rawJson: JSON.stringify(unsupported) }); await submit()
    expect(field('results').text()).toContain('Unavailable'); expect(wrapper.findComponent({ name: 'CurvePlotter' }).props('datasets')).toHaveLength(2)
    await field('mode').setValue('ideal-diode-blocking'); const estimate = structuredClone(response); Object.assign(estimate.result, { support: 'estimate', channel_current_a: -4, diode_current_a: 0, voltage_v: -.4, blocking_margin_v: .2 })
    reverseSharing.mockResolvedValueOnce({ data: estimate, rawJson: JSON.stringify(estimate) }); await submit()
    expect(reverseSharing.mock.calls.at(-1)[0].mode).toBe('ideal-diode-blocking'); expect(field('results').text()).toContain('Selected blocking estimate')
    expect(wrapper.findComponent({ name: 'CurvePlotter' }).props('datasets').at(-1).data).toEqual([{ x: -0, y: .4 }])
  })
  it('keeps independently unknown gate bias null without replacing it with zero', async () => {
    await ready(); await field('diode-recorded-gate').setValue(''); await field('diode-requested-gate').setValue(''); await submit()
    expect(reverseSharing.mock.calls[0][0].diode.recorded_conditions.gate_bias_v).toBe(null)
    expect(reverseSharing.mock.calls[0][0].diode.requested_conditions.gate_bias_v).toBe(null)
    expect(reverseSharing.mock.calls[0][0].channel.recorded_conditions.gate_bias_v).toBe(15)
  })
  it('preserves known zero and rejects blank, quoted, missing or underflowing numeric input before transport', async () => {
    await ready(); await field('current').setValue('0'); await submit(); expect(reverseSharing.mock.calls[0][0].current_a).toBe(0)
    for (const [name, value] of [['current', ''], ['current', '-1e-999'], ['channel-recorded-gate', '1e-999'], ['channel-csv', 'current_a,voltage_v\n0,0\n10,'], ['channel-csv', 'current_a,voltage_v\n0,0\n"10",1']]) {
      await ready(); reverseSharing.mockClear(); await field(name).setValue(value); await submit(); expect(reverseSharing).not.toHaveBeenCalled(); expect(wrapper.find('[role=alert]').exists()).toBe(true)
    }
  })
  it('invalidates results and exports after each evidence, condition or policy edit', async () => {
    await ready()
    for (const [name, value] of [['channel-source', 'changed source'], ['diode-isolation', 'changed evidence'], ['channel-csv', 'current_a,voltage_v\n0,0\n20,2'], ['temperature', '30'], ['diode-requested-gate', '-2'], ['mode', 'ideal-diode-blocking']]) {
      await submit(); expect(field('results').exists()).toBe(true); await field(name).setValue(value)
      expect(wrapper.find('[data-testid=sharing-results]').exists()).toBe(false); expect(wrapper.find('[data-testid=sharing-json]').exists()).toBe(false)
    }
  })
  it('discards edited in-flight replies and keeps the latest request in control', async () => {
    await ready(); let resolveOld, resolveNew
    reverseSharing.mockImplementationOnce(() => new Promise(done => { resolveOld = done })).mockImplementationOnce(() => new Promise(done => { resolveNew = done }))
    await wrapper.get('form').trigger('submit'); await field('current').setValue('-4'); await wrapper.get('form').trigger('submit')
    resolveOld({ data: response, rawJson: 'old' }); await flushPromises(); expect(wrapper.find('[data-testid=sharing-results]').exists()).toBe(false); expect(field('calculate').attributes('disabled')).toBeDefined()
    resolveNew({ data: response, rawJson: 'new' }); await flushPromises(); expect(field('results').exists()).toBe(true); expect(field('calculate').attributes('disabled')).toBeUndefined()
  })
  it('shows structured route errors and clears them after an edit', async () => {
    await ready(); reverseSharing.mockRejectedValueOnce({ response: { data: { detail: [{ msg: 'Conditions do not match' }] } } }); await submit()
    expect(wrapper.get('[role=alert]').text()).toContain('Conditions do not match'); await field('current').setValue('-4'); expect(wrapper.find('[role=alert]').exists()).toBe(false)
  })
})
