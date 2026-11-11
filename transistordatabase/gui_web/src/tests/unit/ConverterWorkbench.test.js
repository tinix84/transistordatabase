import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
const { analyze, presets } = vi.hoisted(() => ({ analyze: vi.fn(), presets: vi.fn() }))
vi.mock('../../services/api.js', () => ({ converterApi: { analyze, presets } }))
import ConverterWorkbench from '../../components/ConverterWorkbench.vue'
const binding = { role: 's12', members: ['s1', 's2'], multiplicity: 2, conduction_current: { signal: 'is2', scale: 1 }, switching_current: { signal: 'itp', scale: -1 }, gate_signal: 'vgs2', bus_voltage: { signal: 'vp', scale: 1 } }
const component = power => ({ status: 'computed', power_w: power, energy_j: power * 4e-6 })
const response = {
  duration_s: 4e-6, operation: 'waveform', request: { time_s: [0, 1e-6, 2e-6, 3e-6, 4e-6] },
  totals: { known_component_power_w: 1004, total_power_w: 1004, total_status: 'computed' },
  branches: [{ role: 's12', members: ['s1', 's2'], multiplicity: 2, known_component_power_w: 1004, total_power_w: 1004, total_status: 'computed',
    components: { conduction: component(2), turn_on: component(250), turn_off: component(250) }, scaled_components: { conduction: component(4), turn_on: component(500), turn_off: component(500) }, model_bindings: { source: 'independent source' },
    evidence: { conduction_signals: { time_s: [0, 4e-6], current_a: [1, 1] }, switching_signals: { time_s: [0, 4e-6], current_a: [-2, -3], gate_v: [0, 1] }, conduction_plot: [{ time_s: 0, power_w: 2, cumulative_j: 0 }, { time_s: 4e-6, power_w: 2, cumulative_j: 8e-6 }], events: [{ event: 'turn_on', time_s: 1e-6, current_a: -2, energy_j: .001 }], switching_cumulative_plot: [{ time_s: 0, energy_j: 0 }, { time_s: 4e-6, energy_j: .002 }] } }],
}
describe('converter role workflow', () => {
  let wrapper
  beforeEach(async () => {
    presets.mockReset().mockResolvedValue({ revision: '1', presets: [{ id: 'sc-dab', label: 'DAB eight devices', operation: 'waveform', bindings: [binding], source: 'independent mapping' }, { id: 'sc-six-switch', label: 'Six-switch envelope', operation: 'held-envelope-estimate', bindings: [binding], source: 'mapping' }, { id: 'sc-llc', label: 'LLC source omission', operation: 'waveform', bindings: [{ ...binding, source_switching_omission: true }], source: 'mapping' }] })
    analyze.mockReset().mockResolvedValue(response)
    wrapper = mount(ConverterWorkbench, { global: { stubs: { CurvePlotter: true } } }); await flushPromises()
  })
  afterEach(() => wrapper.unmount())
  async function fixture() { await wrapper.findAll('button').find(button => button.text() === 'Load independent DAB fixture').trigger('click') }
  async function submit() { await wrapper.get('form').trigger('submit'); await flushPromises() }
  it('rejects a cleared known model condition before JSON transport', async () => {
    await fixture()
    const label = wrapper.findAll('fieldset.model-editor details label').find(node => node.text().includes('Junction temperature'))
    await label.get('input[type=number]').setValue('')
    await submit()
    expect(analyze).not.toHaveBeenCalled()
    expect(wrapper.text()).toContain('Known condition t_j_c needs a finite number')
    expect(label.get('input[type=checkbox]').element.checked).toBe(true)
  })
  it('retains signed independent mappings and requires explicit symmetry selection', async () => {
    await fixture(); await submit()
    const sent = analyze.mock.calls[0][0]
    expect(sent.assumptions.symmetry_selected).toBe(false)
    expect(sent.signals.is2).toEqual([1, 1, 1, 1, 1]); expect(sent.signals.itp).toEqual([0, 2, 9, 3, 0])
    expect(sent.role_models.s12.switching.turn_on.positive.recorded_conditions.v_supply_v).toBe(200)
    expect(sent.role_models.s12.source).toContain('fixture')
    expect(wrapper.text()).toContain('-1 × itp')
    const plots = wrapper.findAllComponents({ name: 'CurvePlotter' })
    expect(plots).toHaveLength(6)
    expect(plots[0].props('datasets')[0].data).toEqual([{ x: 0, y: 1 }, { x: 4e-6, y: 1 }])
    expect(plots[1].props('datasets')[0].data).toEqual([{ x: 0, y: -2 }, { x: 4e-6, y: -3 }])
    expect(wrapper.text()).toContain('1004.00 W')
  })
  it('shares visual ranges without changing the full calculation interval', async () => {
    await fixture(); await submit(); await wrapper.get('[data-testid=converter-range-start]').setValue(1e-6)
    await wrapper.get('[data-testid=converter-range-end]').setValue(3e-6)
    expect(analyze).toHaveBeenCalledTimes(1)
    for (const plot of wrapper.findAllComponents({ name: 'CurvePlotter' })) { expect(plot.props('xMin')).toBe(1e-6); expect(plot.props('xMax')).toBe(3e-6); expect(plot.props('allowZoom')).toBe(false) }
  })
  it('suppresses completed and held results after source/input changes', async () => {
    await fixture(); await submit(); await wrapper.get('[data-testid=converter-source]').setValue('Changed source')
    expect(wrapper.find('[data-testid=converter-results]').exists()).toBe(false)
    let resolve; analyze.mockImplementationOnce(() => new Promise(done => { resolve = done }))
    await submit(); await wrapper.get('[data-testid=signal-itp]').setValue('0,4,9,6,0'); resolve(response); await flushPromises()
    expect(wrapper.find('[data-testid=converter-results]').exists()).toBe(false)
  })
  it('labels omitted totals unknown and does not plot omitted power as zero', async () => {
    const partial = JSON.parse(JSON.stringify(response)); partial.totals.total_power_w = null; partial.totals.total_status = 'partial-explicit-omissions'
    partial.branches[0].scaled_components.turn_off = { status: 'omitted', power_w: null, energy_j: null, reason: 'Unknown secondary switching' }
    analyze.mockResolvedValueOnce(partial); await fixture(); await submit()
    expect(wrapper.get('[data-testid=converter-results]').text()).toContain('Complete total: Unknown W')
    expect(wrapper.text()).toContain('Unknown secondary switching')
  })
  it('shows why a physically inactive role is supported zero', async () => {
    const inactive = JSON.parse(JSON.stringify(response)), reason = 'No physical positions in selected topology'
    inactive.branches[0].evidence = { inactive_reason: reason }
    for (const component of Object.values(inactive.branches[0].scaled_components)) { component.power_w = 0; component.inactive_reason = reason }
    analyze.mockResolvedValueOnce(inactive); await fixture(); await submit()
    expect(wrapper.get('[data-testid=converter-results]').text()).toContain(`Inactive role: ${reason}`)
    expect(wrapper.findAllComponents({ name: 'CurvePlotter' })).toHaveLength(0)
    expect(wrapper.findAll('td').some(cell => cell.text().includes(`computed ${reason}`))).toBe(true)
  })
  it('requires envelope selection, serializes held values, and keeps PWM distinction visible', async () => {
    await wrapper.findAll('button').find(button => button.text() === 'Load independent envelope fixture').trigger('click')
    await submit(); expect(analyze.mock.calls[0][0]).toMatchObject({ operation: 'held-envelope-estimate', signals: { ia: [10, 10], fs: [100000, 100000] }, assumptions: { envelope_selected: false } })
    expect(wrapper.text()).toContain('N−1 held intervals')
  })
  it('rejects duplicate names before the API and allows readable custom mappings', async () => {
    await fixture(); await wrapper.get('[data-testid=signal-name-1]').setValue('is2'); await submit()
    expect(analyze).not.toHaveBeenCalled(); expect(wrapper.get('[role=alert]').text()).toContain('unique')
    await fixture(); await wrapper.findAll('button').find(button => button.text() === 'Customize displayed branch mappings').trigger('click'); await submit()
    expect(analyze.mock.calls[0][0].bindings[0]).toEqual(binding)
  })
  it('rejects missing delimited values without silently collapsing a sample', async () => {
    await fixture(); await wrapper.get('[data-testid=signal-itp]').setValue('0,2,,9,3,0'); await submit()
    expect(analyze).not.toHaveBeenCalled(); expect(wrapper.get('[role=alert]').text()).toContain('missing numeric value')
  })
  it('serializes explicit bracket masks and does not parse hidden source-omitted energy models', async () => {
    await fixture()
    const editor = wrapper.findComponent({ name: 'ConverterRoleEditor' })
    editor.vm.$emit('update:modelValue', { ...editor.props('modelValue'), eventMode: 'masks', onMask: '0,1,0,0', offMask: '0,0,0,1' })
    await submit(); expect(analyze.mock.calls[0][0].role_models.s12.events).toEqual({ mode: 'masks', turn_on: [false, true, false, false], turn_off: [false, false, false, true] })
    await wrapper.findAll('button').find(button => button.text() === 'Load LLC omission fixture').trigger('click')
    const omitted = wrapper.findComponent({ name: 'ConverterRoleEditor' })
    const value = omitted.props('modelValue')
    omitted.vm.$emit('update:modelValue', { ...value, onPositive: { ...value.onPositive, pairs: 'not a curve' } })
    await submit(); expect(analyze.mock.calls.at(-1)[0].role_models.s12.switching).toBeUndefined()
  })
})
