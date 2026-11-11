import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
const { grid, energyMap, thermalHandoff, thermalAnalyze } = vi.hoisted(() => ({ grid: vi.fn(), energyMap: vi.fn(), thermalHandoff: vi.fn(), thermalAnalyze: vi.fn() }))
vi.mock('../../services/api.js', () => ({ analysisApi: { grid, energyMap, thermalHandoff }, thermalApi: { analyze: thermalAnalyze } }))
import GridWorkbench from '../../components/GridWorkbench.vue'
import EnergyMapWorkbench from '../../components/EnergyMapWorkbench.vue'
import WaveformThermalTransfer from '../../components/WaveformThermalTransfer.vue'
import ThermalWorkbench from '../../components/ThermalWorkbench.vue'
import { analysisCsv } from '../../components/analysisFiles.js'

const waveform = {
  duration_s: 2, known_component_power_w: 7, total_power_w: null, total_status: 'partial-explicit-omissions',
  components: { conduction: { status: 'computed', energy_j: 14, power_w: 7 }, turn_on: { status: 'omitted', power_w: null }, turn_off: { status: 'omitted', power_w: null } },
  signals: { time_s: [0, 1, 2], current_a: [2, 4, 6] }, current_statistics: { mean: 4, rms: 4.1, method: 'Python exact fixture' },
  conduction_plot: [{ time_s: 0, power_w: 1, cumulative_j: 0 }, { time_s: 2, power_w: 13, cumulative_j: 14 }],
  switching_cumulative_plot: [], power_segments: [{ start_s: 0, end_s: 2, power_coefficients_w: [1, 2, 3, 4] }],
  events: [], event_policy: { mode: 'omitted', omission_reason: 'Fixture' }, request: { waveform: { current_a: [2, 4, 6] } }, assumptions: [],
  spectrum: { frequency_hz: [0, .5], amplitude_a: [4, 2], n: 3, sample_interval_s: 1, sequence_period_s: 3,
    window: 'hann', detrend: 'mean', normalization: 'Python FFT', discrete_input_rms_a: 4, windowed_discrete_rms_a: 3, parseval_rms_a: 3 },
}
const metric = { name: 'conduction_power_w', unit: 'W', direction: 'minimize' }
const summary = { unit: 'W', full_mean: null, subset_mean: 7, selected_mean: null, coverage: .5, retained_weight: 1, total_weight: 2,
  worst_value: 7, worst_point_ids: ['point-1'], worst_complete: false, zero_weight_in_worst: false }
const gridResult = {
  input_sha256: 'grid-hash', method_version: 'grid-v1', metric, summary_mode: 'full',
  inputs: { configurations: [{ id: 'config-1' }], operating_points: [{ id: 'point-1', weight: 1 }, { id: 'bad-point', weight: 1 }] },
  rows: [{ configuration_id: 'config-1', operating_point_id: 'point-1', weight: 1, status: 'computed', input_sha256: 'pair-hash', result: waveform,
    inputs: { waveform_request: waveform.request }, metrics: { conduction_power_w: { value: 7 } } },
  { configuration_id: 'config-1', operating_point_id: 'bad-point', weight: 1, status: 'unsupported', input_sha256: 'failure-hash', result: null,
    inputs: {}, metrics: { conduction_power_w: { value: null } }, error: 'No negative-current source' }],
  summaries: [{ configuration_id: 'config-1', metrics: { conduction_power_w: summary } }], ranking: [],
}
function fixed() { return { t_j_c: 25, v_g_on_v: null, v_g_off_v: null, r_g_on_ohm: null, r_g_off_ohm: null,
  commutation_device: null, commutation_source: null, energy_convention: 'Independent hand fixture E_on; no device measurement identity' } }
const mapResult = { method_version: 'map-v1', input_sha256: 'map-hash', energy_kind: 'on', currents_a: [0, 10], voltages_v: [400, 800],
  cells: [[{ current_a: 0, voltage_v: 400, energy_j: 0, status: 'available', mode: 'recorded-point', candidates: [], unsupported_slices: [] },
    { current_a: 10, voltage_v: 400, energy_j: null, status: 'ambiguous', candidates: [], unsupported_slices: [] }]],
  estimate: { query: { current_a: 5, voltage_v: 600 }, status: 'estimated', energy_j: .00075, method: 'bounded-bilinear-estimate', corners: [] },
  source_slices: [{ id: 'slice-1', source: 'Manual fixture', source_kind: 'manual', voltage_v: 400, current_a: [0, 10], energy_j: [0, .001], conditions: fixed() }],
  slice_diagnostics: [{ slice_id: 'slice-1', status: 'selected', reason: null }], assumptions: [],
}
const handoff = { scope: 'known-components-only', source_input_sha256: 'a'.repeat(64), input_sha256: 'b'.repeat(64), omitted_components: ['turn_on', 'turn_off'],
  source_request: waveform.request, waveform_result: waveform, method_version: 'waveform-thermal-handoff-v1',
  thermal_input: { segments: waveform.power_segments, events: [{ time_s: 1, energy_j: .002, heat_owner: 'forward', source: 'Exact event source', conditions: {} }],
    network_heat_owner: 'forward', event_mode: 'impulse', gap_policy: 'reject' } }
let wrapper
afterEach(() => wrapper?.unmount())
beforeEach(() => { vi.clearAllMocks(); grid.mockResolvedValue(gridResult); energyMap.mockResolvedValue(mapResult); thermalHandoff.mockResolvedValue(handoff) })
const options = { global: { stubs: { CurvePlotter: true } } }
async function submit() { await wrapper.get('form').trigger('submit'); await flushPromises() }

describe('finite grid and conditioned energy evidence', () => {
  it('neutralizes formula-like CSV metadata while retaining negative numeric values and exact JSON', () => {
    const row = { source: '  =HYPERLINK("example")', id: '@formula', signed_current_a: -3, note: '\tuser source' }
    const exact = JSON.stringify(row)
    const csv = analysisCsv([row])
    expect(csv).toContain('"\'  =HYPERLINK(""example"")"')
    expect(csv).toContain('"\'@formula"')
    expect(csv).toContain('"-3"')
    expect(csv).not.toContain('"\'-3"')
    expect(JSON.stringify(row)).toBe(exact)
  })
  it('uses exact selected Python traces, separate frequency axis and explicit unavailable coverage', async () => {
    wrapper = mount(GridWorkbench, options); await submit()
    expect(grid.mock.calls[0][0]).toMatchObject({ metric, summary_mode: 'full', include_zero_weight_in_worst: false })
    expect(wrapper.text()).toContain('No negative-current source')
    expect(wrapper.text()).toContain('Unavailable')
    expect(wrapper.text()).toContain('0.500000 · 1 / 2')
    const plots = wrapper.findAllComponents({ name: 'CurvePlotter' })
    expect(plots[0].props('datasets')[0].data).toEqual([{ x: 0, y: 2 }, { x: 1, y: 4 }, { x: 2, y: 6 }])
    expect(plots.at(-1).props('xLabel')).toBe('Frequency (Hz)')
    expect(plots.at(-1).props('datasets')[0].data).toEqual([{ x: 0, y: 4 }, { x: .5, y: 2 }])
    await wrapper.get('[data-testid=range-start]').setValue(.5)
    await wrapper.get('[data-testid=range-end]').setValue(1.5)
    expect(plots[0].props('xMin')).toBe(.5); expect(plots.at(-1).props('xMin')).toBe(null)
    expect(plots[0].props('yMin')).toBe(2); expect(plots[0].props('yMax')).toBe(6)
    expect(plots.at(-1).props('yMin')).toBe(null); expect(plots.at(-1).props('yMax')).toBe(null)
    expect(wrapper.text()).toContain('hann window; detrend mean')
    expect(wrapper.findComponent(WaveformThermalTransfer).props('acceptedRequest')).toEqual(waveform.request)
    await wrapper.get('[data-testid=time-cursor-sample]').setValue('1')
    expect(wrapper.get('[data-testid=time-cursor-value]').text()).toBe('1 s')
    expect(wrapper.get('[aria-label="Signed current cursor readout"]').text()).toContain('4.00000 A')
    expect(plots.slice(0, -1).every(plot => plot.props('cursorX') === 1)).toBe(true)
    expect(plots.at(-1).props('cursorX')).toBe(null)
    const stack = wrapper.get('[data-testid=loss-component-stack]')
    expect(stack.findAll('rect[data-component]')).toHaveLength(1)
    expect(stack.get('rect[data-component=conduction]').attributes('data-watts')).toBe('7')
    expect(stack.get('rect[data-component=conduction]').attributes('width')).toBe('600')
    expect(stack.text()).toContain('Turn-on: Unavailable · omitted')
    expect(stack.text()).toContain('Full total: Unavailable')
  })
  it('invalidates completed and pending grid responses after a source edit', async () => {
    wrapper = mount(GridWorkbench, options); await submit()
    await wrapper.get('[aria-label="Configuration 1 ID"]').setValue('edited-id')
    expect(wrapper.find('[data-testid=grid-results]').exists()).toBe(false)
    let resolve; grid.mockImplementationOnce(() => new Promise(done => { resolve = done }))
    await wrapper.get('form').trigger('submit')
    await wrapper.get('[aria-label="Point 1 weight"]').setValue(2)
    resolve(gridResult); await flushPromises()
    expect(wrapper.find('[data-testid=grid-results]').exists()).toBe(false)
    expect(wrapper.get('button[type=submit]').attributes('disabled')).toBeUndefined()
  })
  it('keeps source conditions unchanged when the map filter changes and retains unavailable cells', async () => {
    wrapper = mount(EnergyMapWorkbench, options)
    await wrapper.get('[aria-label="Map Junction temperature (°C)"]').setValue(125)
    await submit()
    expect(energyMap.mock.calls[0][0].conditions.t_j_c).toBe(125)
    expect(energyMap.mock.calls[0][0].slices.map(slice => slice.conditions.t_j_c)).toEqual([25, 25])
    expect(wrapper.get('[data-testid=energy-estimate]').text()).toContain('0.000750000 J')
    expect(wrapper.text()).toContain('ambiguous')
    expect(wrapper.findComponent({ name: 'CurvePlotter' }).props('datasets')[0].data).toEqual([{ x: 0, y: 0 }, { x: 10, y: .001 }])
    await wrapper.get('[aria-label="Map voltages"]').setValue('400,900')
    expect(wrapper.find('[data-testid=energy-map-results]').exists()).toBe(false)
  })
  it('discards a map reply if a slice source changes during the request', async () => {
    let resolve; energyMap.mockImplementationOnce(() => new Promise(done => { resolve = done }))
    wrapper = mount(EnergyMapWorkbench, options)
    await wrapper.get('form').trigger('submit')
    await wrapper.get('[aria-label="Slice 1 ID"]').setValue('different-source')
    resolve(mapResult); await flushPromises()
    expect(wrapper.find('[data-testid=energy-map-results]').exists()).toBe(false)
  })
  it('bounds the prepared matrix and lazy source evidence before mounting a large accepted response', async () => {
    const source = { ...mapResult.source_slices[0], current_a: Array.from({ length:12001 },(_,i) => i), energy_j:Array(12001).fill(0) }
    const coordinates = Array.from({length:17},(_,i) => i)
    energyMap.mockResolvedValueOnce({ ...mapResult, source_slices:[source], currents_a:coordinates, voltages_v:coordinates,
      cells:coordinates.map(v => coordinates.map(i => ({current_a:i,voltage_v:v,energy_j:0,status:'available',candidates:[],unsupported_slices:[]}))) })
    wrapper = mount(EnergyMapWorkbench, options); await submit()
    expect(wrapper.findComponent({name:'CurvePlotter'}).exists()).toBe(false)
    expect(wrapper.get('.heatmap').findAll('tbody tr')).toHaveLength(16)
    expect(wrapper.get('.heatmap').findAll('td')).toHaveLength(256)
    const evidence = wrapper.findComponent({name:'EnergySourceSliceEvidence'})
    expect(evidence.find('tbody').exists()).toBe(false)
    evidence.element.open = true; await evidence.trigger('toggle')
    expect(evidence.findAll('tbody tr')).toHaveLength(50)
    await evidence.findAll('button').find(button => button.text()==='Next slice points').trigger('click')
    expect(evidence.get('tbody td').text()).toBe('50.0000')
    expect(wrapper.get('[data-testid=energy-3d-counts]').text()).toContain('12001 finite points · 0 rendered points')
    expect(wrapper.text()).toContain('point render limit is exceeded')
    await wrapper.findAll('button').find(button => button.text()==='Next current columns').trigger('click')
    expect(wrapper.get('.heatmap').findAll('td')).toHaveLength(16)
  })
})

describe('exact waveform to thermal transfer', () => {
  it('requires explicit physical owners and forwards partial acceptance to Python', async () => {
    wrapper = mount(WaveformThermalTransfer, { props: { analysis: waveform, acceptedRequest: waveform.request } })
    await wrapper.get('button').trigger('click'); expect(thermalHandoff).not.toHaveBeenCalled()
    for (const direction of ['positive', 'negative', 'zero']) await wrapper.get(`[aria-label="${direction}-current heat owner"]`).setValue('forward')
    await wrapper.get('[aria-label="Handoff thermal network"]').setValue('forward')
    await wrapper.get('input[type=checkbox]').setValue(true)
    await wrapper.get('button').trigger('click'); await flushPromises()
    expect(thermalHandoff).toHaveBeenCalledWith({ waveform_request: waveform.request, direction_owners: { positive: 'forward', negative: 'forward', zero: 'forward' }, network_heat_owner: 'forward', event_mode: 'impulse', allow_partial: true })
    expect(wrapper.emitted('thermal-handoff')[0][0]).toEqual(handoff)
  })
  it('rejects recomputed numeric or source evidence that differs from the displayed accepted result', async () => {
    wrapper = mount(WaveformThermalTransfer, { props: { analysis: waveform, acceptedRequest: waveform.request } })
    for (const direction of ['positive', 'negative', 'zero']) await wrapper.get(`[aria-label="${direction}-current heat owner"]`).setValue('forward')
    await wrapper.get('[aria-label="Handoff thermal network"]').setValue('forward')
    thermalHandoff.mockResolvedValueOnce({ ...handoff, waveform_result: { ...waveform, known_component_power_w: 7.000000001 } })
    await wrapper.get('button').trigger('click'); await flushPromises()
    expect(wrapper.emitted('thermal-handoff')).toBeUndefined()
    expect(wrapper.get('[role=alert]').text()).toContain('differs from the displayed result')
    thermalHandoff.mockResolvedValueOnce({ ...handoff, source_request: { waveform: { current_a: [2, 4, 7] } } })
    await wrapper.get('button').trigger('click'); await flushPromises()
    expect(wrapper.emitted('thermal-handoff')).toBeUndefined()
  })
  it('stages unchanged polynomials and events and requires thermal review before evaluation', async () => {
    thermalAnalyze.mockResolvedValue({ operation: 'transient', request: {}, result: {} })
    wrapper = mount(ThermalWorkbench, { ...options, props: { stagedHandoff: handoff } })
    await submit(); expect(thermalAnalyze).not.toHaveBeenCalled()
    expect(wrapper.get('[data-testid=staged-thermal-handoff]').text()).toContain('known-components-only')
    expect(wrapper.find('[aria-label="Heat segments"]').exists()).toBe(false)
    const review = wrapper.findAll('label').find(label => label.text().includes('I reviewed the thermal source'))
    await review.get('input').setValue(true); await submit()
    const request = thermalAnalyze.mock.calls[0][0]
    expect(request.segments).toEqual(handoff.thermal_input.segments)
    expect(request.events).toEqual(handoff.thermal_input.events)
    expect(request.network_heat_owner).toBe('forward'); expect(request.event_mode).toBe('impulse')
  })
})
