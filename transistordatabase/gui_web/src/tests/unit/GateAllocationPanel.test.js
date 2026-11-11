import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
const { allocate } = vi.hoisted(() => ({ allocate: vi.fn() }))
vi.mock('../../services/api.js', () => ({ waveformApi: { allocate } }))
import GateAllocationPanel from '../../components/GateAllocationPanel.vue'

const response = { inputs: { source: '=Fixture', gate: { mode: 'binary' }, conditions: { junction_temperature_c: null }, provenance: { imported_filename: 'signed.csv' } },
  result: { sample_count: 2, semantics: 'Sample-only mask', scope: 'No physical path claim', samples: [
    { time_s: 0, current_a: -3, gate_on: false, gate_on_current_a: 0, gate_off_current_a: -3 },
    { time_s: 1, current_a: 4, gate_on: true, gate_on_current_a: 4, gate_off_current_a: 0 },
  ] }, evidence: { input_sha256: 'input-hash', result_sha256: 'result-hash' } }
describe('gate-state allocation child', () => {
  let wrapper, parseSamples
  const field = name => wrapper.get(`[data-testid=allocation-${name}]`)
  beforeEach(() => {
    allocate.mockReset().mockResolvedValue({ data: response, rawJson: JSON.stringify(response) })
    parseSamples = vi.fn(() => ({ time_s: [0, 1], current_a: [-3, 4], gate_v: [0, 1] }))
    wrapper = mount(GateAllocationPanel, { props: { parseSamples, sampleRevision: 'original', provenance: { imported_filename: 'signed.csv', edited_after_import: false } }, global: { stubs: { CurvePlotter: true } } })
  })
  afterEach(() => wrapper.unmount())
  async function ready() { await field('source').setValue('=Fixture'); await field('selected').setValue(true) }
  async function submit() { await wrapper.get('form').trigger('submit'); await flushPromises() }
  it('requires explicit acknowledgment and source before transport', async () => {
    await submit(); expect(allocate).not.toHaveBeenCalled(); expect(wrapper.text()).toContain('Explicitly select')
    await field('selected').setValue(true); await submit()
    expect(allocate).not.toHaveBeenCalled(); expect(wrapper.text()).toContain('nonblank allocation source')
  })
  it('sends aligned signed SI samples and unknown conditions, and plots markers only', async () => {
    await ready(); await submit()
    expect(parseSamples).toHaveBeenCalledWith(true, true)
    expect(allocate).toHaveBeenCalledWith({ time_s: [0, 1], current_a: [-3, 4], gate: { mode: 'binary', state: [0, 1] }, selected: true, source: '=Fixture',
      conditions: { junction_temperature_c: null, gate_on_voltage_v: null }, provenance: { imported_filename: 'signed.csv', edited_after_import: false } })
    expect(field('results').text()).toContain('-3')
    for (const plot of wrapper.findAllComponents({ name: 'CurvePlotter' })) {
      expect(plot.props('datasets').every(dataset => dataset.showLine === false)).toBe(true)
      expect(plot.props('allowZoom')).toBe(false)
    }
  })
  it('requires gate samples and rejects nonbinary samples without an explicit threshold policy', async () => {
    await ready(); parseSamples.mockReturnValueOnce({ time_s: [0, 1], current_a: [1, 2] }); await submit()
    expect(wrapper.text()).toContain('requires a gate_v sample'); expect(allocate).not.toHaveBeenCalled()
    parseSamples.mockReturnValue({ time_s: [0, 1], current_a: [1, 2], gate_v: [-5, .5] }); await submit()
    expect(wrapper.text()).toContain('requires numeric 0/1'); expect(allocate).not.toHaveBeenCalled()
    await field('mode').setValue('threshold'); await field('threshold').setValue('.5'); await submit()
    expect(allocate.mock.calls[0][0].gate).toEqual({ mode: 'threshold', threshold_v: .5, gate_v: [-5, .5] })
  })
  it('sends explicit hysteresis settings and invalidates returned evidence after threshold or initial edits', async () => {
    await ready()
    parseSamples.mockReturnValue({ time_s: [0, 1], current_a: [-3, 4], gate_v: [-5, 5] })
    await field('mode').setValue('hysteresis')
    await wrapper.get('[data-testid=hysteresis-low]').setValue('-2')
    await wrapper.get('[data-testid=hysteresis-high]').setValue('2')
    const evidence = { method: 'strict-hysteresis-v1', threshold_scale: 'absolute', low_v: -2, high_v: 2,
      calibration_min_v: null, calibration_max_v: null, initial_state: false,
      initial_state_source: 'first-sample-below-low', states: [false, true], turn_on_brackets: [0], turn_off_brackets: [],
      semantics: 'Strict crossings. Equality and deadband hold the preceding sampled state.' }
    const returned = { ...response, result: { ...response.result, gate_digitization: evidence } }
    allocate.mockResolvedValue({ data: returned, rawJson: JSON.stringify(returned) })
    await submit()
    expect(parseSamples).toHaveBeenLastCalledWith(true, false)
    expect(allocate.mock.calls.at(-1)[0].gate).toEqual({ mode: 'hysteresis', gate_v: [-5, 5],
      digitizer: { thresholds: { scale: 'absolute', low_v: -2, high_v: 2 } } })
    expect(wrapper.get('[data-testid=allocation-hysteresis-evidence]').text()).toContain('first-sample-below-low')
    await wrapper.get('[data-testid=hysteresis-initial]').setValue('off')
    expect(wrapper.find('[data-testid=allocation-json]').exists()).toBe(false)
    await submit()
    expect(allocate.mock.calls.at(-1)[0].gate.digitizer.initial_state).toBe(false)
    await wrapper.get('[data-testid=hysteresis-high]').setValue('3')
    expect(wrapper.find('[data-testid=allocation-results]').exists()).toBe(false)
  })
  it('requires relative calibration and discards a stale hysteresis response after reference changes', async () => {
    await ready(); await field('mode').setValue('hysteresis')
    await wrapper.get('[data-testid=hysteresis-scale]').setValue('relative')
    await submit(); expect(allocate).not.toHaveBeenCalled()
    await wrapper.get('[data-testid=hysteresis-minimum]').setValue('-5')
    await wrapper.get('[data-testid=hysteresis-maximum]').setValue('15')
    let resolve
    allocate.mockImplementationOnce(() => new Promise(done => { resolve = done }))
    await wrapper.get('form').trigger('submit')
    expect(allocate.mock.calls.at(-1)[0].gate.digitizer.thresholds).toEqual({
      scale: 'relative', low_fraction: .25, high_fraction: .75, minimum_v: -5, maximum_v: 15 })
    await wrapper.get('[data-testid=hysteresis-minimum]').setValue('-4')
    resolve({ data: response, rawJson: JSON.stringify(response) }); await flushPromises()
    expect(wrapper.find('[data-testid=allocation-results]').exists()).toBe(false)
  })
  it('invalidates results and exports after source, CSV, provenance or policy changes', async () => {
    await ready(); await submit(); expect(field('results').exists()).toBe(true)
    await field('source').setValue('changed'); expect(wrapper.find('[data-testid=allocation-results]').exists()).toBe(false)
    await submit(); await wrapper.setProps({ sampleRevision: 'changed samples' }); expect(wrapper.find('[data-testid=allocation-json]').exists()).toBe(false)
    await submit(); await wrapper.setProps({ provenance: { imported_filename: 'other.csv', edited_after_import: true } }); expect(wrapper.find('[data-testid=allocation-results]').exists()).toBe(false)
    await submit(); await field('mode').setValue('threshold'); expect(wrapper.find('[data-testid=allocation-results]').exists()).toBe(false)
  })
  it('discards an in-flight result and structured error after edits', async () => {
    await ready(); let resolve
    allocate.mockImplementationOnce(() => new Promise(done => { resolve = done }))
    await wrapper.get('form').trigger('submit'); await wrapper.setProps({ sampleRevision: 'new samples' })
    resolve({ data: response, rawJson: JSON.stringify(response) }); await flushPromises()
    expect(wrapper.find('[data-testid=allocation-results]').exists()).toBe(false)
    allocate.mockRejectedValueOnce({ response: { data: { detail: [{ msg: 'Time is invalid' }] } } }); await submit()
    expect(wrapper.get('[role=alert]').text()).toContain('Time is invalid')
    await field('source').setValue('changed'); expect(wrapper.find('[role=alert]').exists()).toBe(false)
  })
  it('retains raw threshold and condition decimals until strict validation', async () => {
    await ready(); await field('mode').setValue('threshold'); await field('threshold').setValue('1e-999'); await submit()
    expect(allocate).not.toHaveBeenCalled(); expect(wrapper.text()).toContain('too small')
    await field('threshold').setValue('.5')
    await wrapper.findAll('input[type=number]')[1].setValue('1e-999'); await submit()
    expect(allocate).not.toHaveBeenCalled(); expect(wrapper.text()).toContain('too small')
  })
})
