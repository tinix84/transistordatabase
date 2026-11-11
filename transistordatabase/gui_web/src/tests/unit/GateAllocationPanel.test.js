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
