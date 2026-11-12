import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
const { analyze } = vi.hoisted(() => ({ analyze: vi.fn() }))
vi.mock('../../services/api.js', () => ({ waveformApi: { analyze } }))
import WaveformWorkbench from '../../components/WaveformWorkbench.vue'

const response = {
  duration_s: 4e-6, known_component_power_w: 750, total_power_w: 750, total_status: 'computed',
  components: { conduction: { status: 'computed', energy_j: 0, power_w: 0 }, turn_on: { status: 'computed', energy_j: .001, power_w: 250, event_count: 1, event_rate_hz: 250000 }, turn_off: { status: 'computed', energy_j: .002, power_w: 500, event_count: 1, event_rate_hz: 250000 } },
  signals: { time_s: [0, 4e-6], current_a: [0, 9], gate_v: [0, 1], device_voltage_v: [400, 2] },
  current_statistics: { mean: 4, rms: 5, method: 'Exact fixture' }, event_policy: { mode: 'gate' },
  conduction_plot: [{ time_s: 0, power_w: 0, cumulative_j: 0 }, { time_s: 4e-6, power_w: 90, cumulative_j: .0001 }],
  switching_cumulative_plot: [{ time_s: 0, energy_j: 0 }, { time_s: 4e-6, energy_j: .003 }],
  power_segments: [{ start_s: 0, end_s: 4e-6, power_coefficients_w: [0, 0, 90, 0] }],
  events: [], request: {}, assumptions: ['Independent fixture'],
}
describe('signed waveform workflow', () => {
  let wrapper
  beforeEach(() => { analyze.mockReset().mockResolvedValue(response); wrapper = mount(WaveformWorkbench, { global: { stubs: { CurvePlotter: true } } }) })
  afterEach(() => wrapper.unmount())
  async function submit() { await wrapper.get('form').trigger('submit'); await flushPromises() }
  it('rejects a cleared known condition before JSON transport', async () => {
    const label = wrapper.findAll('fieldset.model-editor details label').find(node => node.text().includes('Junction temperature'))
    await label.get('input[type=number]').setValue('')
    await submit()
    expect(analyze).not.toHaveBeenCalled()
    expect(wrapper.text()).toContain('Known condition t_j_c needs a finite number')
    expect(label.get('input[type=checkbox]').element.checked).toBe(true)
  })
  it('sends explicit model selection/omissions and renders returned SI traces', async () => {
    await submit()
    const sent = analyze.mock.calls[0][0]
    expect(sent.conduction.positive.selected).toBe(false)
    expect(sent.switching.turn_on).toEqual({ omit: true, omission_reason: 'Explicit conduction-only evaluation' })
    expect(wrapper.get('[data-testid=waveform-results]').text()).toContain('750.000')
    const plots = wrapper.findAllComponents({ name: 'CurvePlotter' })
    expect(plots).toHaveLength(6)
    expect(plots[0].props('datasets')[0].data).toEqual([{ x: 0, y: 0 }, { x: 4e-6, y: 9 }])
    expect(plots.every(plot => plot.props('allowZoom') === false)).toBe(true)
  })
  it('keeps linked visual ranges separate from the calculation interval', async () => {
    await submit()
    await wrapper.get('[data-testid=range-start]').setValue(1e-6)
    await wrapper.get('[data-testid=range-end]').setValue(3e-6)
    expect(analyze).toHaveBeenCalledTimes(1)
    for (const plot of wrapper.findAllComponents({ name: 'CurvePlotter' })) {
      expect(plot.props('xMin')).toBe(1e-6); expect(plot.props('xMax')).toBe(3e-6)
    }
    expect(wrapper.get('[data-testid=waveform-results]').text()).toContain('0.00000400000 s interval')
  })
  it('selects hysteretic events without changing gate samples and exposes returned state evidence', async () => {
    const eventSelect = wrapper.get('[aria-label="Event policy"]')
    await eventSelect.setValue('hysteresis')
    await wrapper.get('[data-testid=hysteresis-low]').setValue('-.5')
    await wrapper.get('[data-testid=hysteresis-high]').setValue('.5')
    const evidence = { method: 'strict-hysteresis-v1', threshold_scale: 'absolute', low_v: -.5, high_v: .5,
      initial_state: false, initial_state_source: 'declared', states: [false, true], turn_on_brackets: [0], turn_off_brackets: [],
      semantics: 'Strict crossings. Equality and deadband hold.' }
    analyze.mockResolvedValueOnce({ ...response, gate_digitization: evidence })
    await wrapper.get('[data-testid=hysteresis-initial]').setValue('off')
    await submit()
    expect(analyze.mock.calls.at(-1)[0].events).toEqual({ mode: 'hysteresis', digitizer: {
      thresholds: { scale: 'absolute', low_v: -.5, high_v: .5 }, initial_state: false } })
    expect(analyze.mock.calls.at(-1)[0].waveform.gate_v).toEqual([0, 0, 1, 1, 0])
    expect(wrapper.get('[data-testid=waveform-hysteresis-evidence]').text()).toContain('declared')
    expect(wrapper.get('[data-testid=waveform-gate-csv]').exists()).toBe(true)
    await wrapper.get('[data-testid=hysteresis-low]').setValue('-.25')
    expect(wrapper.find('[data-testid=waveform-results]').exists()).toBe(false)
  })
  it('holds a pending hysteresis result stale when calibrated bounds change', async () => {
    await wrapper.get('[aria-label="Event policy"]').setValue('hysteresis')
    await wrapper.get('[data-testid=hysteresis-scale]').setValue('relative')
    await wrapper.get('[data-testid=hysteresis-minimum]').setValue('0')
    await wrapper.get('[data-testid=hysteresis-maximum]').setValue('10')
    let resolve
    analyze.mockImplementationOnce(() => new Promise(done => { resolve = done }))
    await wrapper.get('form').trigger('submit')
    expect(analyze.mock.calls.at(-1)[0].events.digitizer.thresholds.minimum_v).toBe(0)
    await wrapper.get('[data-testid=hysteresis-maximum]').setValue('12')
    resolve(response); await flushPromises()
    expect(wrapper.find('[data-testid=waveform-results]').exists()).toBe(false)
  })
  it('adds spectrum policies only when selected and invalidates stale policy evidence', async () => {
    const fft = wrapper.findAll('label.check').find(node => node.text().includes('Compute uniform-sample FFT'))
    await fft.get('input').setValue(true)
    await submit()
    expect(analyze.mock.calls.at(-1)[0].spectrum).toEqual({ window: 'rectangular', detrend: 'none' })
    await wrapper.get('[data-testid=spectrum-mask-enabled]').setValue(true)
    await wrapper.get('[data-testid=spectrum-mask-source]').setValue('Known specification')
    await wrapper.get('[data-testid=spectrum-mask-csv]').setValue('frequency_hz,limit_peak\n0,1\n1,2')
    await wrapper.get('[data-testid=spectrum-bands-enabled]').setValue(true)
    await wrapper.get('[data-testid=spectrum-bands-csv]').setValue('name,lower_hz,upper_hz,endpoints\nFull,0,1,closed')
    let resolve
    analyze.mockImplementationOnce(() => new Promise(done => { resolve = done }))
    await wrapper.get('form').trigger('submit')
    expect(analyze.mock.calls.at(-1)[0].spectrum.mask).toMatchObject({ unit: 'A_peak', frequency_hz: [0, 1], limit_peak: [1, 2] })
    expect(analyze.mock.calls.at(-1)[0].spectrum.bands[0].endpoints).toBe('closed')
    await wrapper.get('[data-testid=spectrum-mask-source]').setValue('Changed specification')
    resolve(response); await flushPromises()
    expect(wrapper.find('[data-testid=waveform-results]').exists()).toBe(false)
    await submit()
    expect(wrapper.find('[data-testid=waveform-results]').exists()).toBe(true)
    await wrapper.get('[data-testid=spectrum-bands-csv]').setValue('name,lower_hz,upper_hz,endpoints\nFull,0,1,open')
    expect(wrapper.find('[data-testid=waveform-results]').exists()).toBe(false)
  })
  it('preserves original advanced numeric lexemes for authoritative backend validation', async () => {
    const advanced = wrapper.findAll('label.check').find(node => node.text().includes('Use JSON editor'))
    await advanced.get('input').setValue(true)
    const editor = wrapper.findAll('label').find(node => node.text().startsWith('Canonical request')).get('textarea')
    for (const raw of [
      '{"spectrum":{"mask":{"frequency_hz":[1.0000000000000000001],"limit_peak":[0]}}}',
      '{"spectrum":{"mask":{"frequency_hz":[1],"limit_peak":[-1e-400]}}}',
      '{"events":{"mode":"hysteresis","digitizer":{"thresholds":{"scale":"relative","low_fraction":0.25,"high_fraction":1.0000000000000000001,"minimum_v":0,"maximum_v":10}}}}',
    ]) {
      await editor.setValue(raw)
      analyze.mockRejectedValueOnce({ response: { data: { detail: 'Original numeric precision rejected' } } })
      await submit()
      expect(analyze.mock.calls.at(-1)[0]).toBe(raw)
      expect(wrapper.get('[role=alert]').text()).toContain('Original numeric precision rejected')
    }
  })
  it('invalidates completed and in-flight results on input edits', async () => {
    await submit()
    await wrapper.get('[data-testid=waveform-csv]').setValue('time_s,current_a\n0,1\n1,1')
    expect(wrapper.find('[data-testid=waveform-results]').exists()).toBe(false)
    let resolve
    analyze.mockImplementationOnce(() => new Promise(done => { resolve = done }))
    await submit()
    await wrapper.get('[data-testid=waveform-csv]').setValue('time_s,current_a\n0,2\n1,2')
    resolve(response); await flushPromises()
    expect(wrapper.find('[data-testid=waveform-results]').exists()).toBe(false)
  })
  it('preserves adjacent final CSV mask blanks and chooses a single delimiter', async () => {
    await wrapper.get('[data-testid=waveform-csv]').setValue('time_s;current_a;turn_on;turn_off\n0;0;1;0\n1;1;0;1\n2;0;;')
    const eventSelect = wrapper.findAll('select').find(select => select.text().includes('Explicit interval masks'))
    await eventSelect.setValue('masks')
    await submit()
    expect(analyze.mock.calls[0][0].events).toEqual({ mode: 'masks', turn_on: [true, false], turn_off: [false, true] })
    expect(wrapper.find('[role=alert]').exists()).toBe(false)
    await wrapper.get('[data-testid=waveform-csv]').setValue('time_s\tcurrent_a\tturn_on\tturn_off\n0\t0\t1\t0\n1\t1\t0\t1\n2\t0\t\t')
    await submit()
    expect(analyze.mock.calls.at(-1)[0].events).toEqual({ mode: 'masks', turn_on: [true, false], turn_off: [false, true] })
  })
  it('rejects missing numeric CSV values and clears structured errors after edits', async () => {
    await wrapper.get('[data-testid=waveform-csv]').setValue('time_s,current_a\n0,"1"\n1,"1"')
    await submit(); expect(analyze).not.toHaveBeenCalled(); expect(wrapper.get('[role=alert]').text()).toContain('quoted CSV is not supported')
    await wrapper.get('[data-testid=waveform-csv]').setValue('time_s,current_a\n0,\n1,1')
    await submit(); expect(analyze).not.toHaveBeenCalled(); expect(wrapper.get('[role=alert]').text()).toContain('missing samples')
    await wrapper.get('[data-testid=waveform-csv]').setValue('time_s,current_a\n0,1\n1,1')
    analyze.mockRejectedValueOnce({ response: { data: { detail: [{ loc: ['body', 'source'], msg: 'Required' }] } } })
    await submit(); expect(wrapper.get('[role=alert]').text()).toContain('source')
    await wrapper.get('[data-testid=waveform-csv]').setValue('time_s,current_a\n0,2\n1,2')
    expect(wrapper.find('[role=alert]').exists()).toBe(false)
  })
  it('retains import identity while clearly recording edits after import', async () => {
    const upload = wrapper.get('input[type=file]')
    Object.defineProperty(upload.element, 'files', { value: [{ name: 'measured.csv', text: async () => 'time_s,current_a\n0,1\n1,1' }] })
    await upload.trigger('change'); await flushPromises(); await submit()
    expect(analyze.mock.calls.at(-1)[0].provenance).toMatchObject({ imported_filename: 'measured.csv', edited_after_import: false, source: null })
    await wrapper.get('[data-testid=waveform-csv]').setValue('time_s,current_a\n0,2\n1,2')
    await submit()
    expect(analyze.mock.calls.at(-1)[0].provenance).toMatchObject({ imported_filename: 'measured.csv', edited_after_import: true })
    expect(wrapper.text()).toContain('(edited after import)')
  })
})
