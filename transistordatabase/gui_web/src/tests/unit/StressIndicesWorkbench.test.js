import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import StressIndicesWorkbench from '../../components/StressIndicesWorkbench.vue'

const { calculate, download } = vi.hoisted(() => ({ calculate: vi.fn(), download: vi.fn() }))
vi.mock('../../services/api.js', () => ({ stressApi: { calculate } }))
vi.mock('../../components/analysisFiles.js', async original => ({ ...await original(), downloadAnalysis: download }))
const metric = (value, unit, reason = null) => ({ value, unit, status: value === null ? 'unknown' : 'calculated', reason })
function response() {
  return { inputs: { source: 'Accepted Python source', conditions: { junction_temperature_c: null }, time_s: [0, 1], current_a: [0, 6], voltage_v: [400, 400], rds_on_ohm: .1, transitions: [{ time_s: .5, frequency_hz: 1000, kind: 'turn_on' }] },
    result: { current_rms_a: Math.sqrt(12), current_mean_square_a2: 12, peak_voltage_v: 400, duration_s: 1, sample_count: 2,
      metrics: { conduction: metric(12, 'A^2'), volume: metric(1920000, 'A^2 V^2'), switching: metric(3600000, 'Hz A^2 V'), cost: metric(.0000625, '1/(ohm V^2)') },
      transition_samples: [{ kind: 'turn_on', time_s: .5, frequency_hz: 1000, current_a: 3, voltage_v: 400, contribution: 3600000 }], definitions: { rms: 'Exact linear interpolation' }, scope: 'Stress proxies only' },
    evidence: { method: 'semiconductor-stress-v1', input_sha256: 'a'.repeat(64), result_sha256: 'b'.repeat(64) } }
}
let wrapper
const envelope = data => ({ data, rawJson: JSON.stringify(data) })
const field = name => wrapper.get(`[data-testid=stress-${name}]`)
async function fixture() { wrapper = mount(StressIndicesWorkbench); await field('example').trigger('click') }
async function submit() {
  await wrapper.get('form').trigger('submit')
  await vi.waitFor(() => expect(field('calculate').attributes('disabled')).toBeUndefined())
  await flushPromises()
}
beforeEach(() => { calculate.mockReset(); download.mockReset(); calculate.mockResolvedValue(envelope(response())) })
afterEach(() => wrapper?.unmount())

describe('semiconductor stress indices', () => {
  it('sends explicit SI samples and event weights, and presents the Python results and provenance', async () => {
    await fixture(); await field('condition-junction_temperature_c').setValue('25'); await submit()
    expect(calculate).toHaveBeenCalledWith(expect.objectContaining({ time_s: [0, 1], current_a: [0, 6], voltage_v: [400, 400], rds_on_ohm: .1,
      conditions: { junction_temperature_c: 25, gate_on_voltage_v: null, gate_off_voltage_v: null, dc_bus_voltage_v: null }, transitions: [{ time_s: .5, frequency_hz: 1000, kind: 'turn_on' }] }))
    expect(field('metric-conduction').text()).toContain('12.0000 A²')
    expect(field('metric-volume').text()).toContain('1.92000e+6 A² V²')
    expect(field('metric-switching').text()).toContain('3.60000e+6 Hz A² V')
    expect(field('metric-cost').text()).toContain('0.0000625000 1/(ohm V²)')
    expect(field('transition-results').text()).toContain('3.00000')
    expect(field('accepted-source').text()).toBe('Accepted Python source')
    expect(field('hash').text()).toContain('a'.repeat(64))
  })

  it('keeps unknown transition and resistance evidence separate from an explicit zero', async () => {
    await fixture(); await field('transition-mode').setValue('unknown'); await field('resistance').setValue('')
    const unknown = response(); unknown.inputs.transitions = null; unknown.inputs.rds_on_ohm = null
    unknown.result.metrics.switching = metric(null, 'Hz A^2 V', 'Transition evidence is unknown.'); unknown.result.metrics.cost = metric(null, '1/(ohm V^2)', 'On resistance is unknown.'); unknown.result.transition_samples = []
    calculate.mockResolvedValueOnce(envelope(unknown)); await submit()
    expect(calculate.mock.calls[0][0]).toMatchObject({ transitions: null, rds_on_ohm: null })
    expect(field('metric-switching').text()).toContain('Unavailable'); expect(field('metric-cost').text()).toContain('Unavailable')
    const zero = response(); zero.result.metrics.switching = metric(0, 'Hz A^2 V'); zero.inputs.transitions = []; zero.result.transition_samples = []
    calculate.mockResolvedValueOnce(envelope(zero)); await field('transition-mode').setValue('none'); await submit()
    expect(calculate.mock.calls[1][0].transitions).toEqual([])
    expect(field('metric-switching').text()).toContain('0.00000'); expect(field('metric-switching').text()).toContain('calculated')
  })

  it('preserves signed input samples and negative returned stress values', async () => {
    await fixture(); await field('samples').setValue('time_s,current_a,voltage_v\n0,-3,-400\n1,-3,-200')
    const signed = response(); signed.result.peak_voltage_v = -200; signed.result.metrics.switching = metric(-2700000, 'Hz A^2 V'); calculate.mockResolvedValueOnce(envelope(signed))
    await submit(); expect(calculate.mock.calls[0][0].current_a).toEqual([-3, -3]); expect(calculate.mock.calls[0][0].voltage_v).toEqual([-400, -200])
    expect(field('metric-switching').text()).toContain('-2.70000e+6'); expect(wrapper.text()).toContain('Maximum signed voltage: -200.000 V')
  })

  it('rejects absent source, malformed SI CSV, blank cells and nonfinite samples before transport', async () => {
    wrapper = mount(StressIndicesWorkbench); await submit(); expect(wrapper.get('[role=alert]').text()).toContain('Declare the waveform source')
    await field('example').trigger('click')
    for (const csv of ['t,i,v\n0,0,400\n1,6,400', 'time_s,current_a,voltage_v\n0,,400\n1,6,400', 'time_s,current_a,voltage_v\n0,NaN,400\n1,6,400', 'time_s,current_a,voltage_v\n0,0x10,400\n1,6,400', 'time_s,current_a,voltage_v\n0,0b10,400\n1,6,400', 'time_s,current_a,voltage_v\n0,1e-400,400\n1,6,400', 'time_s,current_a,voltage_v\n0,0,400']) {
      await field('samples').setValue(csv); await submit(); expect(wrapper.find('[role=alert]').exists()).toBe(true)
    }
    expect(calculate).not.toHaveBeenCalled()
  })

  it('requires actual records for supplied transitions and preserves explicit decimal zero', async () => {
    await fixture(); await field('transitions').setValue('time_s,frequency_hz,kind'); await submit()
    expect(calculate).not.toHaveBeenCalled(); expect(wrapper.get('[role=alert]').text()).toContain('at least one transition record')
    await field('transitions').setValue('time_s,frequency_hz,kind\n0,0e-400,turn_on')
    await field('samples').setValue('time_s,current_a,voltage_v\n0,-0e-400,400\n1,+6.0e+0,400'); await submit()
    expect(calculate.mock.calls[0][0].transitions[0].frequency_hz).toBe(0)
    expect(calculate.mock.calls[0][0].current_a).toEqual([-0, 6])
  })

  it('clears accepted exports on all edits and ignores late responses and late errors', async () => {
    await fixture(); await submit(); await field('source').setValue('New source'); expect(wrapper.find('[data-testid=stress-results]').exists()).toBe(false)
    let release; calculate.mockImplementationOnce(() => new Promise(resolve => { release = resolve })); await wrapper.get('form').trigger('submit')
    await vi.waitFor(() => expect(release).toBeTypeOf('function'))
    await field('condition-gate_on_voltage_v').setValue('15'); release(envelope(response())); await flushPromises()
    expect(wrapper.find('[data-testid=stress-results]').exists()).toBe(false); expect(field('stale').text()).toContain('Inputs changed')
    let reject; calculate.mockImplementationOnce(() => new Promise((_, fail) => { reject = fail })); await wrapper.get('form').trigger('submit')
    await vi.waitFor(() => expect(reject).toBeTypeOf('function'))
    await field('resistance').setValue('.2'); reject(new Error('Old request failed')); await flushPromises(); expect(wrapper.find('[role=alert]').exists()).toBe(false)
    expect(field('calculate').attributes('disabled')).toBeUndefined()
  })

  it('ignores a response after the workbench is closed and surfaces current API validation errors', async () => {
    await fixture(); calculate.mockRejectedValueOnce({ response: { data: { detail: 'time_s must increase strictly.' } } }); await submit()
    expect(wrapper.get('[role=alert]').text()).toContain('increase strictly')
    let release; calculate.mockImplementationOnce(() => new Promise(resolve => { release = resolve })); await wrapper.get('form').trigger('submit')
    await vi.waitFor(() => expect(release).toBeTypeOf('function'))
    wrapper.unmount(); release(envelope(response())); await flushPromises()
    expect(download).not.toHaveBeenCalled()
  })

  it('downloads the exact response and an evidence-complete CSV with safe source text', async () => {
    await fixture(); const data = response(); data.inputs.source = '=user source'; data.result.metrics.cost = metric(null, '1/(ohm V^2)', 'Unknown resistance.')
    const rawJson = JSON.stringify(data).replace('[0,1]', '[0.0,1.0]').replace('[400,400]', '[400.0,400.0]')
    calculate.mockResolvedValueOnce({ data, rawJson }); await submit()
    await field('json').trigger('click'); expect(download).toHaveBeenCalledWith(rawJson, 'semiconductor-stress-indices.json', 'application/json;charset=utf-8')
    await field('csv').trigger('click'); const csv = download.mock.calls[1][0]
    expect(csv).toContain('"\'=user source"'); expect(csv).toContain('"cost","null","1/(ohm V^2)","unknown"')
    for (const text of ['time_s', 'current_a', 'voltage_v', 'transitions', 'conditions', 'semiconductor-stress-v1', 'a'.repeat(64)]) expect(csv).toContain(text)
  })
})
