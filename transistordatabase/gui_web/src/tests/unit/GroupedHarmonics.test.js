import { afterEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
const { analyze, grid } = vi.hoisted(() => ({ analyze: vi.fn(), grid: vi.fn() }))
vi.mock('../../services/api.js', () => ({ waveformApi: { analyze }, analysisApi: { grid } }))
import WaveformWorkbench from '../../components/WaveformWorkbench.vue'
import WaveformAnalysisView from '../../components/WaveformAnalysisView.vue'
import GridWorkbench from '../../components/GridWorkbench.vue'
import GroupedSeriesEvidence from '../../components/GroupedSeriesEvidence.vue'

let wrapper
afterEach(() => { wrapper?.unmount(); vi.restoreAllMocks(); vi.unstubAllGlobals(); vi.useRealTimers(); vi.clearAllMocks() })
const options = { global: { stubs: { CurvePlotter: true, GateAllocationPanel: true, WaveformThermalTransfer: true } } }
const fixture = {
  signals: { time_s: [0, 1, 2, 3], current_a: [2, 2, 2, -2] }, components: {}, known_component_power_w: 0,
  duration_s: 3, total_status: 'partial-explicit-omissions', total_power_w: null,
  current_statistics: { mean: 4 / 3, rms: Math.sqrt(28 / 9), method: 'Exact piecewise-linear' },
  power_segments: [], events: [], event_policy: { mode: 'omitted' }, assumptions: [], request: {},
  spectrum: { frequency_hz: [0, .25, .5], amplitude_a: [1, 2, 1], real_a: [1, 0, 1], imaginary_a: [0, -2, 0],
    phase_rad: [0, -Math.PI / 2, 0], selected_bin_indices: [1], relative_threshold: .5,
    phase_reference: 'first supplied sample', discrete_input_rms_a: 2, window: 'rectangular', detrend: 'none' },
}
const metadata = { configuration_id: 'config', configuration_source: '=source', operating_point_id: 'point',
  operating_point_source: 'Independent sample fixture', weight: 0, series: 'current_a', unit: 'A', status: 'computed', pair_status: 'computed',
  error: null, input_sha256: 'pair-hash', grid_input_sha256: 'grid-hash', method_version: 'grouped-v1',
  window: 'rectangular', detrend: 'none', relative_threshold: 0 }
const time = { ...metadata, sample_count: 4, minimum: -2, maximum: 2, arithmetic_mean: 1,
  sample_standard_deviation: 2, discrete_rms: 2, time_weighted_mean: 4 / 3, time_weighted_rms: Math.sqrt(28 / 9) }
const frequency = { ...metadata, retained_bin_count: 1, bin_index: 1, frequency_hz: .25, magnitude: 2, real: 0, imaginary: -2, phase_rad: -Math.PI / 2 }

describe('physical series harmonics', () => {
  it('sends finite thresholds including zero and keeps grouped exports opt-in', async () => {
    analyze.mockResolvedValue({}); grid.mockRejectedValue(new Error('Request inspected without a result fixture'))
    wrapper = mount(WaveformWorkbench, { global: { stubs: { WaveformAnalysisView: true, GateAllocationPanel: true } } })
    await wrapper.findAll('label').find(node => node.text().includes('Compute uniform-sample FFT')).get('input').setValue(true)
    await wrapper.get('form').trigger('submit'); await flushPromises()
    expect(analyze.mock.calls[0][0].spectrum).toEqual({ window: 'rectangular', detrend: 'none' })
    await wrapper.get('[aria-label="Current harmonic threshold"]').setValue('0')
    await wrapper.get('form').trigger('submit'); await flushPromises()
    expect(analyze.mock.calls.at(-1)[0].spectrum.relative_threshold).toBe(0)
    for (const value of ['', '-.1', '1.1']) {
      await wrapper.get('[aria-label="Current harmonic threshold"]').setValue(value)
      await wrapper.get('form').trigger('submit'); await flushPromises()
    }
    expect(analyze).toHaveBeenCalledTimes(2)
    wrapper.unmount()
    wrapper = mount(GridWorkbench, { global: { stubs: { WaveformAnalysisView: true } } })
    await wrapper.get('form').trigger('submit'); await flushPromises()
    expect(grid.mock.calls[0][0].grouped_series_exports).toBe(false)
    expect(grid.mock.calls[0][0].operating_points[0].spectrum).toEqual({ window: 'rectangular', detrend: 'none' })
    await wrapper.findAll('label').find(node => node.text().includes('Include grouped series statistics')).get('input').setValue(true)
    await wrapper.get('[aria-label="Point 1 harmonic threshold"]').setValue('0')
    await wrapper.get('form').trigger('submit'); await flushPromises()
    expect(grid.mock.calls.at(-1)[0]).toMatchObject({ grouped_series_exports: true, operating_points: [{ spectrum: { relative_threshold: 0 } }] })
  })
  it('shows returned complex current components with signed phase and preserves the full amplitude plot', async () => {
    wrapper = mount(WaveformAnalysisView, { ...options, props: { result: fixture } })
    const table = wrapper.get('[data-testid=current-harmonics]')
    expect(table.findAll('tbody tr')).toHaveLength(1)
    expect(table.findAll('tbody td').map(cell => cell.text())).toEqual(['1', '0.250000', '2.00000', '0.00000', '-2.00000', '-1.57080'])
    expect(table.text()).toContain('Real (A)'); expect(table.text()).toContain('Phase (rad)')
    const plot = wrapper.findAllComponents({ name: 'CurvePlotter' }).at(-1)
    expect(plot.props('datasets')[0].data).toEqual([{ x: 0, y: 1 }, { x: .25, y: 2 }, { x: .5, y: 1 }])
    expect(plot.props('cursorX')).toBe(null)
    await wrapper.setProps({ result: { ...fixture, spectrum: { ...fixture.spectrum, selected_bin_indices: [0], phase_rad: [null, 0, 0], real_a: [0, 0, 0] } } })
    expect(wrapper.get('[data-testid=current-harmonics]').findAll('tbody td').at(-1).text()).toBe('—')
  })
  it('discards grouped exports after threshold edits and late replies', async () => {
    const response = { metric: { name: 'conduction_power_w', unit: 'W' }, inputs: { configurations: [], operating_points: [] },
      rows: [], ranking: [], summaries: [], grouped_series_exports: { time_rows: [time], frequency_rows: [frequency] } }
    let resolve
    grid.mockImplementationOnce(() => new Promise(done => { resolve = done }))
    wrapper = mount(GridWorkbench, { global: { stubs: { WaveformAnalysisView: true, GroupedSeriesEvidence: true } } })
    await wrapper.get('form').trigger('submit')
    await wrapper.get('[aria-label="Point 1 harmonic threshold"]').setValue('0')
    resolve(response); await flushPromises()
    expect(wrapper.find('[data-testid=grid-results]').exists()).toBe(false)
    grid.mockResolvedValueOnce(response)
    await wrapper.get('form').trigger('submit'); await flushPromises()
    expect(wrapper.findComponent(GroupedSeriesEvidence).exists()).toBe(true)
    await wrapper.get('[aria-label="Point 1 harmonic threshold"]').setValue('.5')
    expect(wrapper.findComponent(GroupedSeriesEvidence).exists()).toBe(false)
  })
  it('bounds grouped tables and downloads the complete raw server projections without changing zero or null', async () => {
    vi.useFakeTimers()
    const projection = { method_version: 'grouped-v1', scope: 'Every accepted pair', null_csv: 'NULL',
      time_rows: Array.from({ length: 31 }, (_, i) => ({ ...time, operating_point_id: `p-${i}` })),
      frequency_rows: [frequency, { ...frequency, status: 'not_requested', phase_rad: null, real: null }],
      time_csv: 'source,weight\r\n\'=source,0', frequency_csv: 'real,phase_rad\r\n0,NULL' }
    projection.time_json = JSON.stringify(projection.time_rows)
    projection.frequency_json = JSON.stringify(projection.frequency_rows).replace('"real":0', '"real":-0.0')
    const blobs = []; vi.stubGlobal('Blob', class { constructor(parts, options) { this.parts = parts; this.type = options.type } })
    vi.stubGlobal('URL', { createObjectURL: vi.fn(blob => { blobs.push(blob); return 'blob:test' }), revokeObjectURL: vi.fn() })
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {})
    wrapper = mount(GroupedSeriesEvidence, { props: { projection } })
    expect(wrapper.get('[data-testid=grouped-time-table]').findAll('tbody tr')).toHaveLength(25)
    await wrapper.findAll('button').find(button => button.text() === 'Next time rows').trigger('click')
    expect(wrapper.get('[data-testid=grouped-time-table]').findAll('tbody tr')).toHaveLength(6)
    expect(wrapper.text()).toContain('Discrete sample RMS'); expect(wrapper.text()).toContain('Piecewise-linear time RMS')
    expect(wrapper.get('[data-testid=grouped-frequency-table]').text()).toContain('not_requested')
    expect(wrapper.get('[data-testid=grouped-frequency-table]').text()).toContain('Pair: computed')
    for (const name of ['Download grouped time JSON', 'Download grouped time CSV', 'Download grouped frequency JSON', 'Download grouped frequency CSV']) {
      await wrapper.findAll('button').find(button => button.text() === name).trigger('click')
    }
    expect(JSON.parse(blobs[0].parts[0])).toEqual(projection.time_rows)
    expect(blobs[0].parts[0]).toBe(projection.time_json)
    expect(blobs[1].parts[0]).toBe(projection.time_csv)
    expect(blobs[2].parts[0]).toBe(projection.frequency_json)
    expect(JSON.parse(blobs[2].parts[0])[1]).toEqual(projection.frequency_rows[1])
    expect(blobs[2].parts[0]).toContain('"real":-0.0')
    expect(blobs[3].parts[0]).toBe(projection.frequency_csv)
    await wrapper.setProps({ projection: { ...projection } })
    expect(wrapper.get('[data-testid=grouped-time-table]').findAll('tbody tr')).toHaveLength(25)
    vi.runAllTimers()
  })
})
