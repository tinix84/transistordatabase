import { describe, expect, it, vi } from 'vitest'
import { mount } from '@vue/test-utils'
const { downloadAnalysis } = vi.hoisted(() => ({ downloadAnalysis: vi.fn() }))
vi.mock('../../components/analysisFiles.js', async importOriginal => ({ ...await importOriginal(), downloadAnalysis }))
import { defaultSpectrumAnalysis, spectrumAnalysisRequest } from '../../components/spectrumAnalysis.js'
import SpectrumAnalysisControls from '../../components/SpectrumAnalysisControls.vue'
import SpectrumAnalysisView from '../../components/SpectrumAnalysisView.vue'

describe('declared spectrum inputs', () => {
  it('omits all analysis fields unless explicitly enabled', () => {
    expect(spectrumAnalysisRequest(defaultSpectrumAnalysis())).toEqual({})
  })
  it('preserves null gaps, exact coordinates, source, interpolation and band endpoints', () => {
    const request = spectrumAnalysisRequest({ ...defaultSpectrumAnalysis(), maskEnabled: true, maskSource: ' Bench specification ',
      maskCsv: 'frequency_hz,limit_peak\n0,0\n1.0,\n2e0,1.5', interpolation: 'linear-frequency',
      bandsEnabled: true, bandsCsv: 'name,lower_hz,upper_hz,endpoints\nSwitching,1,2,right-closed' })
    expect(request.mask).toEqual({ unit: 'A_peak', frequency_hz: [0, 1, 2], limit_peak: [0, null, 1.5], source: 'Bench specification', interpolation: 'linear-frequency' })
    expect(request.bands).toEqual([{ name: 'Switching', lower_hz: 1, upper_hz: 2, endpoints: 'right-closed' }])
  })
  it.each(['1.00000000000000000001', '9007199254740993', '1e-400', '-1', 'NaN'])('rejects unsupported original coordinate %s before transport', frequency => {
    expect(() => spectrumAnalysisRequest({ ...defaultSpectrumAnalysis(), maskEnabled: true, maskSource: 'Known', maskCsv: `frequency_hz,limit_peak\n${frequency},1` })).toThrow()
  })
  it.each(['frequency_hz,limit_peak\n1,1\n1,2', 'frequency_hz,limit_peak\n2,1\n1,2', 'frequency_hz,limit_peak\n1,-1', 'frequency_hz,limit_peak\n1,1e-400'])('rejects malformed or lossy limits and coordinate ordering', maskCsv => {
    expect(() => spectrumAnalysisRequest({ ...defaultSpectrumAnalysis(), maskEnabled: true, maskSource: 'Known', maskCsv })).toThrow()
  })
  it('requires a source and bounds the mask without expanding pairs', () => {
    expect(() => spectrumAnalysisRequest({ ...defaultSpectrumAnalysis(), maskEnabled: true })).toThrow('source')
    expect(() => spectrumAnalysisRequest({ ...defaultSpectrumAnalysis(), maskEnabled: true, maskSource: 'Known', maskCsv: 'frequency_hz,limit_peak\n' + Array.from({ length: 5002 }, (_, i) => `${i},1`).join('\n') })).toThrow('5001')
  })
  it('rejects unnamed/duplicate bands and undeclared endpoint policy', () => {
    for (const row of [',1,2,closed', 'a,2,1,closed', 'a,1,2,nearest', 'a,1,2,closed\na,2,3,closed']) {
      expect(() => spectrumAnalysisRequest({ ...defaultSpectrumAnalysis(), bandsEnabled: true, bandsCsv: 'name,lower_hz,upper_hz,endpoints\n' + row })).toThrow()
    }
  })
  it('controls emit a new form object and disclose the two normalization bases', async () => {
    const initial = defaultSpectrumAnalysis()
    const wrapper = mount(SpectrumAnalysisControls, { props: { modelValue: initial } })
    await wrapper.get('[data-testid=spectrum-mask-enabled]').setValue(true)
    expect(wrapper.emitted('update:modelValue')[0][0]).toEqual({ ...initial, maskEnabled: true })
    expect(initial.maskEnabled).toBe(false)
    await wrapper.setProps({ modelValue: { ...initial, maskEnabled: true, bandsEnabled: true } })
    expect(wrapper.text()).toContain('Equality passes')
    expect(wrapper.text()).toContain('without an implicit correction')
    wrapper.unmount()
  })
})

const basis = { observed_bin_count: 120, window: 'hann', detrend: 'none', coherent_gain: .4, window_power_gain: .3,
  mask_basis: 'Coherent-gain-corrected peak', band_basis: 'Raw windowed discrete Parseval RMS', band_overlap_policy: 'Not additive' }
describe('complete spectrum evidence exports', () => {
  it('exports every bin beyond the bounded table, including unknowns and below-top-ten failures', async () => {
    downloadAnalysis.mockClear()
    const bins = Array.from({ length: 120 }, (_, i) => ({ bin_index: i, frequency_hz: i, amplitude_peak: i === 115 ? .2 : 0,
      limit_peak: i === 110 ? null : .1, margin_peak: i === 110 ? null : i === 115 ? -.1 : .1,
      excess_peak: i === 110 ? null : i === 115 ? .1 : 0, status: i === 110 ? 'uncovered' : i === 115 ? 'fail' : 'pass', coverage: i === 110 ? 'gap' : 'exact' }))
    const spectrum = { analysis_basis: basis, mask_screening: { bins, unit: 'A_peak', source: 'Declared fixture', interpolation: 'exact',
      status: 'fail', covered_bin_count: 119, observed_bin_count: 120, violating_bin_indices: [115], uncovered_bin_indices: [110],
      comparison: 'Strict comparison; equality passes', margin_convention: 'Limit minus observed; negative violates', scope: 'Observed bins only' } }
    const wrapper = mount(SpectrumAnalysisView, { props: { spectrum }, global: { stubs: { CurvePlotter: true } } })
    expect(wrapper.findAll('tbody tr')).toHaveLength(100)
    expect(wrapper.get('[data-testid=spectrum-mask-failing-bins]').text()).toContain('115')
    expect(wrapper.findComponent({ name: 'CurvePlotter' }).props('datasets')[0].data).toHaveLength(120)
    await wrapper.get('[data-testid=spectrum-mask-download]').trigger('click')
    expect(downloadAnalysis.mock.calls[0][0].split('\r\n')).toHaveLength(121)
    expect(downloadAnalysis.mock.calls[0][0]).toContain('"115","115","0.2"')
    expect(downloadAnalysis.mock.calls[0][0]).toContain('Declared fixture')
    await wrapper.get('[data-testid=spectrum-analysis-download]').trigger('click')
    expect(JSON.parse(downloadAnalysis.mock.calls[1][0])).toEqual(spectrum)
    wrapper.unmount()
  })
  it('retains tiny RMS and unavailable empty bands in bands-only JSON/CSV', async () => {
    downloadAnalysis.mockClear()
    const spectrum = { analysis_basis: basis, band_rms: [
      { name: 'Tiny', lower_hz: 0, upper_hz: 1, endpoints: 'closed', rms: 1e-200, unit: 'A_rms', status: 'computed', bin_count: 2 },
      { name: 'Empty', lower_hz: .1, upper_hz: .2, endpoints: 'closed', rms: null, unit: 'A_rms', status: 'no-bins', bin_count: 0 }] }
    const wrapper = mount(SpectrumAnalysisView, { props: { spectrum }, global: { stubs: { CurvePlotter: true } } })
    expect(wrapper.text()).toContain('1.00000e-200')
    expect(wrapper.text()).toContain('Unavailable A_rms')
    await wrapper.get('[data-testid=spectrum-bands-download]').trigger('click')
    expect(downloadAnalysis.mock.calls[0][0]).toContain('1e-200')
    expect(downloadAnalysis.mock.calls[0][0]).toContain('Raw windowed discrete Parseval RMS')
    await wrapper.get('[data-testid=spectrum-analysis-download]').trigger('click')
    expect(JSON.parse(downloadAnalysis.mock.calls[1][0])).toEqual(spectrum)
    wrapper.unmount()
  })
})
