import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
const { api } = vi.hoisted(() => ({ api: { capabilities: vi.fn(), preview: vi.fn(), extract: vi.fn() } }))
vi.mock('../../services/api.js', () => ({ measurementApi: api }))
vi.mock('../../components/CurvePlotter.vue', () => ({ default: { props: ['datasets', 'xMin', 'xMax'], template: '<div class="chart-stub" />' } }))
import MeasurementWorkbench from '../../components/MeasurementWorkbench.vue'
const source = { filename: '=source.csv', sha256: 'ab'.repeat(32), source_kind: 'unknown', disclosure: null, parser: 'python.csv', parser_version: 'stdlib' }
const wave = { time_s: [0, 1e-6, 2e-6], voltage_v: [100, 50, 0], current_a: [0, -2, -4], gate_v: [0, 5, 10] }
const discovery = { source, available: { traces: ['t', 'v', 'i', 'g'], steps: [0], measurements: ['eoff'], step_variables: ['idc', 'vdc'] }, waveform: null, log_records: [], warnings: [] }
const preview = { ...discovery, waveform: wave, statistics: { current_a: { mean: -2, rms: 2.3, minimum: -4, maximum: 0 } } }
const analysis = { e_on: 1e-6, e_off: 2e-6, i_on: 4, i_off: 3, v_overshoot: 100, energy_window_convention: 'Local 1–99% union', event_evidence: { turn_on: { gate_edge_s: 1e-6, integration_window_s: [1e-6, 2e-6], local_current_low_a: 0, local_current_high_a: 4 } } }
const result = { ...preview, status: 'review-not-persisted', analysis, points: { turn_on: [{ e_x: 1e-6, i_x: 4 }], turn_off: [{ e_x: 2e-6, i_x: 3 }] }, raw_record: { gate_v: wave.gate_v }, core_json_document: { schema: 'test' }, core_json_text: '{"large":18446744073709551615}' }
let wrapper, blobs, names
const field = id => wrapper.get(`[data-testid="measurement-${id}"]`)
async function upload(id = 'file', name = '=source.csv', bytes = [239, 187, 191, 34, 116, 34]) {
  Object.defineProperty(field(id).element, 'files', { configurable: true, value: [{ name, size: bytes.length, arrayBuffer: async () => Uint8Array.from(bytes).buffer }] })
  await field(id).trigger('change'); await flushPromises()
}
async function map(gate = true) {
  await field('inspect').trigger('click'); await flushPromises()
  for (const [key, name, unit] of [['time', 't', 'us'], ['voltage', 'v', 'V'], ['current', 'i', 'A'], ...(gate ? [['gate', 'g', 'V']] : [])]) { await field('map-' + key).setValue(name); await field('unit-' + key).setValue(unit) }
}
async function declare() { for (const [key,value] of Object.entries({ bus_voltage_v: 100, v_g_on: 10, v_g_off: 0 })) { await field('known-' + key).setValue('true'); await field('condition-' + key).setValue(String(value)) } }
async function mappedPreview() {
  api.preview.mockResolvedValue(preview)
  await flushPromises()
  expect(field('preview').element.disabled).toBe(false)
  await field('preview').trigger('click'); await flushPromises()
  expect(wrapper.find('[role=alert]').exists()).toBe(false)
  expect(field('analyze').element.disabled).toBe(false)
}
async function analyze() { await field('analyze').trigger('click'); await flushPromises() }
beforeEach(() => {
  for (const method of Object.values(api)) method.mockReset()
  api.capabilities.mockResolvedValue({ csv: true, log: true, raw: true }); api.preview.mockResolvedValue(discovery); api.extract.mockResolvedValue(result)
  blobs = []; names = []
  vi.spyOn(URL, 'createObjectURL').mockImplementation(blob => { blobs.push(blob); return 'blob:measurement' })
  vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {})
  vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function () { names.push(this.download) })
  wrapper = mount(MeasurementWorkbench)
})
afterEach(() => { wrapper.unmount(); vi.restoreAllMocks() })
describe('measurement import workflow', () => {
  it('preserves original bytes, explicit mapping and zero known gate rail through Python review and native text download', async () => {
    await upload(); await map(); await declare(); await field('polarity-current').setValue('-1'); await mappedPreview(); await analyze()
    expect(api.preview.mock.calls[0][0]).toMatchObject({ content_base64: '77u/InQi', filename: '=source.csv', mappings: { time: null, voltage: null, current: null, gate: null } })
    const request = api.extract.mock.calls[0][0]
    expect(request.mappings.current).toEqual({ name: 'i', unit: 'A', polarity: -1 })
    expect(request.conditions).toMatchObject({ t_j: null, v_g_off: 0, bus_voltage_v: 100, r_g_on: null })
    expect(field('samples').text()).toContain('-2.0000000'); expect(field('analysis').text()).toContain('Absolute device voltage peak')
    expect(field('statistics').text()).toContain('Signed current (A)')
    await field('native-download').trigger('click'); expect(await blobs.at(-1).text()).toBe(result.core_json_text)
    expect(names.at(-1)).toBe('measurement-native-device.json')
  })
  it('rejects blank known values and missing gate extraction while allowing gate-absent source preview', async () => {
    await upload(); await map(false); api.preview.mockResolvedValue({ ...preview, waveform: { ...wave, gate_v: null } }); await field('preview').trigger('click'); await flushPromises()
    expect(field('samples').text()).toContain('Unknown'); await analyze(); expect(api.extract).not.toHaveBeenCalled(); expect(wrapper.text()).toContain('Map a gate waveform')
    await field('map-gate').setValue('g'); await field('unit-gate').setValue('V'); await declare(); await field('known-t_j').setValue('true'); await mappedPreview(); await analyze()
    expect(api.extract).not.toHaveBeenCalled(); expect(wrapper.text()).toContain('Junction temperature (°C) requires a finite value.')
  })
  it('ignores held responses and removes results/exports after mapping, source or test condition edits', async () => {
    await upload(); await map(); await declare(); await mappedPreview()
    let release; api.extract.mockImplementation(() => new Promise(resolve => { release = resolve }))
    await field('analyze').trigger('click'); await field('condition-v_g_on').setValue('12'); release(result); await flushPromises()
    expect(wrapper.find('[data-testid=measurement-result]').exists()).toBe(false)
    expect(field('analyze').attributes('disabled')).toBeDefined()
    api.extract.mockResolvedValue(result); await mappedPreview(); await analyze(); await field('map-current').setValue('v')
    expect(wrapper.find('[data-testid=measurement-result]').exists()).toBe(false)
    await upload('file','replacement.csv',[1,2,3]); expect(wrapper.find('[data-testid=measurement-hash]').exists()).toBe(false)
  })
  it('retains native input as untouched JSON text and prevents stale source-kind disclosure display', async () => {
    await upload(); await map(); const text = '{"large":18446744073709551615}'
    await upload('native-file','native.json',[...new TextEncoder().encode(text)]); await declare(); await mappedPreview(); await analyze()
    expect(api.extract.mock.calls[0][0].device_document_text).toBe(text)
    await field('source-kind').setValue('simulation'); expect(wrapper.find('[data-testid=measurement-hash]').exists()).toBe(false)
  })
  it('distinguishes all supplied log integrals from one selected derived step and makes textual CSV cells safe', async () => {
    await field('format').setValue('log'); await upload(); await field('inspect').trigger('click'); await flushPromises()
    await field('log-energy_off').setValue('eoff'); await field('log-energy-unit').setValue('J'); await field('step-index').setValue('2')
    const log = { ...discovery, log_records: [{ event: 'turn_off', step_index: 2, status: 'supplied-integral', energy_j: 2.69621e-6, current_a: -5, bus_voltage_v: 5, expression: '=integral', window_s: [39e-6,41e-6] }] }
    api.preview.mockResolvedValue(log); await field('preview').trigger('click'); await flushPromises()
    api.extract.mockResolvedValue({ ...log, status: 'review-not-persisted', analysis: null, points: {}, core_json_document: null }); await analyze()
    expect(api.extract.mock.calls[0][0].step_index).toBe(2); expect(field('log-records').text()).toContain('0.0000026962100')
    await field('csv-download').trigger('click'); const csv = await blobs.at(-1).text(); expect(csv).toContain('"\'=integral"'); expect(csv).toContain('"-5"')
    await field('review-download').trigger('click'); expect(JSON.parse(await blobs.at(-1).text()).log_records[0].expression).toBe('=integral')
  })
})


describe('measurement selection and linked views', () => {
  it('keeps RAW step selection empty during discovery and requires a declared index before mapped preview', async () => {
    await field('format').setValue('raw'); await upload(); await map()
    expect(api.preview.mock.calls[0][0].step_index).toBeNull()
    expect(field('step-index').element.value).toBe('')
    await field('preview').trigger('click'); await flushPromises()
    expect(wrapper.get('[role=alert]').text()).toContain('Step index requires a finite value.')
    expect(api.preview).toHaveBeenCalledTimes(1)
    await field('step-index').setValue('0'); await mappedPreview()
    expect(api.preview.mock.calls.at(-1)[0].step_index).toBe(0)
  })
  it('shares display time bounds without replacing plotted datasets or invalidating scientific results', async () => {
    await upload(); await map(); await declare(); await mappedPreview(); await analyze()
    const charts=wrapper.findAllComponents({name:'CurvePlotter'}), datasets=charts.map(chart=>chart.props('datasets')); expect(charts).toHaveLength(3)
    await field('range-start').setValue('0.0000005'); await field('range-end').setValue('0.0000015')
    for(let index=0;index<charts.length;index++) {expect(charts[index].props('datasets')).toBe(datasets[index]);expect(charts[index].props('xMin')).toBe(.5e-6);expect(charts[index].props('xMax')).toBe(1.5e-6)}
    expect(wrapper.find('[data-testid=measurement-result]').exists()).toBe(true)
  })
})

