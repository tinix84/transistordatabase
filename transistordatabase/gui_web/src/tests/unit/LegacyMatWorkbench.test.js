import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
const { api } = vi.hoisted(() => ({ api: { preview: vi.fn(), array: vi.fn(), export: vi.fn(), finalize: vi.fn(), calculate: vi.fn() } }))
vi.mock('../../services/api.js', () => ({ matApi: api }))
vi.mock('../../components/CurvePlotter.vue', () => ({ default: { props: ['datasets'], template: '<div class="chart-stub" />' } }))
import LegacyMatWorkbench from '../../components/LegacyMatWorkbench.vue'
import LegacyMatField from '../../components/LegacyMatField.vue'

const sourceHash = 'ab'.repeat(32), branch = 'forwardConductionTrace'
const mechanisms = ['forwardConductionTrace', 'forwardTurnOnTrace', 'forwardTurnOffTrace', 'reverseConductionTrace', 'reverseTurnOnTrace', 'reverseTurnOffTrace']
const array = (values, path, dtype = '<f8') => ({ kind: 'array', dtype, itemsize: dtype.includes('U') ? 4 : 8, shape: [1, values.length], path, values: [values] })
function archive(name = 'source_125C.mat') {
  return { schema: 'transistordatabase-legacy-mat', version: 1, parser_version: 1, workspace_schema: 'scData', source: { name, sha256: sourceHash, byte_length: 4, format: 'mat5' }, source_version: '0.2', source_base64: 'AAECAw==',
    conversion_history: [], known_fields: ['name', 'onStateCurrent'], unknown_fields: ['custom'], legacy_coordinates: 'x/image-height,y/image-width',
    proposed_units: { provenance: 'Legacy SC editor labels', values: { onStateCurrent: 'A', onStateVoltage: 'V' } },
    variables: { scData: { kind: 'struct', shape: [1, 1], fields: {
      name: array([], 'scData.name', '<U1'), onStateCurrent: array([-20, -10, 10, 20], 'scData.onStateCurrent'), onStateVoltage: array([-3, -1, 1, 3], 'scData.onStateVoltage'), custom: array([0], 'scData.custom'),
    } } }, calibrations: { 'scData.forwardConductionTrace': { axes: { x: { status: 'reconstructed-for-comparison', scale: 'lin', label: 'Voltage (V)', reconstructed_values: [1, 3], residuals: [0, 0], outside_calibration_indices: [11], legacy_pixel_locations: [10, 30], max_absolute_residual: 0 } } } },
    diagnostics: [{ path: 'scData.turnOn', code: 'synthetic-placeholder', message: 'The ±1e100 zero-energy signature is not measured zero loss' }], status: 'preview-only-requires-explicit-finalization' }
}
const nativeRows = rows => ({ kind: 'list', values: rows.map(values => ({ kind: 'list', values })) })
function finalized(request) {
  return { status: 'finalized-not-persisted', declarations: request, limitations: ['Selected recorded branch only'], core_json_document: { schema: 'transistordatabase-core', version: 1, transistor: { kind: 'aggregate', name: 'Transistor', fields: {
    switch: { kind: 'aggregate', name: 'Switch', fields: { channel_data: { kind: 'list', values: [{ kind: 'dataclass', name: 'ChannelCharacteristics', fields: { t_j: 73, v_g: 15, graph_v_i: { kind: 'ndarray', dtype: '<f8', shape: [2, 2], values: nativeRows([[1, 3], [10, 20]]) } } }] } } }, diode: { kind: 'aggregate', name: 'Diode', fields: {} },
  } } } }
}

describe('legacy MAT browser workflow', () => {
  let wrapper, blobs, downloads
  const field = id => wrapper.get(`[data-testid="mat-${id}"]`)
  async function upload(name = 'source_125C.mat', bytes = [0, 1, 2, 3]) {
    const file = { name, size: bytes.length, arrayBuffer: async () => Uint8Array.from(bytes).buffer }
    Object.defineProperty(field('file').element, 'files', { configurable: true, value: [file] })
    await field('file').trigger('change'); await flushPromises()
  }
  async function declare(kind = 'conduction') {
    const selectedBranch = kind === 'conduction' ? branch : 'forwardTurnOnTrace'
    const values = { 'identity-name': 'Declared part', 'identity-type': 'MOSFET', 'identity-manufacturer': 'Declared vendor', 'identity-housing_type': 'TO-247', 'identity-author': 'Source reviewer',
      'rating-v_abs_max': 650, 'rating-i_abs_max': 100, 'rating-i_cont': 60, 'rating-t_j_max': 150, 'area-housing_area': 0.0001, 'area-cooling_area': 0.0002, 'declaration-source': 'Independent datasheet table 1' }
    for (const [key, value] of Object.entries(values)) await field(key).setValue(String(value))
    for (const key of mechanisms.filter(key => key !== selectedBranch)) await field('omit-' + key).setValue('Preserve this source branch in the archive')
    await field('select-' + selectedBranch).setValue(true)
    for (const [key, value] of Object.entries({ target: 'switch', source: 'original-trace', convention: 'trace-positive-magnitudes', order: 'recorded-order', zero: 'reject-zero' })) await field(`${key}-${selectedBranch}`).setValue(value)
    await field('indices-' + selectedBranch).setValue('0,1')
    await field('evidence-' + selectedBranch).setValue('Figure 2, independent declared 73 °C / 15 V conditions')
    await field(`condition-${selectedBranch}-t_j_c`).setValue('73')
    await field(`condition-${selectedBranch}-gate_voltage_v`).setValue('15')
    await field('confirm-units').setValue(true)
    return selectedBranch
  }
  async function finalize() { await field('finalize').trigger('submit'); await flushPromises() }
  beforeEach(() => {
    for (const method of Object.values(api)) method.mockReset()
    api.preview.mockResolvedValue(archive()); api.finalize.mockImplementation(async request => finalized(request))
    api.export.mockResolvedValue(new Blob([Uint8Array.from([0, 1, 2, 3])]))
    blobs = []; downloads = []
    vi.spyOn(URL, 'createObjectURL').mockImplementation(blob => { blobs.push(blob); return 'blob:mat' })
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {})
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function () { downloads.push(this.download) })
    wrapper = mount(LegacyMatWorkbench)
  })
  afterEach(() => { wrapper.unmount(); vi.restoreAllMocks() })

  it('preserves upload bytes and previews source signs, empties, units, unknowns and original calibration', async () => {
    await upload()
    expect(api.preview).toHaveBeenCalledWith({ mat_base64: 'AAECAw==', source_name: 'source_125C.mat' })
    expect(field('hash').text()).toBe(sourceHash)
    expect(wrapper.text()).toContain('Empty · unknown'); expect(wrapper.text()).toContain('custom')
    expect(wrapper.text()).toContain('±1e100'); expect(wrapper.text()).toContain('Outside-calibration indices: 11')
    expect(field('coordinate-convention').text()).toContain('x/image-height,y/image-width')
    expect(field('identity-name').element.value).toBe('')
    expect(wrapper.findAll('.chart-stub').length).toBe(1)
    await field('archive-json').trigger('click')
    expect(JSON.parse(await blobs.at(-1).text())).toEqual(archive())
    await field('source-export').trigger('click'); await flushPromises()
    expect(api.export).toHaveBeenLastCalledWith({ mat_base64: 'AAECAw==', source_name: 'source_125C.mat', mode: 'source-preserved' })
    expect([...new Uint8Array(await blobs.at(-1).arrayBuffer())]).toEqual([0, 1, 2, 3])
    await field('legacy-export').trigger('click'); await flushPromises()
    expect(api.export.mock.calls.at(-1)[0].mode).toBe('legacy-mat5')
  })

  it('sends independent declarations, original indices and citation evidence then downloads native JSON verbatim', async () => {
    await upload(); await declare(); await finalize()
    const request = api.finalize.mock.calls.at(-1)[0], curve = request.curves[0]
    expect(request.identity.name).toBe('Declared part'); expect(request.declaration_source).toBe('Independent datasheet table 1')
    expect(curve).toMatchObject({ source: 'original-trace', sample_indices: [0, 1], extrapolation_policy: 'reject', evidence_reference: 'Figure 2, independent declared 73 °C / 15 V conditions', conditions: { t_j_c: 73, gate_voltage_v: 15, gate_resistance_ohm: null, supply_voltage_v: null, gate_off_voltage_v: null, commutation_device: null } })
    expect(Object.keys(request.omission_reasons)).toHaveLength(5)
    expect(field('core-result').text()).toContain('73 °C')
    expect(field('core-result').text()).toContain('Voltage (V)')
    await field('core-json').trigger('click')
    expect(JSON.parse(await blobs.at(-1).text())).toEqual(finalized(request).core_json_document)
    await field(`extrapolation-${branch}`).setValue('explicit-estimate')
    expect(wrapper.find('[data-testid=mat-core-result]').exists()).toBe(false)
    await finalize(); expect(api.finalize.mock.calls.at(-1)[0].curves[0].extrapolation_policy).toBe('explicit-estimate')
  })

  it('uses Python bounded values, reports coverage and rejects stale lookup exports after edits', async () => {
    const calculation = { status: 'supported', target: 'switch', current_a: 15, voltage_v: 2, resistance_ohm: 2 / 15, current_coverage_a: [10, 20], evidence_reference: 'Figure 2', method: 'Existing Python bounded lookup', method_version: 1 }
    api.calculate.mockResolvedValue(calculation)
    await upload(); await declare(); await finalize()
    await field('calculate-target').setValue('switch'); await field('calculate-current').setValue('15')
    await field('calculate').trigger('submit'); await flushPromises()
    expect(api.calculate.mock.calls.at(-1)[0]).toMatchObject({ current_a: 15, target: 'switch', declaration_source: 'Independent datasheet table 1' })
    expect(field('calculation-voltage').text()).toBe('2'); expect(field('calculation-resistance').text()).toBe(String(2 / 15))
    await field('calculation-json').trigger('click'); expect(JSON.parse(await blobs.at(-1).text())).toEqual(calculation)
    await field('calculation-csv').trigger('click'); expect(await blobs.at(-1).text()).toBe(`current_A,voltage_V,resistance_ohm,coverage_min_A,coverage_max_A\r\n15,2,${2 / 15},10,20\r\n`)
    await field('calculate-current').setValue('5')
    expect(wrapper.find('[data-testid=mat-calculation]').exists()).toBe(false)
    api.calculate.mockRejectedValueOnce({ response: { data: { detail: 'Current 5 A is outside recorded coverage [10, 20] A' } } })
    await field('calculate').trigger('submit'); await flushPromises()
    expect(wrapper.get('[role=alert]').text()).toContain('outside recorded coverage')
    let release
    api.calculate.mockImplementationOnce(() => new Promise(resolve => { release = resolve }))
    await field('calculate-current').setValue('15'); await field('calculate').trigger('submit')
    await field(`condition-${branch}-t_j_c`).setValue('74'); release(calculation); await flushPromises()
    expect(wrapper.find('[data-testid=mat-core-result]').exists()).toBe(false)
    expect(wrapper.find('[data-testid=mat-calculation]').exists()).toBe(false)
  })

  it('retains zero interpretation choices and surfaces Python negative-energy and placeholder rejection', async () => {
    await upload(); const selected = await declare('turn-on')
    await field('source-' + selected).setValue('combined'); await field('convention-' + selected).setValue('signed-passive')
    await field('status-' + selected).setValue('source-zero-unverified'); await field('ack-' + selected).setValue('The source zero has no measurement evidence')
    api.finalize.mockRejectedValueOnce({ response: { data: { detail: 'Synthetic placeholder energy cannot supply a passive loss curve' } } })
    await finalize()
    expect(api.finalize.mock.calls.at(-1)[0].curves[0]).toMatchObject({ data_status: 'source-zero-unverified', source_zero_acknowledgement: 'The source zero has no measurement evidence', ideal_model_scope: null })
    expect(wrapper.get('[role=alert]').text()).toContain('Synthetic placeholder')
    await field('status-' + selected).setValue('declared-ideal-model'); await field('scope-' + selected).setValue('Explicit ideal branch in this simulation only')
    api.finalize.mockRejectedValueOnce({ response: { data: { detail: 'Negative source energy remains unsupported' } } }); await finalize()
    expect(api.finalize.mock.calls.at(-1)[0].curves[0]).toMatchObject({ data_status: 'declared-ideal-model', source_zero_acknowledgement: null })
    expect(wrapper.get('[role=alert]').text()).toContain('Negative source energy')
  })

  it('ignores late preview, mapping and export responses after source replacement', async () => {
    let release
    api.preview.mockImplementationOnce(() => new Promise(resolve => { release = resolve }))
    await upload('old.mat'); await upload('new.mat', [4, 5]); release(archive('old.mat')); await flushPromises()
    expect(wrapper.text()).not.toContain('old.mat')
    await declare()
    api.finalize.mockImplementationOnce(() => new Promise(resolve => { release = resolve }))
    await field('finalize').trigger('submit'); const request = api.finalize.mock.calls.at(-1)[0]
    await field(`condition-${branch}-t_j_c`).setValue('74'); release(finalized(request)); await flushPromises()
    expect(wrapper.find('[data-testid=mat-core-result]').exists()).toBe(false)
    api.export.mockImplementationOnce(() => new Promise(resolve => { release = resolve }))
    await field('source-export').trigger('click'); await upload('replacement.mat', [6]); release(new Blob(['old'])); await flushPromises()
    expect(downloads).toHaveLength(0); expect(field('identity-name').element.value).toBe('')
  })

  it('labels unsupported workspace evaluation and rejects oversize files before API upload', async () => {
    api.preview.mockResolvedValueOnce({ ...archive(), workspace_schema: 'hsData', variables: { hsData: { kind: 'struct', shape: [1, 1], fields: {} } } })
    await upload(); expect(wrapper.text()).toContain('Preview and original-byte archive only')
    expect(wrapper.find('[data-testid=mat-mapping]').exists()).toBe(false)
    const file = { name: 'too-large.mat', size: 24 * 1024 * 1024 + 1, arrayBuffer: vi.fn() }
    Object.defineProperty(field('file').element, 'files', { configurable: true, value: [file] }); await field('file').trigger('change'); await flushPromises()
    expect(file.arrayBuffer).not.toHaveBeenCalled(); expect(api.preview).toHaveBeenCalledTimes(1)
    expect(wrapper.get('[role=alert]').text()).toContain('24 MiB')
  })
})

describe('original MAT binary array display', () => {
  let wrapper
  afterEach(() => wrapper?.unmount())
  const source = { mat_base64: 'AA==', source_name: 'array.mat' }
  function binaryNode(dtype, itemsize, shape) { return { kind: 'array', dtype, itemsize, shape, path: 'scData.custom', encoding: 'source-array-reference', source_array_path: ['scData', 'custom'] } }
  function blob(dtype, itemsize, shape, bytes) { return { source_sha256: sourceHash, dtype, itemsize, shape, byte_length: bytes.length, data_base64: btoa(String.fromCharCode(...bytes)) } }
  it('displays original uint64 exactly through BigInt and paginates without Number coercion', async () => {
    const bytes = new Uint8Array(8), view = new DataView(bytes.buffer); view.setBigUint64(0, 18446744073709551615n, true)
    api.array.mockResolvedValueOnce(blob('<u8', 8, [1, 1], bytes))
    wrapper = mount(LegacyMatField, { props: { node: binaryNode('<u8', 8, [1, 1]), name: 'custom', source, sourceHash } })
    await wrapper.get('button').trigger('click'); await flushPromises()
    expect(wrapper.text()).toContain('18446744073709551615')
  })
  it('decodes UTF-32 Unicode items using itemsize and retains exact whitespace', async () => {
    const bytes = new Uint8Array(12), view = new DataView(bytes.buffer)
    for (const [i, code] of [65, 10, 0x1f60a].entries()) view.setUint32(i * 4, code, true)
    api.array.mockResolvedValueOnce(blob('<U1', 4, [1, 3], bytes))
    wrapper = mount(LegacyMatField, { props: { node: binaryNode('<U1', 4, [1, 3]), name: 'custom', source, sourceHash } })
    await wrapper.get('button').trigger('click'); await flushPromises()
    expect(wrapper.findAll('tbody td.literal').map(cell => cell.element.textContent)).toEqual(['A', '\n', '😊'])
  })
  it('rejects late source arrays after replacement', async () => {
    let release; api.array.mockImplementationOnce(() => new Promise(resolve => { release = resolve }))
    wrapper = mount(LegacyMatField, { props: { node: binaryNode('<u1', 1, [1, 1]), name: 'custom', source, sourceHash } })
    await wrapper.get('button').trigger('click'); await wrapper.setProps({ source: { ...source, source_name: 'new.mat' } })
    release(blob('<u1', 1, [1, 1], [42])); await flushPromises()
    expect(wrapper.find('table').exists()).toBe(false)
  })
})
