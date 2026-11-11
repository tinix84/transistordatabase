import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
const api = vi.hoisted(() => ({ preview: vi.fn(), snapshot: vi.fn(), links: vi.fn(), coefficients: vi.fn(), diff: vi.fn() }))
vi.mock('../../services/api.js', () => ({ catalogSourceApi: api }))
import CatalogSourceWorkbench from '../../components/CatalogSourceWorkbench.vue'

const cell = (column, value, extra = {}) => ({ column, cell: `${column}:1`, value, cached_value: value, formula: null, error: null, ...extra })
const preview = {
  source_file: { filename: 'quoted.csv', size_bytes: 52, sha256: 'abc123' },
  mapping_fields: { part_number: ['text'], voltage_v: ['V', 'kV'], rds_ohm: ['Ohm', 'mOhm'] },
  sheets: [{ name: 'CSV', state: 'visible', row_count: 2, hidden_rows: [2], formula_count: 1, error_count: 1, native_link_count: 0,
    rows: [{ row: 1, hidden: false, cells: [cell('1', 'Part'), cell('2', 'Rating'), cell('3', 'Rating')] },
      { row: 2, hidden: true, cells: [cell('1', 'A,"B\nC'), cell('2', null, { formula: '=OTHER()', cached_value: null }), cell('3', '#VALUE!', { error: '#VALUE!' })] }] }],
}
const snapshot = { schema_version: 1, version: 'version123', sources: [{ id: 's1', file: 'quoted.csv' }], records: [{ id: 's1:2', part_number: 'A', row: 2 }], row_errors: [{ row: 2, errors: [{ field: 'rds_ohm', cell: '3:2', reason: 'missing-value' }] }] }

describe('Catalog source browser', () => {
  let wrapper, readers
  const button = text => wrapper.findAll('button').find(b => b.text() === text)
  async function upload(file = { name: 'quoted.csv', size: 52, base64: 'AP9hCg==' }) {
    const input = wrapper.get('[aria-label="Catalog source file"]')
    Object.defineProperty(input.element, 'files', { configurable: true, value: [file] })
    await input.trigger('change'); await flushPromises()
  }
  async function inspect() { await upload(); await button('Preview source').trigger('click'); await flushPromises() }
  async function map() { await wrapper.get('[aria-label="Map part_number"]').setValue('1'); await wrapper.get('[aria-label="Map voltage_v"]').setValue('2'); await wrapper.get('[aria-label="Unit voltage_v"]').setValue('V') }
  beforeEach(() => {
    readers = []
    vi.stubGlobal('FileReader', class {
      constructor() { readers.push(this) }
      readAsDataURL(file) { this.result = `data:application/octet-stream;base64,${file.base64}`; if (!file.hold) this.onload() }
    })
    Object.values(api).forEach(mock => mock.mockReset())
    api.preview.mockResolvedValue(structuredClone(preview)); api.snapshot.mockResolvedValue(structuredClone(snapshot))
    wrapper = mount(CatalogSourceWorkbench)
  })
  afterEach(() => { wrapper.unmount(); vi.unstubAllGlobals() })
  it('uploads exact bytes with declared parsing and exposes duplicate IDs, hidden rows and formula/cache/errors', async () => {
    await upload()
    await wrapper.get('[aria-label="CSV separator"]').setValue(';')
    await wrapper.get('[aria-label="Decimal convention"]').setValue(',')
    await button('Preview source').trigger('click'); await flushPromises()
    expect(api.preview).toHaveBeenCalledWith({ filename: 'quoted.csv', content_base64: 'AP9hCg==', encoding: 'utf-8-sig', delimiter: ';', decimal: ',', preview_rows: 200 })
    expect(wrapper.text()).toContain('2 · Rating (duplicate header)')
    expect(wrapper.text()).toContain('3 · Rating (duplicate header)')
    expect(wrapper.text()).toContain('2 · hidden')
    expect(wrapper.text()).toContain('Formula: =OTHER()')
    expect(wrapper.text()).toContain('Saved cache: Unknown / missing')
    expect(wrapper.text()).toContain('Source error: #VALUE!')
    expect(button('Create catalog snapshot').attributes('disabled')).toBeDefined()
    await wrapper.get('[aria-label="Map part_number"]').setValue('1')
    await wrapper.get('[aria-label="Map rds_ohm"]').setValue('3')
    expect(wrapper.get('[aria-label="Unit rds_ohm"]').element.value).toBe('')
    expect(button('Create catalog snapshot').attributes('disabled')).toBeDefined()
  })
  it('creates a mapped snapshot with unknown context, passes only its catalog to FOM, and invalidates on edits', async () => {
    await inspect(); await map()
    await wrapper.get('[aria-label="Unit voltage_v"]').setValue('kV')
    await button('Create catalog snapshot').trigger('click'); await flushPromises()
    const request = api.snapshot.mock.calls[0][0]
    expect(request.mapping).toEqual({ part_number: { column: '1', unit: 'text' }, voltage_v: { column: '2', unit: 'kV' } })
    expect(request.metadata).toEqual({ label: null, currency: null, snapshot: null, quote_quantity: null, technology: 'Unknown', workbook_profile: null, conditions: { junction_temperature_c: null, gate_voltage_v: null, gate_resistance_ohm: null } })
    expect(wrapper.text()).toContain('missing-value')
    await button('Explore snapshot in FOM').trigger('click')
    expect(wrapper.emitted('explore-catalog')[0][0]).toEqual({ schema_version: 1, sources: snapshot.sources, records: snapshot.records })
    await wrapper.get('[aria-label="Recorded currency"]').setValue('CHF')
    expect(wrapper.find('[aria-label="Created catalog snapshot"]').exists()).toBe(false)
  })
  it('ignores a source read that finishes after a replacement file', async () => {
    await upload({ name: 'old.csv', size: 10, base64: 'b2xk', hold: true })
    await upload({ name: 'new.csv', size: 10, base64: 'bmV3' })
    readers[0].onload(); await flushPromises()
    await button('Preview source').trigger('click'); await flushPromises()
    expect(api.preview.mock.calls[0][0]).toMatchObject({ filename: 'new.csv', content_base64: 'bmV3' })
  })
  it('requires an explicit conduction profile and explicit coefficient units', async () => {
    api.preview.mockResolvedValue({ ...preview, mapping_fields: { ...preview.mapping_fields, err_j_a: ['J/A', 'mJ/A'] } })
    await inspect(); await map()
    expect(wrapper.get('[aria-label="Workbook conduction profile"]').element.selectedOptions[0].text).toBe('Unknown · no workbook derivation')
    await wrapper.get('[aria-label="Workbook conduction profile"]').setValue('igbt')
    await wrapper.get('[aria-label="Map err_j_a"]').setValue('3')
    expect(button('Create catalog snapshot').attributes('disabled')).toBeDefined()
    await wrapper.get('[aria-label="Unit err_j_a"]').setValue('J/A')
    await button('Create catalog snapshot').trigger('click'); await flushPromises()
    expect(api.snapshot.mock.lastCall[0].metadata.workbook_profile).toBe('igbt')
    expect(api.snapshot.mock.lastCall[0].mapping.err_j_a).toEqual({ column: '3', unit: 'J/A' })
    await wrapper.get('[aria-label="Workbook conduction profile"]').setValue('mosfet')
    expect(wrapper.find('[aria-label="Created catalog snapshot"]').exists()).toBe(false)
  })
  it('rejects held preview/snapshot responses after source convention or unit edits', async () => {
    let release
    api.preview.mockImplementationOnce(() => new Promise(resolve => { release = resolve }))
    await upload(); await button('Preview source').trigger('click')
    await wrapper.get('[aria-label="CSV encoding"]').setValue('cp1252')
    release(preview); await flushPromises()
    expect(wrapper.find('[aria-label="Source cell preview"]').exists()).toBe(false)
    await inspect(); await map()
    api.snapshot.mockImplementationOnce(() => new Promise(resolve => { release = resolve }))
    await button('Create catalog snapshot').trigger('click')
    await wrapper.get('[aria-label="Unit voltage_v"]').setValue('kV')
    release(snapshot); await flushPromises()
    expect(wrapper.find('[aria-label="Created catalog snapshot"]').exists()).toBe(false)
  })
  it('keeps static links as text and J/A coefficient/cache/condition evidence distinct', async () => {
    api.links.mockResolvedValue({ entries: [{ sheet: 'CSV', cell: 'A2', kind: 'stored-link', target: '../archive.pdf', resolution: 'workbook-relative', target_mode: 'External' }], native_link_count: 1, geturl_formula_count: 1, geturl_formula_locations: [{ sheet: 'CSV', cell: 'B2', formula: '=GetURL(A2)', status: 'not-evaluated' }], unsupported_hyperlink_formulas: [] })
    api.coefficients.mockResolvedValue({ scope: 'Documented layouts', coefficients: [{ sheet: 'MOSFET', part_number: 'Part', component: 'e_on', cell: { cell: 'Y2', formula: '=.000632/36', error: null }, header: { value: 'EON (TJ=175°C)' }, recognized_literal_coefficient_j_per_a: .000632 / 36, cached_coefficient_j_per_a: null, reference_current_a: 36, reference_voltage_v: 600, reference_temperature_c: 175, reference_temperature_header_cell: 'Y1', gate_resistance_ohm: null, literal_energy_j: .000632, status: 'literal-division-cache-missing' }] })
    await inspect(); await button('Inspect static links').trigger('click'); await flushPromises()
    expect(wrapper.text()).toContain('../archive.pdf')
    expect(wrapper.findAll('a')).toHaveLength(0)
    expect(wrapper.text()).toContain('not-evaluated')
    await button('Inspect J/A coefficients').trigger('click'); await flushPromises()
    expect(wrapper.text()).toContain('unit J/A')
    expect(wrapper.text()).toContain('36 A')
    expect(wrapper.text()).toContain('literal-division-cache-missing')
  })
  it('uploads before/after JSON, requires explicit keys, shows duplicate errors and drops stale comparisons', async () => {
    async function json(side, value) {
      const input = wrapper.get(`[aria-label="${side} snapshot JSON"]`)
      Object.defineProperty(input.element, 'files', { configurable: true, value: [{ size: 100, text: async () => JSON.stringify(value) }] })
      await input.trigger('change'); await flushPromises()
    }
    await json('Before', snapshot); await json('After', snapshot)
    expect(button('Compare snapshots').attributes('disabled')).toBeDefined()
    await wrapper.get('[aria-label="Compare key part_number"]').setValue(true)
    api.diff.mockRejectedValueOnce({ response: { data: { detail: 'Comparison keys must be present and unique; disambiguate duplicate rows' } } })
    await button('Compare snapshots').trigger('click'); await flushPromises()
    expect(wrapper.text()).toContain('disambiguate duplicate rows')
    let release; api.diff.mockImplementationOnce(() => new Promise(resolve => { release = resolve }))
    await button('Compare snapshots').trigger('click')
    await wrapper.get('[aria-label="Compare key row"]').setValue(true)
    release({ added: ['a'], changed: [], missing: [], application: 'comparison-only' }); await flushPromises()
    expect(wrapper.find('[aria-label="Snapshot differences"]').exists()).toBe(false)
    api.diff.mockResolvedValue({ added: ['a'], missing: ['b'], changed: [{ key: ['A', 2], before_id: 'x', after_id: 'y', fields: ['voltage_v'] }], application: 'comparison-only; no stored source overwritten' })
    await button('Compare snapshots').trigger('click'); await flushPromises()
    expect(api.diff.mock.calls.at(-1)[0].key_fields).toEqual(['part_number', 'row'])
    expect(wrapper.text()).toContain('1 added · 1 changed · 1 missing')
    expect(wrapper.text()).toContain('voltage_v')
  })
  it('rejects a snapshot beyond the transport budget before reading or parsing it', async () => {
    const text = vi.fn()
    const input = wrapper.get('[aria-label="Before snapshot JSON"]')
    Object.defineProperty(input.element, 'files', { configurable: true, value: [{ size: 50000001, text }] })
    await input.trigger('change'); await flushPromises()
    expect(text).not.toHaveBeenCalled()
    expect(wrapper.text()).toContain('50,000,000 bytes')
  })
  it('rejects overflowing JSON numbers before they can serialize to null in a comparison', async () => {
    const input = wrapper.get('[aria-label="Before snapshot JSON"]')
    Object.defineProperty(input.element, 'files', { configurable: true, value: [{ size: 80, text: async () => '{"schema_version":1,"sources":[],"records":[{"voltage_v":1e400}]}' }] })
    await input.trigger('change'); await flushPromises()
    expect(wrapper.text()).toContain('nonfinite numeric value')
    expect(button('Compare snapshots').attributes('disabled')).toBeDefined()
  })
})
