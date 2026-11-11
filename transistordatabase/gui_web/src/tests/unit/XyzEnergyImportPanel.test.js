import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
const { importXyzEnergy } = vi.hoisted(() => ({ importXyzEnergy: vi.fn() }))
vi.mock('../../services/api.js', () => ({ analysisApi: { importXyzEnergy } }))
import XyzEnergyImportPanel from '../../components/XyzEnergyImportPanel.vue'
import EnergyMapWorkbench from '../../components/EnergyMapWorkbench.vue'
import { xyzResult, xyzText } from '../fixtures/xyzEnergy.js'
let wrapper
const field = name => wrapper.get(`[aria-label="XYZ ${name}"]`)
async function ready() {
  wrapper = mount(XyzEnergyImportPanel)
  await field('source text').setValue(xyzText); await field('source identifier').setValue('Manual fixture')
  await field('energy unit').setValue('µJ'); await field('energy convention').setValue('Recorded switching energy')
  await field('common conditions').setValue(true)
}
async function submit() { await wrapper.get('form').trigger('submit'); await flushPromises() }
beforeEach(() => { vi.clearAllMocks(); const data = xyzResult(); importXyzEnergy.mockResolvedValue({ data, rawJson: JSON.stringify(data) }) })
afterEach(() => { wrapper?.unmount(); vi.restoreAllMocks() })
describe('XYZ energy observation import', () => {
  it('keeps the prepared form and places the XYZ form beside it', async () => {
    wrapper = mount(EnergyMapWorkbench, { global: { stubs: { CurvePlotter: true } } })
    expect(wrapper.findAll('form')).toHaveLength(1); expect(wrapper.find('[data-testid=xyz-energy]').exists()).toBe(false)
    await wrapper.get('[aria-label="Energy map input"]').setValue('xyz')
    expect(wrapper.findAll('form')).toHaveLength(1); expect(wrapper.get('h3').text()).toBe('Import XYZ energy observations')
    await wrapper.get('[aria-label="Energy map input"]').setValue('prepared')
    expect(wrapper.find('[aria-label="Map currents"]').exists()).toBe(true)
  })
  it('sends UTF-8 text and explicit options without browser numeric parsing or repair', async () => {
    await ready(); await field('source text').setValue('énergie\n9,600,90\n3,200,NA\n0,200,0')
    await field('has header').setValue(true); await field('missing energy tokens').setValue('NA')
    await submit()
    const request = importXyzEnergy.mock.calls[0][0]
    expect(new TextDecoder().decode(Uint8Array.from(atob(request.content_base64), char => char.charCodeAt(0)))).toBe('énergie\n9,600,90\n3,200,NA\n0,200,0')
    expect(request).toMatchObject({ input_kind: 'paste', conditions: { t_j_c: null }, options: { decimal: '.', has_header: true, units: { energy: 'µJ' }, missing_energy_tokens: ['NA'], columns: { current: 0, voltage: 1, energy: 2 } } })
  })
  it('renders the server matrix with high voltage above low voltage and explicit source row identity', async () => {
    await ready(); await submit()
    const matrix = wrapper.get('[data-testid=xyz-matrix]'), rows = matrix.findAll('tbody tr')
    expect(rows.map(row => row.attributes('data-voltage'))).toEqual(['600','200'])
    expect(rows[0].findAll('td').map(cell => cell.attributes('data-current'))).toEqual(['0','3','9'])
    expect(rows[0].findAll('td').map(cell => cell.attributes('data-energy'))).toEqual(['0.000012','NULL','0.00008999999999999999'])
    expect(matrix.get('td[data-current="0"][data-voltage="200"]').text()).toContain('0 J')
    expect(matrix.get('td[data-current="3"][data-voltage="600"]').text()).toContain('hole')
    await matrix.get('td[data-current="9"][data-voltage="600"] button').trigger('click')
    expect(wrapper.get('[data-testid=xyz-cell-evidence]').text()).toContain('9,600,90')
  })
  it.each([['source identifier','changed'],['delimiter','tsv'],['energy unit','mJ'],['missing energy tokens','NA'],['current column','1'],['source kind','measured'],['energy kind','e_off'],['Junction temperature (°C)','0']])('invalidates result and exports after %s changes', async (name,value) => {
    await ready(); await submit(); await field(name).setValue(value)
    expect(wrapper.find('[data-testid=xyz-results]').exists()).toBe(false)
    expect(wrapper.find('[data-testid=xyz-json]').exists()).toBe(false)
  })
  it('discards pending responses after text changes', async () => {
    await ready(); let resolve
    importXyzEnergy.mockImplementationOnce(() => new Promise(done => { resolve = done }))
    await wrapper.get('form').trigger('submit'); await flushPromises()
    await field('source text').setValue('0,0,0'); resolve({ data: xyzResult(), rawJson: '{}' }); await flushPromises()
    expect(wrapper.find('[data-testid=xyz-results]').exists()).toBe(false)
    expect(wrapper.get('[data-testid=xyz-parse]').attributes('disabled')).toBeUndefined()
  })
  it('sends exact original file bytes and discards a stale asynchronous file read', async () => {
    await ready(); await field('source text').setValue('x'.repeat(4_000_001)); await field('source input').setValue('file')
    const bytes = new Uint8Array([239,187,191,48,44,48,44,48,13,10])
    Object.defineProperty(field('source file').element, 'files', { value: [{ name: 'original.csv', size: bytes.length, arrayBuffer: async () => bytes.buffer }], configurable: true })
    await field('source file').trigger('change'); await submit()
    expect(importXyzEnergy.mock.calls[0][0].content_base64).toBe(btoa(String.fromCharCode(...bytes)))
    expect(importXyzEnergy.mock.calls[0][0].source_name).toBe('original.csv')
    let resolve; Object.defineProperty(field('source file').element, 'files', { value: [{ name: 'old.csv', size: 3, arrayBuffer: () => new Promise(done => { resolve = done }) }], configurable: true })
    await field('source file').trigger('change'); await wrapper.get('form').trigger('submit')
    await field('source identifier').setValue('new source'); resolve(bytes.buffer); await flushPromises()
    expect(importXyzEnergy).toHaveBeenCalledTimes(1); expect(wrapper.find('[data-testid=xyz-results]').exists()).toBe(false)
  })
  it('requires declared conditions and rejects duplicated mapping and byte limits before transport', async () => {
    await ready(); await field('common conditions').setValue(false); await submit()
    expect(wrapper.get('[role=alert]').text()).toContain('Confirm'); expect(importXyzEnergy).not.toHaveBeenCalled()
    await field('common conditions').setValue(true); await field('current column').setValue('1'); await submit()
    expect(wrapper.get('[role=alert]').text()).toContain('different column'); expect(importXyzEnergy).not.toHaveBeenCalled()
    await field('current column').setValue('0'); await field('source text').setValue('é'.repeat(2_000_001)); await submit()
    expect(wrapper.get('[role=alert]').text()).toContain('4 MB'); expect(importXyzEnergy).not.toHaveBeenCalled()
  })
  it('validates original condition numeric text before conversion and keeps unknown separate from zero', async () => {
    await ready(); await field('Junction temperature (°C)').setValue('1e-400'); await submit()
    expect(importXyzEnergy).not.toHaveBeenCalled(); expect(wrapper.get('[role=alert]').text()).toContain('too small')
    await field('Junction temperature (°C)').setValue('0'); await submit()
    expect(importXyzEnergy.mock.calls[0][0].conditions.t_j_c).toBe(0)
    await field('Junction temperature (°C)').setValue(''); await submit()
    expect(importXyzEnergy.mock.calls[1][0].conditions.t_j_c).toBe(null)
  })
  it('shows parser diagnostics and keeps failed imports without exports', async () => {
    await ready(); importXyzEnergy.mockRejectedValueOnce({ response: { data: { detail: 'Line 3: current must be nonnegative.' } } })
    await submit(); expect(wrapper.get('[role=alert]').text()).toContain('Line 3'); expect(wrapper.find('[data-testid=xyz-results]').exists()).toBe(false)
  })
  it('retains unknown and repeated coordinate states without numeric substitutes', async () => {
    await ready(); const data = xyzResult()
    data.cells[0][0] = { ...data.cells[0][0], energy_j: null, status: 'unknown' }
    data.cells[1][2] = { ...data.cells[1][2], energy_j: null, status: 'ambiguous', row_ids: ['line-1','line-6'] }
    importXyzEnergy.mockResolvedValueOnce({ data, rawJson: JSON.stringify(data) }); await submit()
    expect(wrapper.get('td[data-current="0"][data-voltage="200"]').text()).toContain('NULL J')
    expect(wrapper.get('td[data-current="9"][data-voltage="600"]').text()).toContain('ambiguous')
    expect(wrapper.get('td[data-current="9"][data-voltage="600"]').text()).toContain('line-1, line-6')
  })
  it('pages large duplicate evidence and writes common CSV metadata once', async () => {
    await ready(); const data = xyzResult()
    data.rows = Array.from({ length: 3000 }, (_, index) => ({ ...data.rows[0], id: `line-${index + 1}`, cell_status: 'ambiguous' }))
    data.cells = [[{ current_a: 9, voltage_v: 600, energy_j: null, status: 'ambiguous', row_ids: data.rows.map(row => row.id), reason: 'Repeated coordinate' }]]
    data.currents_a = [9]; data.voltages_v = [600]; data.source.source = '=metadata-only-once'
    importXyzEnergy.mockResolvedValueOnce({ data, rawJson: JSON.stringify(data) }); await submit()
    await wrapper.get('[data-testid=xyz-matrix] button').trigger('click')
    expect(wrapper.get('[data-testid=xyz-cell-evidence]').findAll('pre')).toHaveLength(100)
    expect(wrapper.get('[data-testid=xyz-cell-evidence]').text()).toContain('3000 source rows')
    await wrapper.findAll('button').find(button => button.text() === 'Next cell rows').trigger('click')
    expect(wrapper.get('[data-testid=xyz-cell-evidence] pre').text()).toContain('line-101')
    const blobs = []; vi.spyOn(URL, 'createObjectURL').mockImplementation(blob => { blobs.push(blob); return 'blob:fixture' })
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {})
    await wrapper.get('[data-testid=xyz-rows-csv]').trigger('click')
    const csv = await blobs[0].text()
    expect(csv.match(/metadata-only-once/g)).toHaveLength(1)
    expect(csv).toContain("'=metadata-only-once"); expect(csv.split('\r\n')).toHaveLength(3001)
  })
  it('normalizes tiny accepted energies before applying the linear color scale', async () => {
    await ready(); const data = xyzResult()
    data.cells = [[{ ...data.cells[0][0], energy_j: 5e-324 }, { ...data.cells[0][1], energy_j: 1e-323 }]]
    data.currents_a = [0,3]; data.voltages_v = [200]
    importXyzEnergy.mockResolvedValueOnce({ data, rawJson: JSON.stringify(data) }); await submit()
    expect(wrapper.get('[data-testid=xyz-matrix] tbody td').attributes('style')).toContain('0.46')
  })
  it('pages the full accepted matrix without mounting more than 256 cells', async () => {
    await ready(); const data = xyzResult(), coordinates = Array.from({length:17},(_,i) => i)
    data.currents_a=coordinates; data.voltages_v=coordinates
    data.cells=coordinates.map(v => coordinates.map(i => ({current_a:i,voltage_v:v,energy_j:0,status:'available',row_ids:[],reason:null})))
    importXyzEnergy.mockResolvedValueOnce({data,rawJson:JSON.stringify(data)}); await submit()
    const matrix=wrapper.get('[data-testid=xyz-matrix]')
    expect(matrix.findAll('td')).toHaveLength(256)
    expect(matrix.get('tbody tr').attributes('data-voltage')).toBe('16')
    await wrapper.findAll('button').find(button => button.text()==='Next current columns').trigger('click')
    expect(matrix.findAll('td')).toHaveLength(16)
    expect(matrix.get('td').attributes('data-current')).toBe('16')
    await wrapper.findAll('button').find(button => button.text()==='Next voltage rows').trigger('click')
    expect(matrix.findAll('td')).toHaveLength(1)
    expect(matrix.get('tbody tr').attributes('data-voltage')).toBe('0')
  })
})
