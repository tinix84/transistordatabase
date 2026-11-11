import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
const { preview, exportXml } = vi.hoisted(() => ({ preview: vi.fn(), exportXml: vi.fn() }))
vi.mock('../../services/api.js', () => ({ plecsDataApi: { preview, export: exportXml } }))
import PlecsWorkbench from '../../components/PlecsWorkbench.vue'

const table = {
  kind: 'turn-off', method: 'Table only', status: 'supported-table', scale: .001,
  current_a: [0, 10], temperature_c: [25], voltage_v: [-400], gate_voltage_v: null,
  gate_resistance_ohm: null, formula: null, values: [[[0, 2]]], source_path: '/package/0/block/0/off',
  slices: [{ label: '25 °C, -400 V', temperature_c: 25, supply_voltage_v: -400, unit: 'J', plot_data: [[0, 0], [10, .002]] }],
}
const response = {
  schema_version: '1', namespace_uri: 'http://www.plexim.com/xml/semiconductors/', version: '1.1',
  source: { name: 'source.xml', sha256: 'ab'.repeat(32) }, source_xml: '<source/>', status: 'supported-tables',
  can_export: true, can_archive: true, diagnostics: [], packages: [{ id: 'p1', partnumber: 'Real identity',
    vendor: 'Source vendor', device_class: 'MOSFET', ratings: null,
    blocks: [{ id: 'b1', device_type: 'MOSFET', tables: [table], preserved_metadata: [] }],
    thermal: [{ network_type: 'Foster', status: 'supported', elements: [{ R: .2, Tau: .001 }], source_path: '/thermal', xml: '<Foster/>' }], preserved_metadata: [] }],
}
describe('PLECS document workflow', () => {
  let wrapper
  beforeEach(() => {
    preview.mockReset().mockResolvedValue(structuredClone(response)); exportXml.mockReset()
    wrapper = mount(PlecsWorkbench, { global: { stubs: { CurvePlotter: true, PlecsCoreExport: true } } })
  })
  afterEach(() => wrapper.unmount())
  const button = name => wrapper.findAll('button').find(item => item.text() === name)
  async function inspect() { await wrapper.get('[data-testid=plecs-xml]').setValue('<source/>'); await wrapper.get('form').trigger('submit'); await flushPromises() }
  it('shows Python SI slices without converting source scale, and retains unknown ratings/conditions', async () => {
    await inspect()
    expect(preview).toHaveBeenCalledWith({ xml_text: '<source/>', source_name: null })
    expect(wrapper.text()).toContain('Absolute voltage/current ratings: Unknown')
    expect(wrapper.text()).toContain('Gate Unknown V · Rg Unknown Ω')
    expect(wrapper.text()).toContain('Signed supply -400 V')
    expect(wrapper.text()).toContain('0.002')
    expect(wrapper.findComponent({ name: 'CurvePlotter' }).props('datasets')[0].data).toEqual([{ x: 0, y: 0 }, { x: 10, y: .002 }])
    expect(button('Download supported table XML').attributes('disabled')).toBeUndefined()
    expect(wrapper.text()).toContain('Foster')
  })
  it('retains formula/base evidence without drawing a false model or allowing supported export', async () => {
    const unsupported = structuredClone(response)
    unsupported.can_export = false; unsupported.status = 'unsupported-content'
    unsupported.diagnostics = [{ code: 'unsupported-formula', path: '/off', message: 'Formula is retained without execution' }]
    Object.assign(unsupported.packages[0].blocks[0].tables[0], { status: 'unsupported-formula', formula: "E*lookup('custom',T)" })
    unsupported.packages[0].blocks[0].preserved_metadata.push({ kind: 'CustomTables', xml: '<CustomTables><Table/></CustomTables>' })
    preview.mockResolvedValueOnce(unsupported); await inspect()
    expect(wrapper.text()).toContain('Unexecuted formula'); expect(wrapper.text()).toContain("E*lookup('custom',T)")
    expect(wrapper.findComponent({ name: 'CurvePlotter' }).exists()).toBe(false)
    expect(button('Download supported table XML').attributes('disabled')).toBeDefined()
    expect(button('Download preserved source XML').attributes('disabled')).toBeUndefined()
    expect(wrapper.text()).toContain('CustomTables')
  })
  it('distinguishes source Cauer elements from independent Foster poles', async () => {
    const cauer = structuredClone(response); cauer.can_export = false
    cauer.packages[0].thermal[0] = { network_type: 'Cauer', status: 'preserved-unsupported', elements: [{ R: .1, C: .2 }], xml: '<Cauer/>', source_path: '/cauer' }
    preview.mockResolvedValueOnce(cauer); await inspect()
    expect(wrapper.text()).toContain('Cauer chain retained as source')
    expect(wrapper.text()).toContain('not interpreted as independent Foster poles')
    cauer.packages[0].blocks[0].thermal = cauer.packages[0].thermal
    cauer.packages[0].thermal = []
    preview.mockResolvedValueOnce(cauer); await wrapper.get('form').trigger('submit'); await flushPromises()
    expect(wrapper.text()).toContain('Block b1 thermal networks')
    expect(wrapper.text()).toContain('Cauer chain retained as source')
  })
  it('invalidates completed and held previews after text or source identity edits', async () => {
    await inspect(); await wrapper.get('[data-testid=plecs-source]').setValue('changed.xml')
    expect(wrapper.find('[data-testid=plecs-results]').exists()).toBe(false)
    let release; preview.mockImplementationOnce(() => new Promise(resolve => { release = resolve }))
    await wrapper.get('form').trigger('submit'); await wrapper.get('[data-testid=plecs-xml]').setValue('<changed/>')
    release(response); await flushPromises()
    expect(wrapper.find('[data-testid=plecs-results]').exists()).toBe(false)
    expect(wrapper.text()).toContain('Inspect again')
  })
  it('rejects blank XML locally and shows structured backend shape diagnostics', async () => {
    await wrapper.get('form').trigger('submit'); await flushPromises(); expect(preview).not.toHaveBeenCalled()
    expect(wrapper.get('[role=alert]').text()).toContain('Choose an XML file')
    preview.mockRejectedValueOnce({ response: { data: { detail: [{ message: 'Voltage row length differs from current axis' }] } } })
    await inspect(); expect(wrapper.get('[role=alert]').text()).toContain('Voltage row length differs')
    expect(wrapper.find('[data-testid=plecs-results]').exists()).toBe(false)
  })
  it('decodes an explicitly Latin-1 XML upload before preview', async () => {
    const text = '<?xml version="1.0" encoding="ISO-8859-1"?><source>é</source>'
    const file = { name: 'latin1.xml', arrayBuffer: async () => Uint8Array.from([...text].map(char => char.charCodeAt(0))).buffer }
    const input = wrapper.get('[data-testid=plecs-file]')
    Object.defineProperty(input.element, 'files', { value: [file] }); await input.trigger('change'); await flushPromises()
    await wrapper.get('form').trigger('submit'); await flushPromises()
    expect(preview).toHaveBeenCalledWith({ xml_base64: btoa(text), source_name: 'latin1.xml' })
    expect(wrapper.get('[data-testid=plecs-xml]').element.value).toBe(text)
    await wrapper.get('[data-testid=plecs-xml]').setValue(text + ' ')
    await wrapper.get('form').trigger('submit'); await flushPromises()
    expect(preview.mock.calls.at(-1)[0]).toEqual({ xml_text: text + ' ', source_name: 'latin1.xml' })
  })
  it('retains identical chart datasets while unrelated XML export state changes', async () => {
    await inspect()
    const plotted = wrapper.findComponent({ name: 'CurvePlotter' }).props('datasets')
    let reject; exportXml.mockImplementationOnce(() => new Promise((resolve, fail) => { reject = fail }))
    await button('Download preserved source XML').trigger('click')
    expect(wrapper.findComponent({ name: 'CurvePlotter' }).props('datasets')).toBe(plotted)
    reject(new Error('Export test rejection')); await flushPromises()
    expect(wrapper.findComponent({ name: 'CurvePlotter' }).props('datasets')).toBe(plotted)
  })
})
