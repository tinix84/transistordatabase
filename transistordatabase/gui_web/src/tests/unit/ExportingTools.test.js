import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import ExportingTools from '../../components/ExportingTools.vue'
import BatchExportPanel from '../../components/BatchExportPanel.vue'
import { transistorApi } from '../../services/api.js'
vi.mock('../../services/api.js', () => ({ transistorApi: { export: vi.fn(), batchExport: vi.fn(), getAll: vi.fn() } }))
const devices = [{ metadata: { name: 'part-one', manufacturer: 'Source lab' } }, { metadata: { name: 'part-two', manufacturer: 'Second lab' } }]
let wrapper, downloadNames
beforeEach(() => {
  vi.clearAllMocks()
  vi.stubGlobal('URL', { createObjectURL: vi.fn(() => 'blob:original-bytes'), revokeObjectURL: vi.fn() })
  downloadNames = []
  vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function () { downloadNames.push(this.download) })
  transistorApi.getAll.mockResolvedValue(devices)
  transistorApi.export.mockResolvedValue(new Blob(['{"source_id":1152921504606846977}']))
  transistorApi.batchExport.mockResolvedValue(new Blob(['ZIP bytes']))
  wrapper = mount(ExportingTools, { props: { transistors: devices, selectedFromSearch: [devices[0]] } })
})
afterEach(() => { wrapper.unmount(); vi.restoreAllMocks(); vi.unstubAllGlobals() })
const button = text => wrapper.findAll('button').find(item => item.text() === text)
const download = () => wrapper.get('[data-testid=download-export]')
const format = value => wrapper.get('[data-testid=export-format]').setValue(value)
async function conditions(values = {}) {
  for (const [key, value] of Object.entries({ t_j: 37, v_supply: 430, v_g_on: 11, v_g_off: -3, r_g_on: 4, r_g_off: 7, ...values })) {
    await wrapper.get(`[data-testid=gecko-${key}]`).setValue(String(value))
  }
}
describe('Python-backed device exports', () => {
  it('sends native JSON download bytes directly to the browser without number conversion', async () => {
    await download().trigger('click'); await flushPromises()
    expect(transistorApi.export).toHaveBeenCalledWith('part-one', 'json', {})
    const blob = await transistorApi.export.mock.results[0].value
    expect(URL.createObjectURL).toHaveBeenCalledWith(blob)
    expect(await blob.text()).toContain('1152921504606846977')
    expect(wrapper.text()).toContain('Download prepared for 1 device')
  })
  it('requires all Gecko conditions and sends explicit recorded unknown values', async () => {
    await format('gecko')
    expect(download().attributes('disabled')).toBeDefined()
    await conditions({ v_g_off: 'null', r_g_off: 'null' })
    expect(download().attributes('disabled')).toBeUndefined()
    await download().trigger('click'); await flushPromises()
    expect(transistorApi.export).toHaveBeenCalledWith('part-one', 'gecko', {
      component: 'switch', t_j: 37, v_supply: 430, v_g_on: 11, v_g_off: null, r_g_on: 4, r_g_off: null,
    })
  })
  it('keeps request identity and filename when the form changes during a held download', async () => {
    let release
    const bytes = new Blob(['MAT5 bytes'])
    transistorApi.export.mockImplementationOnce(() => new Promise(resolve => { release = resolve }))
    await format('matlab')
    await download().trigger('click')
    await format('gecko')
    await button('Clear selection').trigger('click')
    release(bytes); await flushPromises()
    expect(transistorApi.export).toHaveBeenCalledWith('part-one', 'matlab', {})
    expect(URL.createObjectURL).toHaveBeenCalledWith(bytes)
    expect(downloadNames).toEqual(['part-one_matlab.mat'])
    expect(wrapper.text()).toContain('Download prepared for 1 device.')
    expect(wrapper.findComponent({ name: 'ExportDownloadPanel' }).emitted('export-complete')).toEqual([[{ count: 1, format: 'matlab' }]])
  })
  it('rejects invalid Gecko input and shows API source failures without a download', async () => {
    await format('gecko'); await conditions({ r_g_on: -1 })
    expect(download().attributes('disabled')).toBeDefined()
    expect(transistorApi.export).not.toHaveBeenCalled()
    await conditions()
    transistorApi.export.mockRejectedValueOnce(new Error('No exact recorded channel slice'))
    await download().trigger('click'); await flushPromises()
    expect(wrapper.get('[role=alert]').text()).toContain('No exact recorded channel slice')
    expect(URL.createObjectURL).not.toHaveBeenCalled()
  })
  it('preserves checkbox and search selection and uses one explicit common batch slice', async () => {
    await button('Select visible devices').trigger('click')
    await format('gecko'); await conditions()
    await download().trigger('click'); await flushPromises()
    expect(transistorApi.batchExport).toHaveBeenCalledWith(['part-one', 'part-two'], 'gecko', {
      selection: { component: 'switch', t_j: 37, v_supply: 430, v_g_on: 11, v_g_off: -3, r_g_on: 4, r_g_off: 7 },
    })
    await button('Clear selection').trigger('click')
    expect(download().attributes('disabled')).toBeDefined()
    await wrapper.setProps({ selectedFromSearch: [devices[1]] })
    expect(wrapper.get('legend').text()).toContain('1 selected')
  })
  it('provides real MAT5 and ratings summary formats with workbench navigation', async () => {
    await format('matlab'); await download().trigger('click'); await flushPromises()
    expect(transistorApi.export).toHaveBeenCalledWith('part-one', 'matlab', {})
    expect(wrapper.text()).toContain('not SC scData or a Simulink model')
    await format('csv')
    expect(wrapper.text()).toContain('CSV contains ratings only')
    await button('Open PLECS data workbench').trigger('click')
    await button('Open DPT workbench').trigger('click')
    expect(wrapper.emitted('navigate-workbench')).toEqual([['plecs-data'], ['dpt']])
    expect(wrapper.findAll('option').map(item => item.attributes('value'))).not.toContain('spice')
    expect(wrapper.findAll('option').map(item => item.attributes('value'))).not.toContain('simulink')
    await button('Search database').trigger('click')
    expect(wrapper.emitted('open-database-search')).toHaveLength(1)
  })
  it('uses the same honest format controls for the batch dialog without simulated progress', async () => {
    wrapper.unmount(); wrapper = mount(BatchExportPanel); await flushPromises()
    await button('Select visible devices').trigger('click')
    await download().trigger('click'); await flushPromises()
    expect(transistorApi.batchExport).toHaveBeenCalledWith(['part-one', 'part-two'], 'json', {})
    expect(wrapper.text()).not.toContain('Estimated')
    expect(wrapper.emitted('export-complete')[0][0]).toEqual({ count: 2, format: 'json' })
  })
})
