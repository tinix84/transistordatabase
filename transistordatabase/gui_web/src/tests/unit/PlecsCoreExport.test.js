import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
const { getAll, contexts, previewCore, exportCore } = vi.hoisted(() => ({ getAll: vi.fn(), contexts: vi.fn(), previewCore: vi.fn(), exportCore: vi.fn() }))
vi.mock('../../services/api.js', () => ({ transistorApi: { getAll }, plecsDataApi: { contexts, previewCore, exportCore } }))
import PlecsCoreExport from '../../components/PlecsCoreExport.vue'
const options = { components: { switch: { conduction: [{ gate_voltage_v: 15, curve_count: 2, temperatures_c: [25, 150] }], turn_on: [{ gate_voltage_v: 15, gate_resistance_ohm: null }], turn_off: [], thermal_available: true }, diode: { conduction: [], turn_on: [], turn_off: [], thermal_available: false } } }
const response = { document: { packages: [{ blocks: [{ tables: [{ source_path: 'conduction', kind: 'conduction', slices: [{ label: '25 °C', unit: 'V', plot_data: [[0, 0], [10, 1]] }] }] }] }] }, report: { status: 'selected-subset', simulator_validation: 'not-performed', omitted_mechanisms: ['turn_on', 'turn_off', 'thermal'], fixed_conditions_storage: 'Recorded separately', evidence: [{ kind: 'conduction', conditions: { gate_voltage_v: 15 }, additional_conditions: null, temperatures_c: [25, 150], voltages_v: null, curve_count: 2 }] } }
describe('existing core exact PLECS export', () => {
  let wrapper
  beforeEach(async () => {
    getAll.mockReset().mockResolvedValue([{ metadata: { name: 'stored-device' } }]); contexts.mockReset().mockResolvedValue(options)
    previewCore.mockReset().mockResolvedValue(response); exportCore.mockReset()
    wrapper = mount(PlecsCoreExport, { global: { stubs: { CurvePlotter: true } } }); await flushPromises()
  })
  afterEach(() => wrapper.unmount())
  async function choose() { await wrapper.get('[data-testid=plecs-existing-device]').setValue('stored-device'); await flushPromises(); await wrapper.get('[data-testid=plecs-existing-component]').setValue('switch') }
  async function fill() {
    await choose(); await wrapper.get('[data-testid=plecs-include-conduction]').setValue(true)
    await wrapper.get('[data-testid=plecs-context-conduction]').setValue('0')
    for (const key of ['turn_on', 'turn_off', 'thermal']) await wrapper.get(`[data-testid=plecs-omit-${key}]`).setValue(`Outside selected ${key} scope`)
    await wrapper.get('[data-testid=plecs-core-convention]').setValue('preserve-signed')
  }
  async function submit() { await wrapper.get('form').trigger('submit'); await flushPromises() }
  it('never picks the first device, component or measured gate context implicitly', async () => {
    expect(wrapper.get('[data-testid=plecs-existing-device]').element.value).toBe('')
    expect(contexts).not.toHaveBeenCalled(); await choose()
    expect(wrapper.get('[data-testid=plecs-include-conduction]').element.checked).toBe(false)
    await wrapper.get('[data-testid=plecs-include-conduction]').setValue(true)
    expect(wrapper.get('[data-testid=plecs-context-conduction]').element.value).toBe('')
    await submit(); expect(previewCore).not.toHaveBeenCalled()
  })
  it('sends exact recorded selectors and deliberate omissions with source SI evidence', async () => {
    await fill(); await submit()
    expect(previewCore).toHaveBeenCalledWith('stored-device', { component: 'switch', conduction: { gate_voltage_v: 15 }, turn_on: null, turn_off: null, include_thermal: false, omission_reasons: { turn_on: 'Outside selected turn_on scope', turn_off: 'Outside selected turn_off scope', thermal: 'Outside selected thermal scope' }, recovery_voltage_convention: 'preserve-signed' })
    expect(wrapper.get('[data-testid=plecs-core-export-result]').text()).toContain('selected-subset')
    expect(wrapper.findComponent({ name: 'CurvePlotter' }).props('datasets')[0].data).toEqual([{ x: 0, y: 0 }, { x: 10, y: 1 }])
  })
  it('matches an actual unknown Rg as null rather than choosing zero', async () => {
    await fill(); await wrapper.get('[data-testid=plecs-include-turn_on]').setValue(true)
    await wrapper.get('[data-testid=plecs-context-turn_on]').setValue('0'); await submit()
    expect(previewCore.mock.calls[0][1].turn_on).toEqual({ gate_voltage_v: 15, gate_resistance_ohm: null })
    expect(previewCore.mock.calls[0][1].omission_reasons.turn_on).toBeUndefined()
  })
  it('clears accepted and held exports on any selection/omission edit', async () => {
    await fill(); await submit(); await wrapper.get('[data-testid=plecs-omit-thermal]').setValue('Changed declared scope')
    expect(wrapper.find('[data-testid=plecs-core-export-result]').exists()).toBe(false)
    let release; previewCore.mockImplementationOnce(() => new Promise(resolve => { release = resolve }))
    await wrapper.get('form').trigger('submit'); await wrapper.get('[data-testid=plecs-include-thermal]').setValue(true)
    release(response); await flushPromises(); expect(wrapper.find('[data-testid=plecs-core-export-result]').exists()).toBe(false)
  })
  it('shows exact unsupported-condition errors without returning partial fake tables', async () => {
    previewCore.mockRejectedValueOnce({ response: { data: { detail: 'turn_on: incomplete temperature × voltage rectangle' } } })
    await fill(); await submit()
    expect(wrapper.get('[role=alert]').text()).toContain('incomplete temperature × voltage rectangle')
    expect(wrapper.find('[data-testid=plecs-core-export-result]').exists()).toBe(false)
  })
  it('shows malformed source record diagnostics while allowing deliberate valid context selection', async () => {
    contexts.mockResolvedValueOnce({ ...options, diagnostics: [{ component: 'switch', kind: 'turn_on', record_index: 3, code: 'invalid-record-conditions', message: 'Recorded junction temperature must be finite' }] })
    await choose()
    expect(wrapper.text()).toContain('Record 3: Recorded junction temperature must be finite')
    await wrapper.get('[data-testid=plecs-include-conduction]').setValue(true)
    expect(wrapper.get('[data-testid=plecs-context-conduction]').findAll('option')).toHaveLength(2)
  })
})
