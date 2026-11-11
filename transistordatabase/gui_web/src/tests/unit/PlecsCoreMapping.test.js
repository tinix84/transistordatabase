import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
const { finalize } = vi.hoisted(() => ({ finalize: vi.fn() }))
vi.mock('../../services/api.js', () => ({ plecsDataApi: { finalize } }))
import PlecsCoreMapping from '../../components/PlecsCoreMapping.vue'
const declarations = { ratings: { v_abs_max: 1200, i_abs_max: 80, i_cont: 40, t_j_max: 175 }, thermal_properties: { housing_area: .0001, cooling_area: .00005 }, block_targets: { '0': 'switch', '1': 'diode' } }
const response = { status: 'finalized-not-persisted', declarations, source: { sha256: 'abc' }, core: { metadata: { name: 'Example' }, switch: { channel_data: [1], e_on_data: [1, 2], e_off_data: [1] }, diode: { channel_data: [1], e_rr_data: [1] } }, roundtrip_xml_base64: 'PHNvdXJjZS8+' }
const document = { packages: [{ id: '0', partnumber: 'Example', thermal: [], blocks: [{ id: '0', device_type: 'MOSFET', tables: [{ source_path: 'on', kind: 'turn-on', gate_voltage_v: null }] }, { id: '1', device_type: 'Diode', tables: [{ source_path: 'recovery', kind: 'turn-off', gate_voltage_v: null }] }] }] }
describe('explicit PLECS core mapping', () => {
  let wrapper
  beforeEach(() => { finalize.mockReset().mockResolvedValue(response); wrapper = mount(PlecsCoreMapping, { props: { document, sourceRequest: { xml_base64: 'PHNvdXJjZS8+', source_name: 'source.xml' } } }) })
  afterEach(() => wrapper.unmount())
  async function fill() {
    await wrapper.get('[data-testid=plecs-map-package]').setValue('0')
    for (const [key, value] of Object.entries({ ...declarations.ratings, ...declarations.thermal_properties })) await wrapper.get(`[data-testid=plecs-declare-${key}]`).setValue(String(value))
    await wrapper.get('[data-testid=plecs-target-0]').setValue('switch'); await wrapper.get('[data-testid=plecs-target-1]').setValue('diode')
    for (const gate of wrapper.findAll('[data-testid=plecs-declare-gate]')) await gate.setValue('15')
    await wrapper.get('[data-testid=plecs-recovery-convention]').setValue('negative-to-positive-bus')
  }
  async function submit() { await wrapper.get('form').trigger('submit'); await flushPromises() }
  it('starts with blank declarations and deliberate package and target choices', async () => {
    await wrapper.get('[data-testid=plecs-map-package]').setValue('0'); await submit()
    expect(finalize).not.toHaveBeenCalled(); expect(wrapper.get('[role=alert]').text()).toContain('Declare a finite Absolute voltage')
    expect(wrapper.get('[data-testid=plecs-target-0]').element.value).toBe('')
  })
  it('sends declared SI values, independent block targets and explicit signed recovery mapping', async () => {
    await fill(); await submit()
    expect(finalize).toHaveBeenCalledWith({ xml_base64: 'PHNvdXJjZS8+', source_name: 'source.xml', package_id: '0',
      ...declarations, gate_voltage_by_table: { on: 15, recovery: 15 }, recovery_voltage_convention: 'negative-to-positive-bus', package_thermal_target: null })
    expect(wrapper.get('[data-testid=plecs-core-result]').text()).toContain('finalized-not-persisted')
    expect(wrapper.get('[data-testid=plecs-core-result]').text()).toContain('1200')
    expect(wrapper.findAll('tbody tr').map(row => row.findAll('td').map(cell => cell.text()))).toEqual([['switch', '1', '2', '1', '0'], ['diode', '1', '0', '0', '1']])
  })
  it('clears accepted and pending core results when declarations change', async () => {
    await fill(); await submit(); await wrapper.get('[data-testid=plecs-declare-v_abs_max]').setValue('1300')
    expect(wrapper.find('[data-testid=plecs-core-result]').exists()).toBe(false)
    let release; finalize.mockImplementationOnce(() => new Promise(resolve => { release = resolve }))
    await wrapper.get('form').trigger('submit'); await wrapper.get('[data-testid=plecs-declare-i_cont]').setValue('35')
    release(response); await flushPromises(); expect(wrapper.find('[data-testid=plecs-core-result]').exists()).toBe(false)
  })
  it('shows backend unsupported mapping reason rather than a fabricated core', async () => {
    finalize.mockRejectedValueOnce({ response: { data: { detail: 'Nonzero diode turn-on has no core representation' } } })
    await fill(); await submit()
    expect(wrapper.get('[role=alert]').text()).toContain('Nonzero diode turn-on')
    expect(wrapper.find('[data-testid=plecs-core-result]').exists()).toBe(false)
  })
})
