import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
const { inventory, analyze, getAll, downloadAnalysis } = vi.hoisted(() => ({ inventory: vi.fn(), analyze: vi.fn(), getAll: vi.fn(), downloadAnalysis: vi.fn() }))
vi.mock('../../services/api.js', () => ({ transistorApi: { getAll }, deviceThermalApi: { inventory, analyze } }))
vi.mock('../../components/analysisFiles.js', async importOriginal => ({ ...await importOriginal(), downloadAnalysis }))
import NativeDeviceElectrothermal from '../../components/NativeDeviceElectrothermal.vue'
const id = 'Synthetic_Native_Device', rawSha = 'a'.repeat(64), dependency = 'b'.repeat(64)
function sourceInventory() {
  const families = {}
  for (const family of ['channel', 'turn_on', 'turn_off']) families[family] = [25, 125].map((temperature, index) => ({
    index, temperature_c: temperature, representation: family === 'channel' ? 'graph_v_i' : 'graph_i_e',
    current_domain_a: [0, 4], status: 'available', diagnostic: null,
    recorded_conditions: family === 'channel' ? { v_g: 15 } : { v_g: family === 'turn_on' ? 15 : -15,
      v_g_off: null, r_g: 5.6, v_supply: 600, load_inductance: .001, commutation_inductance: 1e-9,
      commutation_device: 'Synthetic diode', method: 'direct', event: family, convention: 'passive', measurement_testbench: 'Synthetic bench' } }))
  return { transistor_id: id, repository_source: { sha256: rawSha }, families }
}
function response(status = 'converged') {
  return { operation: 'native-device-electrothermal', request: { current_a: 2, duty: 1, frequency_hz: 0 },
    table: { device_name: id, scope: 'One switch junction; no diode heat inferred', temperature_domain_c: [25, 125], current_domain_a: [0, 4],
      temperature_c: [25, 125], component_power_w: { channel: [4, 8], turn_on: [0, 0], turn_off: [0, 0] }, total_power_w: [4, 8],
      source_identity: { selected_dependency_sha256: dependency, selected_dependency_scope: 'Selected physics only', repository_source: { sha256: rawSha } },
      assumptions: ['No extrapolation'], active_contributions: { channel: true, turn_on: false, turn_off: false }, record_evidence: [] },
    result: { status, accepted_solution: status === 'converged' ? { temperature_c: [33.69565217], power_w: [4.347826], residual_k: [1e-10] } : null,
      nonconvergence_scope: 'Iteration failure does not prove no physical solution', history: [] },
    accepted_component_power_w: status === 'converged' ? { channel: 4.347826, turn_on: 0, turn_off: 0 } : null }
}
describe('native-device feedback workflow', () => {
  let wrapper
  beforeEach(() => {
    inventory.mockReset().mockResolvedValue(sourceInventory()); analyze.mockReset().mockResolvedValue(response())
    getAll.mockReset().mockResolvedValue([{ id }]); downloadAnalysis.mockReset()
    wrapper = mount(NativeDeviceElectrothermal, { global: { stubs: { CurvePlotter: true } } })
  })
  afterEach(() => wrapper.unmount())
  async function ready() {
    await wrapper.get('[data-testid=native-device-id]').setValue(id)
    await wrapper.get('[data-testid=native-thermal-source]').setValue('Explicit synthetic fixed-sink path')
    await wrapper.get('[data-testid=native-inventory-refresh]').trigger('click'); await flushPromises()
    await wrapper.get('[data-testid=native-estimate-selected]').setValue(true)
  }
  async function conductionOnly() {
    await ready()
    await wrapper.get('[data-testid=native-current]').setValue('2')
    await wrapper.get('[data-testid=native-duty]').setValue('1')
    await wrapper.get('[data-testid=native-frequency]').setValue('0')
    await wrapper.get('[data-testid=native-group-channel-0]').trigger('click')
  }
  async function calculate() { await wrapper.get('[data-testid=native-calculate]').trigger('click'); await flushPromises() }
  it('requires explicit selection, freezes source bytes and exports bounded source tables', async () => {
    expect(wrapper.get('[data-testid=native-calculate]').attributes('disabled')).toBeDefined()
    await conductionOnly(); await calculate()
    expect(analyze.mock.calls[0][0]).toMatchObject({ transistor_id: id, expected_file_sha256: rawSha,
      selected: true, current_a: 2, duty: 1, frequency_hz: 0, switching_policy: 'held-current-pwm',
      channel: { indices: [0, 1], conditions: { v_g: 15 } }, turn_on: null, turn_off: null })
    expect(wrapper.get('[data-testid=native-accepted]').text()).toContain('33.6957')
    expect(wrapper.get('[data-testid=native-source-identity]').text()).toContain(rawSha)
    const plot = wrapper.findComponent({ name: 'CurvePlotter' })
    expect(plot.props('datasets')).toHaveLength(4)
    expect(plot.props('datasets')[0].data).toEqual([{ x: 25, y: 4 }, { x: 125, y: 8 }])
    await wrapper.get('[data-testid=native-thermal-json]').trigger('click')
    expect(downloadAnalysis.mock.calls[0][0]).toEqual(response())
    await wrapper.get('[data-testid=native-thermal-csv]').trigger('click')
    const csv = downloadAnalysis.mock.calls[1][0]
    expect(csv.split('\r\n')).toHaveLength(3)
    expect(csv).toContain(rawSha); expect(csv).toContain(dependency)
    expect(csv).not.toContain('graph_v_i')
  })
  it('retains independent signed gate fields and complete hardware/method contexts', async () => {
    await ready()
    for (const family of ['channel', 'turn_on', 'turn_off']) await wrapper.get(`[data-testid=native-group-${family}-0]`).trigger('click')
    await calculate()
    const sent = analyze.mock.calls[0][0]
    expect(sent.turn_on.conditions.v_g).toBe(15)
    expect(sent.turn_off.conditions.v_g).toBe(-15)
    expect(sent.turn_off.conditions.v_g_off).toBeNull()
    expect(sent.turn_off.conditions.commutation_inductance).toBe(1e-9)
    expect(sent.turn_off.conditions.measurement_testbench).toBe('Synthetic bench')
  })
  it.each(['0.999999999999999999999', '1.000000000000000000001', '1e-400'])('rejects original lossy duty %s before transport', async duty => {
    await conductionOnly(); await wrapper.get('[data-testid=native-duty]').setValue(duty)
    await calculate(); expect(analyze).not.toHaveBeenCalled()
    expect(wrapper.get('[role=alert]').text()).toMatch(/precision|too small/)
  })
  it('rejects mixed fixed source conditions and duplicate temperatures', async () => {
    const changed = sourceInventory(); changed.families.channel[1].recorded_conditions.v_g = 12
    inventory.mockResolvedValueOnce(changed); await conductionOnly()
    await wrapper.get('[data-testid=native-record-channel-1]').setValue(true)
    await calculate(); expect(analyze).not.toHaveBeenCalled()
    expect(wrapper.get('[role=alert]').text()).toContain('mixes fixed conditions')
  })
  it('invalidates completed and pending evidence after a scientific input edit', async () => {
    await conductionOnly(); await calculate()
    await wrapper.get('[data-testid=native-resistance]').setValue('2')
    expect(wrapper.find('[data-testid=native-thermal-result]').exists()).toBe(false)
    let resolve; analyze.mockImplementationOnce(() => new Promise(done => { resolve = done }))
    await wrapper.get('[data-testid=native-calculate]').trigger('click')
    await wrapper.get('[data-testid=native-current]').setValue('3')
    resolve(response()); await flushPromises()
    expect(wrapper.find('[data-testid=native-thermal-result]').exists()).toBe(false)
  })
  it('reports domain exit without presenting the last iterate as an accepted solution', async () => {
    analyze.mockResolvedValueOnce(response('source-domain-exit'))
    await conductionOnly(); await calculate()
    expect(wrapper.find('[data-testid=native-accepted]').exists()).toBe(false)
    expect(wrapper.get('[data-testid=native-no-solution]').text()).toContain('No accepted solution')
    expect(wrapper.get('[data-testid=native-thermal-result]').text()).toContain('source-domain-exit')
  })
  it('surfaces stale repository409 and clears inventory when the device ID changes', async () => {
    analyze.mockRejectedValueOnce({ response: { data: { detail: 'Source changed; refresh inventory' }, status: 409 } })
    await conductionOnly(); await calculate()
    expect(wrapper.get('[role=alert]').text()).toContain('refresh inventory')
    await wrapper.get('[data-testid=native-device-id]').setValue('Another_Device')
    expect(wrapper.find('[data-testid=native-family-channel]').exists()).toBe(false)
    expect(wrapper.find('[data-testid=native-thermal-result]').exists()).toBe(false)
  })
})
