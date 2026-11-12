import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
const { analyze } = vi.hoisted(() => ({ analyze: vi.fn() }))
vi.mock('../../services/api.js', () => ({ thermalApi: { analyze } }))
import ThermalWorkbench from '../../components/ThermalWorkbench.vue'

const response = { operation: 'steady', request: { source: 'Independent fixture' }, result: { forward_temperature_c: 37, reverse_temperature_c: 27 } }
describe('thermal workbench', () => {
  let wrapper
  beforeEach(() => { analyze.mockReset().mockResolvedValue(response); wrapper = mount(ThermalWorkbench, { global: { stubs: { CurvePlotter: true } } }) })
  afterEach(() => wrapper.unmount())
  async function submit() { await wrapper.get('form').trigger('submit'); await flushPromises() }
  it('sends explicit common/branch paths and renders the Python result', async () => {
    await submit()
    expect(analyze.mock.calls[0][0]).toMatchObject({ operation: 'steady', sink_temperature_c: 25, forward_power_w: 10, reverse_power_w: 0,
      common_resistance_k_w: .2, forward_resistance_k_w: 1, reverse_resistance_k_w: 2, interface: null })
    expect(wrapper.get('[aria-label="Thermal result"]').text()).toContain('37')
  })
  it('sends identified simultaneous contacts and preserves source conditions and stale evidence', async () => {
    await wrapper.get('[data-testid=thermal-contacts-enabled]').setValue(true)
    const first = wrapper.get('[data-testid=thermal-contact-0]')
    await first.get('[data-testid=contact-rho]').setValue('1')
    await first.get('[data-testid=contact-area]').setValue('1')
    await first.get('[data-testid=contact-source]').setValue('Recorded sink contact')
    await first.get('[data-testid=contact-conditions]').setValue('pressure_pa,\ntemperature_c,25')
    await wrapper.get('[data-testid=thermal-contact-add]').trigger('click')
    const second = wrapper.get('[data-testid=thermal-contact-1]')
    await second.get('[data-testid=contact-path]').setValue('forward')
    await second.get('[data-testid=contact-rho]').setValue('2')
    await second.get('[data-testid=contact-area]').setValue('1')
    await second.get('[data-testid=contact-source]').setValue('Recorded forward contact')
    const contacts = [{ interface_id: 'contact-1', path: 'common', resistivity_k_m2_w: 1, area_m2: 1,
      source: 'Recorded sink contact', conditions: { pressure_pa: null, temperature_c: 25 } },
      { interface_id: 'contact-2', path: 'forward', resistivity_k_m2_w: 2, area_m2: 1, source: 'Recorded forward contact', conditions: {} }]
    analyze.mockResolvedValueOnce({ ...response, result: { shared_temperature_c: 28, forward_temperature_c: 32, reverse_temperature_c: 31,
      interface_contributions: contacts.map(row => ({ ...row, resistance_k_w: row.resistivity_k_m2_w })),
      added_path_resistances_k_w: { common: 1, forward: 2, reverse: 0 }, path_resistances_k_w: { common: 1, forward: 2, reverse: 0 } } })
    await submit()
    expect(analyze.mock.calls.at(-1)[0].interfaces).toEqual(contacts)
    expect(analyze.mock.calls.at(-1)[0].interface).toBeNull()
    expect(wrapper.get('[data-testid=thermal-contact-evidence]').text()).toContain('Recorded sink contact')
    expect(wrapper.get('[data-testid=thermal-contact-evidence]').text()).toContain('"pressure_pa": null')
    expect(wrapper.get('[data-testid=thermal-contacts-download]').exists()).toBe(true)
    expect(wrapper.get('[aria-label="Thermal result"]').text()).toContain('Shared temperature')
    await second.get('[data-testid=contact-area]').setValue('2')
    expect(wrapper.find('[aria-label="Thermal result"]').exists()).toBe(false)
    let resolve; analyze.mockImplementationOnce(() => new Promise(done => { resolve = done }))
    await wrapper.get('form').trigger('submit')
    await first.get('[data-testid=contact-source]').setValue('Revised source')
    resolve(response); await flushPromises()
    expect(wrapper.find('[aria-label="Thermal result"]').exists()).toBe(false)
  })
  it('preserves reserved condition names as actual JSON evidence keys', async () => {
    await wrapper.get('[data-testid=thermal-contacts-enabled]').setValue(true)
    const first = wrapper.get('[data-testid=thermal-contact-0]')
    await first.get('[data-testid=contact-source]').setValue('Known source')
    await first.get('[data-testid=contact-rho]').setValue('0')
    await first.get('[data-testid=contact-area]').setValue('1')
    await first.get('[data-testid=contact-conditions]').setValue('__proto__,\nconstructor,12\ntoString,24')
    await submit()
    const conditions = analyze.mock.calls.at(-1)[0].interfaces[0].conditions
    expect(Object.hasOwn(conditions, '__proto__')).toBe(true)
    expect(JSON.parse(JSON.stringify(conditions))).toEqual(JSON.parse('{"__proto__":null,"constructor":12,"toString":24}'))
  })
  it('rejects blank contact evidence and nonzero lexical underflow before object transport', async () => {
    await wrapper.get('[data-testid=thermal-contacts-enabled]').setValue(true)
    await submit(); expect(analyze).not.toHaveBeenCalled()
    expect(wrapper.get('[role=alert]').text()).toContain('source')
    const first = wrapper.get('[data-testid=thermal-contact-0]')
    await first.get('[data-testid=contact-source]').setValue('Known source')
    await first.get('[data-testid=contact-rho]').setValue('1e-400')
    await first.get('[data-testid=contact-area]').setValue('1')
    await submit(); expect(analyze).not.toHaveBeenCalled()
    expect(wrapper.get('[role=alert]').text()).toContain('too small')
  })
  it('retains original advanced thermal JSON for backend numeric guards', async () => {
    const advanced = wrapper.findAll('label.check').find(node => node.text().includes('Use advanced JSON'))
    await advanced.get('input').setValue(true)
    const raw = '{"interfaces":[{"resistivity_k_m2_w":-1e-400}]}'
    await wrapper.get('[aria-label="Thermal JSON"]').setValue(raw)
    analyze.mockRejectedValueOnce({ response: { data: { detail: 'Nonzero input underflow' } } })
    await submit()
    expect(analyze.mock.calls.at(-1)[0]).toBe(raw)
    expect(wrapper.get('[role=alert]').text()).toContain('Nonzero input underflow')
  })
  it('clears completed and in-flight stale output after an edit', async () => {
    await submit(); await wrapper.get('[aria-label="Forward heat"]').setValue('11')
    expect(wrapper.find('[aria-label="Thermal result"]').exists()).toBe(false)
    let resolve; analyze.mockImplementationOnce(() => new Promise(done => { resolve = done }))
    await submit(); await wrapper.get('[aria-label="Forward heat"]').setValue('12')
    resolve(response); await flushPromises()
    expect(wrapper.find('[aria-label="Thermal result"]').exists()).toBe(false)
  })
  it('retains exact cubic coefficients and explicit event ownership', async () => {
    await wrapper.get('[aria-label="Thermal operation"]').setValue('transient')
    await wrapper.get('[aria-label="Heat segments"]').setValue('0,1,0,0,100,-100')
    await wrapper.get('[aria-label="Event heat convention"]').setValue('impulse')
    await wrapper.get('[aria-label="Thermal events"]').setValue('.037,.1,forward')
    await submit()
    const sent = analyze.mock.calls[0][0]
    expect(sent.segments[0].power_coefficients_w).toEqual([0, 0, 100, -100])
    expect(sent.events[0]).toMatchObject({ time_s: .037, energy_j: .1, heat_owner: 'forward' })
    expect(sent.network_heat_owner).toBe('forward')
    expect(sent.initial_state).toBe('cold')
  })
  it('requires an explicit temperature-estimate selection in its canonical request', async () => {
    await wrapper.get('[aria-label="Thermal operation"]').setValue('electrothermal')
    await submit(); expect(analyze.mock.calls[0][0].loss_models[0].selected).toBe(false)
    await wrapper.get('[aria-label="Select branch 1 temperature estimate"]').setValue(true)
    await submit(); expect(analyze.mock.calls[1][0].loss_models[0].selected).toBe(true)
  })
  it('supports typed errors and rejects malformed paired data before HTTP', async () => {
    analyze.mockRejectedValueOnce({ response: { data: { detail: [{ msg: 'Missing physical path' }] } } })
    await submit(); expect(wrapper.get('[role="alert"]').text()).toContain('Missing physical path')
    await wrapper.get('[aria-label="Thermal operation"]').setValue('transient')
    await wrapper.get('[aria-label="Heat segments"]').setValue('0,1,100')
    await submit(); expect(analyze).toHaveBeenCalledTimes(1)
    expect(wrapper.get('[role="alert"]').text()).toContain('6 values')
  })
  it('preserves unknown gate conditions and separately supplied package limits', async () => {
    await wrapper.get('[aria-label="Thermal operation"]').setValue('screening')
    await submit()
    expect(analyze.mock.calls[0][0].conditions.gate_v).toBeNull()
    expect(analyze.mock.calls[0][0].package_limit).toBeNull()
    expect(analyze.mock.calls[0][0].reference_kind).toBe('case')
  })
  it('renders accepted branch results as values with units rather than a JSON cell', async () => {
    analyze.mockResolvedValueOnce({ operation: 'electrothermal', request: { source: 'Fixture', loss_models: [{}] }, result: {
      status: 'converged', tolerance_k: 1e-8, accepted_solution: { iteration: 6, temperature_c: [30.102040816], power_w: [10.204081632], residual_k: [1e-9] },
      history: [{ iteration: 6, temperature_c: [30.102040816], power_w: [10.204081632], residual_k: [1e-9] }] } })
    await submit()
    const branches = wrapper.get('.branch-results')
    expect(branches.text()).toContain('Junction (°C)')
    expect(branches.text()).toContain('Loss (W)')
    expect(branches.text()).toContain('Residual (K)')
    expect(branches.text()).toContain('30.10204')
    expect(branches.text()).not.toContain('temperature_c')
    expect(wrapper.text()).toContain('Accepted branch solution')
  })
})
