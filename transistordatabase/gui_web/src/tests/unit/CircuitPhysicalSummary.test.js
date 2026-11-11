import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { reactive } from 'vue'
const { summarizeCircuit } = vi.hoisted(() => ({ summarizeCircuit: vi.fn() }))
vi.mock('../../services/circuitSummaryApi.js', () => ({ summarizeCircuit }))
import CircuitPhysicalSummary from '../../components/CircuitPhysicalSummary.vue'
import { serializeMembers, stageMembers } from '../../components/circuitSummaryForm.js'

const bindings = [{ role: 'pair', members: ['Q1', 'Q2'], multiplicity: 2 }]
const response = {
  configuration_id: 'config', scenario_id: 'scenario', input_sha256: 'server-sha',
  electrical: { total_power_w: 2, known_component_power_w: 2, total_status: 'computed' },
  maximum: { tj_c: 35, junction_ids: ['Q1:j', 'Q2:j'], known_heat_tj_c: null, known_heat_junction_ids: [], status: 'complete' },
  coverage: { layout_complete_members: 2, member_count: 2, unallocated_known_heat_w: 0, omissions: [] },
  boundary: { temperature_c: 25, source: 'source', conditions: { description: 'fixed' } },
  members: [], evidence_by_role: { pair: { source: 'server evidence' } }, provenance: { source: 'identity' },
  iteration_status: 'not-required-constant-heat', export_json: '{"exact":"server audit"}\n', export_csv: 'exact,server\r\naudit,csv\r\n',
}

describe('physical circuit input and results', () => {
  let wrapper, converterInput
  beforeEach(async () => {
    summarizeCircuit.mockReset().mockResolvedValue(response)
    converterInput = reactive({ source: 'waveform identity' })
    wrapper = mount(CircuitPhysicalSummary, { props: { bindings, converterInput,
      converterRequest: () => ({ time_s: [0, 1], provenance: { source: converterInput.source } }) } })
    await wrapper.get('[data-testid=circuit-enable]').setValue(true)
  })
  afterEach(() => wrapper.unmount())
  async function stage() { await wrapper.findAll('button').find(button => button.text() === 'Stage members from current converter mapping').trigger('click') }
  async function submit() { await wrapper.get('form').trigger('submit'); await flushPromises() }

  it('stages physical IDs without inferring layout or heat ownership', async () => {
    await stage(); await submit()
    const sent = summarizeCircuit.mock.calls[0][0]
    expect(sent.members.map(member => member.id)).toEqual(['Q1', 'Q2'])
    expect(sent.members[0]).toMatchObject({ layout_complete: false, junctions: [], loss_owners: { conduction: { negative: { die: null, junction_id: null } } } })
    expect(sent.acknowledge_known_only).toBe(false)
    expect(sent.converter.provenance.source).toBe('waveform identity')
    expect(wrapper.text()).toContain('35.0000 °C')
    expect(wrapper.text()).toContain('Q1:j, Q2:j')
    expect(wrapper.findAll('fieldset')).toHaveLength(1)
  })

  it('retains blank resistance as unknown and explicit zero as zero', () => {
    const members = stageMembers(bindings)
    members[0].junctions = [{ id: 'Q1:j', rtheta_k_w: '', source: '', conditions: { description: null } }]
    expect(serializeMembers(members)[0].junctions[0].rtheta_k_w).toBeNull()
    members[0].junctions[0].rtheta_k_w = 0
    expect(serializeMembers(members)[0].junctions[0].rtheta_k_w).toBe(0)
    members[0].junctions[0].rtheta_k_w = '1e-999'
    expect(() => serializeMembers(members)).toThrow('too small')
  })

  it('rejects a raw underflowing sink temperature before transport', async () => {
    await wrapper.get('[data-testid=circuit-sink]').setValue('1e-999'); await submit()
    expect(summarizeCircuit).not.toHaveBeenCalled()
    expect(wrapper.get('[role=alert]').text()).toContain('too small')
  })

  it('clears results and rejects a late response after source or thermal edits', async () => {
    await stage(); await submit(); converterInput.source = 'changed'; await flushPromises()
    expect(wrapper.find('[data-testid=circuit-results]').exists()).toBe(false)
    let resolve
    summarizeCircuit.mockImplementationOnce(() => new Promise(done => { resolve = done }))
    await wrapper.get('form').trigger('submit')
    await wrapper.get('[data-testid=circuit-sink]').setValue(10)
    resolve(response); await flushPromises()
    expect(wrapper.find('[data-testid=circuit-results]').exists()).toBe(false)
  })

  it('labels known-only temperatures separately and never substitutes the full total', async () => {
    summarizeCircuit.mockResolvedValueOnce({ ...response,
      electrical: { total_power_w: null, known_component_power_w: 2, total_status: 'partial' },
      maximum: { tj_c: null, junction_ids: [], known_heat_tj_c: 35, known_heat_junction_ids: ['Q1:j', 'Q2:j'], status: 'incomplete' },
      coverage: { ...response.coverage, omissions: ['Omitted diode switching'] },
    })
    await stage(); await wrapper.get('[data-testid=circuit-known-only]').setValue(true); await submit()
    expect(summarizeCircuit.mock.calls[0][0].acknowledge_known_only).toBe(true)
    expect(wrapper.text()).toContain('Electrical full total: Unavailable W')
    expect(wrapper.text()).toContain('Full maximum junction temperature: Unavailable °C')
    expect(wrapper.text()).toContain('Acknowledged known-heat-only maximum: 35.0000 °C')
    expect(wrapper.text()).toContain('Omitted diode switching')
  })

  it('passes advanced JSON to the server without silently combining form values', async () => {
    const raw = { configuration_id: 'raw', converter: { provenance: { source: 'raw source' } }, members: [] }
    await wrapper.get('[data-testid=circuit-json]').setValue(JSON.stringify(raw))
    await wrapper.get('[data-testid=circuit-advanced]').setValue(true); await submit()
    expect(summarizeCircuit).toHaveBeenCalledWith(JSON.stringify(raw))
    await wrapper.get('[data-testid=circuit-json]').setValue('{')
    expect(wrapper.find('[data-testid=circuit-results]').exists()).toBe(false)
  })
})
