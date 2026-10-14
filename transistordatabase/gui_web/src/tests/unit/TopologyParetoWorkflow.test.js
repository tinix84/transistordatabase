import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'

const { calculateTopologyPareto } = vi.hoisted(() => ({
  calculateTopologyPareto: vi.fn()
}))

vi.mock('../../services/api.js', () => ({
  transistorApi: { calculateTopologyPareto }
}))

import TopologyParetoPanel from '../../components/TopologyParetoPanel.vue'

const sourceResponse = {
  topology: 'boost',
  boundary: {
    v_in: 100, v_out: 200, p_out: 1000, f_sw: 20000,
    l_value: 0.001, c_out: 0.001, t_j: 25, t_case: 60,
    mode: 'loss-balanced', conduction_regime: 'CCM'
  },
  mode: 'loss-balanced',
  conduction_regime: 'CCM',
  loss_scope: 'channel_conduction_only',
  objective_directions: {
    channel_conduction_loss_w: 'minimize',
    package_body_volume_mm3: 'minimize',
    component_cost_usd: 'minimize'
  },
  dominance_rule: 'Exact objective ties remain non-dominated.',
  cost_basis: 'one-unit USD list price per physical package; counts are included',
  eligible_candidate_count: 2,
  pareto_candidate_ids: ['cree-one', 'infineon-one'],
  model_limitations: ['Switching losses are excluded.'],
  candidates: [
    {
      candidate_id: 'cree-one', label: 'CREE pair', eligible: true, pareto: true,
      ineligible_reasons: [],
      objectives: { channel_conduction_loss_w: 18.849, package_body_volume_mm3: 853.2, component_cost_usd: 19.28 },
      operating_point: {
        model: { assumptions: ['Triangular inductor current in continuous conduction mode.', 'Capacitor ESR is ignored.'] },
        calculated: { inductor_volt_second_residual_v: 0.004662 },
        losses: { scope: 'Switch and diode channel conduction only.' },
        temperatures: { switch_c: null, diode_c: null, switch_unavailable_reason: 'TDB resistance unavailable.', diode_unavailable_reason: 'TDB resistance unavailable.' }
      },
      ratings: { pass: true },
      catalog: {
        switch: { orderable_part: 'C3M0060065J', unit_cost_usd: 9.64, cost_accessed_on: '2026-10-14', price_source_url: 'https://example.com/price', dimensions_source_url: 'https://example.com/datasheet' }
      }
    },
    {
      candidate_id: 'infineon-one', label: 'Infineon pair', eligible: true, pareto: true,
      ineligible_reasons: [],
      objectives: { channel_conduction_loss_w: 17.623, package_body_volume_mm3: 860.4, component_cost_usd: 20.42 },
      operating_point: {
        calculated: { inductor_volt_second_residual_v: 0.004179 },
        losses: { scope: 'Switch and diode channel conduction only.' },
        temperatures: { switch_c: null, diode_c: null, switch_unavailable_reason: 'TDB resistance unavailable.', diode_unavailable_reason: 'TDB resistance unavailable.' }
      },
      ratings: { pass: true },
      catalog: {
        switch: { orderable_part: 'IPBE65R050CFD7AATMA1', unit_cost_usd: 10.78, cost_accessed_on: '2026-10-14', price_source_url: 'https://example.com/price', dimensions_source_url: 'https://example.com/datasheet' }
      }
    },
    {
      candidate_id: 'missing-curves', label: 'Missing exact curve', eligible: false, pareto: false,
      ineligible_reasons: ['Missing required switch channel curve at the requested exact gate and temperature condition.'],
      objectives: null, operating_point: null, ratings: null, catalog: {}
    }
  ]
}

describe('Topology Pareto browser workflow', () => {
  let wrapper

  beforeEach(() => {
    calculateTopologyPareto.mockReset()
    calculateTopologyPareto.mockResolvedValue(sourceResponse)
    wrapper = mount(TopologyParetoPanel, {
      props: {
        transistors: [
          { metadata: { name: 'CREE_C3M0060065J' } },
          { metadata: { name: 'Infineon_IPBE65R050CFD7A' } }
        ]
      }
    })
  })

  afterEach(() => {
    wrapper.unmount()
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('sends one shared SI boundary and sourced candidate catalog data', async () => {
    await wrapper.get('button.btn-primary').trigger('click')
    await flushPromises()

    const request = calculateTopologyPareto.mock.calls[0][0]
    expect(request.boundary).toMatchObject({
      v_in: 100, v_out: 200, p_out: 1000, f_sw: 20000,
      l_value: 0.001, c_out: 0.001, t_j: 25, t_case: 60
    })
    expect(request.candidates).toHaveLength(3)
    expect(request.candidates[0]).toMatchObject({
      candidate_id: 'cree-one',
      switch_transistor_id: 'CREE_C3M0060065J',
      diode_transistor_id: 'CREE_C3M0060065J',
      parallel_switches: 1, parallel_diodes: 1, v_g: 15, diode_v_g: 0
    })
    expect(request.candidates[2].parallel_switches).toBe(2)
    expect(request.catalog.CREE_C3M0060065J).toMatchObject({
      unit_cost_usd: 9.64,
      cost_accessed_on: '2026-10-14',
      length_mm: 9.125,
      price_source_url: 'https://www.digikey.com/en/products/detail/wolfspeed-inc/C3M0060065J/12525328'
    })
    expect(wrapper.text()).toContain('All listed objectives are minimized')
    expect(wrapper.text()).toContain('Exact objective ties remain non-dominated.')
    expect(wrapper.text()).toContain('Missing exact curve')
    expect(wrapper.text()).toContain('Missing required switch channel curve')
    expect(wrapper.text()).toContain('Capacitor ESR is ignored.')
    expect(wrapper.findAll('svg.pareto-plot')).toHaveLength(3)
  })

  it('shows backend errors and keeps the candidate request available in JSON export', async () => {
    const createObjectURL = vi.fn(() => 'blob:pareto-result')
    const revokeObjectURL = vi.fn()
    vi.stubGlobal('URL', { createObjectURL, revokeObjectURL })
    const clicked = vi.fn()
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(clicked)

    calculateTopologyPareto.mockRejectedValueOnce({
      response: { data: { detail: 'Boost Pareto screening requires v_out > v_in.' } }
    })
    await wrapper.get('button.btn-primary').trigger('click')
    await flushPromises()
    expect(wrapper.get('[role="alert"]').text()).toContain('v_out > v_in')

    calculateTopologyPareto.mockResolvedValueOnce(sourceResponse)
    await wrapper.get('button.btn-primary').trigger('click')
    await flushPromises()
    await wrapper.get('button.btn-info').trigger('click')
    expect(clicked).toHaveBeenCalledOnce()
    expect(createObjectURL).toHaveBeenCalledOnce()
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:pareto-result')
  })
})
