import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'

const { getAll, getDPTData, generateDPTNetlist, validateDPTData } = vi.hoisted(() => ({
  getAll: vi.fn(),
  getDPTData: vi.fn(),
  generateDPTNetlist: vi.fn(),
  validateDPTData: vi.fn()
}))

vi.mock('../../services/api.js', () => ({
  transistorApi: { getAll, getDPTData, generateDPTNetlist, validateDPTData }
}))

import DPTManager from '../../components/DPTManager.vue'

const dptResponse = {
  has_e_on: true,
  has_e_off: true,
  has_e_rr: false,
  metadata: { t_j: 25, v_supply: 400, v_g: 15, r_g: 10, comment: 'Energy summary' },
  raw_measurement_data: [
    {
      metadata: {
        dataset_type: 'dpt_u_i',
        comment: 'First raw acquisition set',
        t_j: 25,
        v_supply: 400,
        v_g: 15,
        v_g_off: -5,
        r_g: 10,
        measurement_testbench: 'Local test bench'
      },
      dpt_on_vds: [
        [[-7.9605e-8, 450], [-7.9445e-8, 460], [0, 400], [1e-6, 210]],
        [[0, 390], [1e-6, 200]]
      ],
      dpt_on_id: [
        [[0, 0], [1e-6, 20]],
        [[0, 0], [1e-6, 22]]
      ],
      dpt_off_vds: [],
      dpt_off_id: [
        [[0, '-Infinity'], [1e-6, 18]],
        [[0, 600], [1e-6, 20]]
      ]
    },
    {
      metadata: { dataset_type: 'dpt_u_i_r', comment: 'Second raw record', t_j: 125 },
      dpt_on_vds: [],
      dpt_on_id: [],
      dpt_off_vds: [[[0, 600], [2e-6, 580]]],
      dpt_off_id: [[[0, 40], [2e-6, 35]]]
    }
  ]
}

const CurvePlotterStub = {
  props: ['title', 'datasets', 'xLabel', 'xValueDecimals', 'yLabel', 'showLegend'],
  template: `
    <div
      class="curve-plotter-stub"
      :data-title="title"
      :data-x-label="xLabel"
      :data-x-decimals="xValueDecimals"
      :data-y-label="yLabel"
    >
      <span>{{ title }}</span>
      <output>{{ JSON.stringify(datasets[0].data) }}</output>
    </div>
  `
}

describe('DPTManager raw waveform viewing', () => {
  let wrapper

  beforeEach(async () => {
    getAll.mockReset().mockResolvedValue([{ metadata: { name: 'Waveform Device' } }])
    getDPTData.mockReset().mockResolvedValue(structuredClone(dptResponse))
    generateDPTNetlist.mockReset().mockResolvedValue({
      netlist: 'test netlist', transistor_name: 'Waveform Device'
    })
    validateDPTData.mockReset().mockResolvedValue({
      valid: true, errors: [], warnings: []
    })
    wrapper = mount(DPTManager, {
      global: { stubs: { CurvePlotter: CurvePlotterStub } }
    })
    await flushPromises()
    await wrapper.get('select').setValue('Waveform Device')
    await flushPromises()
  })

  afterEach(() => {
    wrapper.unmount()
  })

  it('shows conditions and separate turn-on/off voltage and current traces', () => {
    expect(getDPTData).toHaveBeenCalledWith('Waveform Device')
    expect(wrapper.text()).toContain('Supply voltage400 V')
    expect(wrapper.text()).toContain('Junction temperature25 °C')
    expect(wrapper.text()).toContain('Test benchLocal test bench')
    expect(wrapper.findAll('.curve-plotter-stub').map(chart => chart.attributes('data-title')))
      .toEqual(['Turn-on Vds', 'Turn-on Id', 'Turn-off Id'])
    expect(wrapper.findAll('.curve-plotter-stub').every(
      chart => chart.attributes('data-x-label') === 'Time [µs]'
    )).toBe(true)
    expect(wrapper.findAll('.curve-plotter-stub').every(
      chart => chart.attributes('data-x-decimals') === '6'
    )).toBe(true)
    const plottedSamples = JSON.parse(
      wrapper.find('.curve-plotter-stub[data-title="Turn-on Vds"] output').text()
    )
    expect(plottedSamples[0].x).toBeCloseTo(-0.079605, 12)
    expect(plottedSamples[1].x).toBeCloseTo(-0.079445, 12)
    expect(plottedSamples[3].x).toBe(1)
    expect(wrapper.find('[data-testid="dpt-waveform-off-vds"] [role="status"]').text())
      .toContain('Turn-off Vds')
    expect(wrapper.text()).toContain('Non-finite raw samples are preserved and shown as gaps.')
    expect(wrapper.find('.curve-plotter-stub[data-title="Turn-off Id"]').text())
      .toContain('{"x":0,"y":null}')
  })

  it('selects another acquisition and measurement record', async () => {
    await wrapper.get('[data-testid="dpt-acquisition-select"]').setValue('1')
    expect(wrapper.find('.curve-plotter-stub[data-title="Turn-on Vds"]').text())
      .toContain('{"x":0,"y":390}')
    expect(wrapper.find('.curve-plotter-stub[data-title="Turn-on Vds"]').text())
      .toContain('{"x":1,"y":200}')

    await wrapper.get('[data-testid="dpt-measurement-select"]').setValue('1')

    expect(wrapper.text()).toContain('Junction temperature125 °C')
    expect(wrapper.find('.curve-plotter-stub[data-title="Turn-off Vds"]').text())
      .toContain('{"x":0,"y":600}')
    expect(wrapper.find('[data-testid="dpt-waveform-on-vds"] [role="status"]').text())
      .toContain('Turn-on Vds')
    expect(wrapper.find('[data-testid="dpt-waveform-on-id"] [role="status"]').text())
      .toContain('Turn-on Id')
  })

  it('shows a clear empty state when no raw records are returned', async () => {
    getDPTData.mockResolvedValueOnce({
      has_e_on: true,
      has_e_off: false,
      has_e_rr: false,
      metadata: null,
      raw_measurement_data: []
    })
    await wrapper.get('select').setValue('')
    await wrapper.get('select').setValue('Waveform Device')
    await flushPromises()

    expect(wrapper.text()).toContain('No raw DPT waveform measurements are stored')
  })

  it('keeps the existing netlist and validation tabs available', async () => {
    await wrapper.get('button.tab-btn:nth-of-type(2)').trigger('click')
    await wrapper.get('.btn-generate').trigger('click')
    await flushPromises()
    expect(generateDPTNetlist).toHaveBeenCalledWith(
      'Waveform Device',
      expect.objectContaining({ v_dc: 400, i_target: 20 })
    )
    expect(wrapper.text()).toContain('test netlist')

    await wrapper.get('button.tab-btn:nth-of-type(3)').trigger('click')
    await wrapper.get('.btn-validate').trigger('click')
    await flushPromises()
    expect(validateDPTData).toHaveBeenCalledWith('Waveform Device')
    expect(wrapper.text()).toContain('All checks passed')
  })
})
