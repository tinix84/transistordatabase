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
      metadata: {
        dataset_type: 'dpt_u_i_r', comment: 'Second raw record', t_j: 125, v_supply: 600
      },
      dpt_on_vds: [
        [[0, 610], [2e-6, 300]],
        [[0, 620], [3e-6, 280]],
        [[0, 630], [4e-6, 270]]
      ],
      dpt_on_id: [
        [[0, 0], [2e-6, 30]],
        [[0, 0], [3e-6, 32]],
        [[0, 0], [4e-6, 34]]
      ],
      dpt_off_vds: [
        [[0, 600], [2e-6, 580]],
        [[0, 620], [3e-6, 590]],
        [[0, 630], [4e-6, 600]]
      ],
      dpt_off_id: [
        [[0, 40], [2e-6, 35]],
        [[0, 42], [3e-6, 36]],
        [[0, 44], [4e-6, 38]]
      ]
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
      <output>{{ JSON.stringify(datasets) }}</output>
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
    )[0].data
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
    expect(wrapper.find('.curve-plotter-stub[data-title="Turn-on Vds"]').exists()).toBe(true)
    expect(wrapper.find('.curve-plotter-stub[data-title="Turn-on Id"]').exists()).toBe(true)
  })

  it('filters the single-record choices by condition and metadata text', async () => {
    await wrapper.get('[data-testid="dpt-measurement-search"]').setValue('600 v')

    const measurementSelect = wrapper.get('[data-testid="dpt-measurement-select"]')
    expect(measurementSelect.findAll('option')).toHaveLength(1)
    expect(measurementSelect.element.value).toBe('1')
    expect(wrapper.text()).toContain('Showing 1 of 2 stored measurements.')
    expect(wrapper.text()).toContain('Junction temperature125 °C')
    expect(wrapper.find('.curve-plotter-stub[data-title="Turn-on Vds"]').text())
      .toContain('{"x":2,"y":300}')

    await wrapper.get('[data-testid="dpt-measurement-search"]').setValue('local test bench')

    expect(wrapper.get('[data-testid="dpt-measurement-select"]').findAll('option'))
      .toHaveLength(1)
    expect(wrapper.get('[data-testid="dpt-measurement-select"]').element.value).toBe('0')
    expect(wrapper.text()).toContain('Test benchLocal test bench')
  })

  it('allows comparison entry when only one of multiple stored records matches', async () => {
    await wrapper.get('[data-testid="dpt-measurement-search"]').setValue('600 v')

    const compareButton = wrapper.get('[data-testid="dpt-compare-toggle"]')
    expect(compareButton.text()).toContain('1 available')
    await compareButton.trigger('click')

    expect(wrapper.find('[data-testid="dpt-compare-record-0"]').exists()).toBe(false)
    expect(wrapper.find('[data-testid="dpt-compare-record-1"]').exists()).toBe(true)
    expect(wrapper.findAll('.comparison-record-option')).toHaveLength(1)
    expect(wrapper.get('[data-testid="dpt-compare-toggle"]').text())
      .toContain('Return to one measurement')
    expect(wrapper.get('[data-testid="dpt-comparison-selection-status"]')
      .text()).toContain('Select at least two stored measurements')
  })

  it('allows comparison entry when none of multiple stored records match', async () => {
    await wrapper.get('[data-testid="dpt-measurement-search"]').setValue('no matching condition')

    const compareButton = wrapper.get('[data-testid="dpt-compare-toggle"]')
    expect(compareButton.text()).toContain('0 available')
    await compareButton.trigger('click')

    expect(wrapper.get('[data-testid="dpt-compare-toggle"]').text())
      .toContain('Return to one measurement')
    expect(wrapper.findAll('.comparison-record-option')).toHaveLength(0)
    expect(wrapper.get('[data-testid="dpt-comparison-selection-status"]')
      .text()).toContain('Select at least two stored measurements')
  })

  it('shows a no-match state and restores records when the filter is cleared', async () => {
    await wrapper.get('[data-testid="dpt-measurement-search"]').setValue('no matching condition')

    expect(wrapper.get('[data-testid="dpt-measurement-search-empty"]').text())
      .toContain('No stored DPT measurements match this search')
    expect(wrapper.find('[data-testid="dpt-measurement-select"]').exists()).toBe(false)
    expect(wrapper.find('.waveform-grid').exists()).toBe(false)

    await wrapper.get('[data-testid="dpt-measurement-search-clear"]').trigger('click')

    expect(wrapper.get('[data-testid="dpt-measurement-select"]').findAll('option'))
      .toHaveLength(2)
    expect(wrapper.find('.curve-plotter-stub[data-title="Turn-on Vds"]').exists()).toBe(true)
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

  it('overlays selected records with independent acquisitions, conditions and gaps', async () => {
    await wrapper.get('[data-testid="dpt-compare-toggle"]').trigger('click')

    expect(wrapper.find('[data-testid="dpt-compare-record-0"]').element.checked).toBe(true)
    expect(wrapper.find('[data-testid="dpt-compare-record-1"]').element.checked).toBe(true)
    expect(wrapper.find('[data-testid="dpt-comparison-record-0"]').text())
      .toContain('Junction temperature25 °C')
    expect(wrapper.find('[data-testid="dpt-comparison-record-1"]').text())
      .toContain('Junction temperature125 °C')
    expect(wrapper.find('[data-testid="dpt-comparison-record-1"]').text())
      .toContain('Supply voltage600 V')
    expect(wrapper.find('[data-testid="dpt-comparison-acquisition-0"]').findAll('option'))
      .toHaveLength(2)
    expect(wrapper.find('[data-testid="dpt-comparison-acquisition-1"]').findAll('option'))
      .toHaveLength(3)

    await wrapper.get('[data-testid="dpt-comparison-acquisition-0"]').setValue('1')
    await wrapper.get('[data-testid="dpt-comparison-acquisition-1"]').setValue('2')

    const onVds = JSON.parse(
      wrapper.find('[data-testid="dpt-waveform-on-vds"] output').text()
    )
    expect(onVds).toHaveLength(2)
    expect(onVds[0].label).toContain('First raw acquisition set · Acquisition 2')
    expect(onVds[0].data[1]).toEqual({ x: 1, y: 200 })
    expect(onVds[1].label).toContain('Second raw record · Acquisition 3')
    expect(onVds[1].data[1]).toEqual({ x: 4, y: 270 })
    const onId = JSON.parse(
      wrapper.find('[data-testid="dpt-waveform-on-id"] output').text()
    )
    expect(onId).toHaveLength(2)
    expect(onId[0].data[1]).toEqual({ x: 1, y: 22 })
    const offVds = JSON.parse(
      wrapper.find('[data-testid="dpt-waveform-off-vds"] output').text()
    )
    expect(offVds).toHaveLength(1)
    expect(offVds[0].data[1]).toEqual({ x: 4, y: 600 })

    await wrapper.get('[data-testid="dpt-comparison-acquisition-0"]').setValue('0')
    const offId = JSON.parse(
      wrapper.find('[data-testid="dpt-waveform-off-id"] output').text()
    )
    expect(offId[0].data[0]).toEqual({ x: 0, y: null })
    expect(offId[1].data[0]).toEqual({ x: 0, y: 44 })
    expect(wrapper.find('[data-testid="dpt-waveform-off-vds"]').text())
      .toContain('No Turn-off Vds samples for: Measurement 1 · dpt_u_i · First raw acquisition set · Acquisition 1')
    expect(wrapper.findAll('.curve-plotter-stub').every(
      chart => chart.attributes('data-x-decimals') === '6'
    )).toBe(true)
  })

  it('filters comparison candidates while retaining hidden selected overlays and acquisitions', async () => {
    await wrapper.get('[data-testid="dpt-compare-toggle"]').trigger('click')
    await wrapper.get('[data-testid="dpt-comparison-acquisition-0"]').setValue('1')
    await wrapper.get('[data-testid="dpt-comparison-acquisition-1"]').setValue('2')
    const apiPayload = await getDPTData.mock.results[0].value
    const originalRawRecords = structuredClone(apiPayload.raw_measurement_data)

    await wrapper.get('[data-testid="dpt-measurement-search"]').setValue('local test bench')

    expect(wrapper.find('[data-testid="dpt-compare-record-0"]').exists()).toBe(true)
    expect(wrapper.find('[data-testid="dpt-compare-record-1"]').exists()).toBe(false)
    expect(wrapper.find('[data-testid="dpt-compare-record-0"]').element.checked).toBe(true)
    expect(wrapper.get('[data-testid="dpt-hidden-comparison-note"]').text())
      .toContain('1 selected measurement hidden by this search stay in the comparison')
    expect(wrapper.get('[data-testid="dpt-comparison-acquisition-0"]').element.value)
      .toBe('1')
    expect(wrapper.get('[data-testid="dpt-comparison-acquisition-1"]').element.value)
      .toBe('2')
    expect(wrapper.get('[data-testid="dpt-comparison-record-1"]').exists()).toBe(true)
    const onVds = JSON.parse(
      wrapper.find('[data-testid="dpt-waveform-on-vds"] output').text()
    )
    expect(onVds[0].data[1]).toEqual({ x: 1, y: 200 })
    expect(onVds[1].data[1]).toEqual({ x: 4, y: 270 })
    expect(apiPayload.raw_measurement_data).toEqual(originalRawRecords)

    await wrapper.get('[data-testid="dpt-measurement-search-clear"]').trigger('click')

    expect(wrapper.find('[data-testid="dpt-compare-record-1"]').exists()).toBe(true)
    expect(wrapper.find('[data-testid="dpt-compare-record-1"]').element.checked).toBe(true)
    expect(wrapper.find('[data-testid="dpt-hidden-comparison-note"]').exists()).toBe(false)
  })

  it('requires two selected records and returns to the existing single-record viewer', async () => {
    await wrapper.get('[data-testid="dpt-compare-toggle"]').trigger('click')
    await wrapper.get('[data-testid="dpt-compare-record-1"]').setValue(false)

    expect(wrapper.get('[data-testid="dpt-comparison-selection-status"]').text())
      .toContain('Select at least two stored measurements')
    expect(wrapper.findAll('.waveform-panel')).toHaveLength(0)

    await wrapper.get('[data-testid="dpt-compare-toggle"]').trigger('click')
    expect(wrapper.find('[data-testid="dpt-measurement-select"]').exists()).toBe(true)
    expect(wrapper.find('.curve-plotter-stub[data-title="Turn-on Vds"]').exists()).toBe(true)
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
