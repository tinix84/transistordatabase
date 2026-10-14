import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'

const {
  getAll,
  getDPTData,
  generateDPTNetlist,
  validateDPTData,
  previewDPTRawImport,
  previewDPTRawTraces,
  importDPTRawMeasurement,
  previewDPTEnergy,
  saveDPTEnergy
} = vi.hoisted(() => ({
  getAll: vi.fn(),
  getDPTData: vi.fn(),
  generateDPTNetlist: vi.fn(),
  validateDPTData: vi.fn(),
  previewDPTRawImport: vi.fn(),
  previewDPTRawTraces: vi.fn(),
  importDPTRawMeasurement: vi.fn(),
  previewDPTEnergy: vi.fn(),
  saveDPTEnergy: vi.fn()
}))

vi.mock('../../services/api.js', () => ({
  transistorApi: {
    getAll,
    getDPTData,
    generateDPTNetlist,
    validateDPTData,
    previewDPTRawImport,
    previewDPTRawTraces,
    importDPTRawMeasurement,
    previewDPTEnergy,
    saveDPTEnergy
  }
}))

import DPTManager from '../../components/DPTManager.vue'

const dptResponse = {
  has_e_on: true,
  has_e_off: true,
  has_e_rr: false,
  waveform_preview: {
    max_points_per_acquisition: 800,
    max_total_points: 40000,
    returned_points: 3200,
    sampling: 'uniform_with_endpoints',
    downsampled: true,
    downsampled_acquisition_count: 4
  },
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
        load_inductance: 0.75,
        commutation_inductance: 0.125,
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

function deferred() {
  let resolve
  let reject
  const promise = new Promise((resolvePromise, rejectPromise) => {
    resolve = resolvePromise
    reject = rejectPromise
  })
  return { promise, resolve, reject }
}

async function chooseRawFile(wrapper, file) {
  const input = wrapper.get('[data-testid="dpt-raw-file"]')
  Object.defineProperty(input.element, 'files', {
    value: [file],
    configurable: true
  })
  await input.trigger('change')
}

async function loadRawHeader(wrapper, file) {
  await chooseRawFile(wrapper, file)
  await wrapper.get('[data-testid="dpt-raw-preview-file"]').trigger('click')
  await flushPromises()
  await wrapper.get('[data-testid="dpt-raw-vds-trace"]').setValue('V(ds)')
  await wrapper.get('[data-testid="dpt-raw-id-trace"]').setValue('I(M1)')
}

describe('DPTManager raw waveform viewing', () => {
  let wrapper

  beforeEach(async () => {
    getAll.mockReset().mockResolvedValue([
      { metadata: { name: 'Waveform Device' } },
      { metadata: { name: 'Second Device' } }
    ])
    getDPTData.mockReset().mockResolvedValue(structuredClone(dptResponse))
    generateDPTNetlist.mockReset().mockResolvedValue({
      netlist: 'test netlist', transistor_name: 'Waveform Device'
    })
    validateDPTData.mockReset().mockResolvedValue({
      valid: true, errors: [], warnings: []
    })
    previewDPTRawImport.mockReset().mockResolvedValue({
      filename: 'switch.raw',
      plot_name: 'Transient Analysis',
      sample_count: 100,
      time_trace_name: 'time',
      trace_names: ['time', 'V(ds)', 'I(M1)', 'V(other)']
    })
    previewDPTRawTraces.mockReset().mockResolvedValue({
      sample_count: 100,
      preview_sample_count: 3,
      time_range_s: [0, 99e-6],
      traces: {
        'V(ds)': [[0, 400], [50e-6, 200], [99e-6, 20]],
        'I(M1)': [[0, 0], [50e-6, 20], [99e-6, 35]]
      }
    })
    importDPTRawMeasurement.mockReset().mockResolvedValue({
      on_sample_count: 3, off_sample_count: 4
    })
    previewDPTEnergy.mockReset().mockResolvedValue({
      loss_type: 'e_on',
      event: 'turn_on',
      points: [
        {
          measurement_index: 0, acquisition_index: 0, i_x: 10, energy_j: 0.0014,
          start_s: 0, end_s: 2e-6, sample_count: 3, vds_sign: 1, id_sign: -1
        },
        {
          measurement_index: 0, acquisition_index: 1, i_x: 20, energy_j: 0.0028,
          start_s: 1e-6, end_s: 3e-6, sample_count: 4, vds_sign: -1, id_sign: 1
        }
      ],
      graph_i_e: [[10, 20], [0.0014, 0.0028]],
      conditions: {
        t_j: 25, v_supply: 400, v_g: 15, v_g_off: -5, r_g: 10, r_g_off: 12
      },
      integration: {
        quantity: 'raw-window switching energy',
        formula: 'E = sum of signed trapezoids',
        power_formula: 'p(t) = (Vds_sign * Vds) * (Id_sign * Id)',
        interval: 'exact stored first-to-last event sample time',
        boundary_detection: 'none; imported user-selected windows are used as stored'
      },
      save_action: 'add',
      existing_curve_index: null,
      target_curve_count: 0,
      target_edit_token: null,
      source_token: 'a'.repeat(64),
      request_token: 'b'.repeat(64)
    })
    saveDPTEnergy.mockReset().mockResolvedValue({
      action: 'added', loss_type: 'e_on', curve_index: 0, point_count: 2
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
    expect(getDPTData).toHaveBeenCalledWith('Waveform Device', {
      waveformPreviewPoints: 800
    })
    expect(wrapper.text()).toContain('Supply voltage400 V')
    expect(wrapper.text()).toContain('Junction temperature25 °C')
    expect(wrapper.text()).toContain('Test benchLocal test bench')
    expect(wrapper.text()).toContain('Load inductance0.75 µH')
    expect(wrapper.text()).toContain('Commutation inductance0.125 µH')
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
    expect(wrapper.get('[data-testid="dpt-waveform-preview-notice"]').text())
      .toContain('at most 800 points each')
    expect(wrapper.get('[data-testid="dpt-waveform-preview-notice"]').text())
      .toContain('capped at 40000 sample points')
    expect(wrapper.find('.curve-plotter-stub[data-title="Turn-off Id"]').text())
      .toContain('{"x":0,"y":null}')
    const waveformDataset = JSON.parse(
      wrapper.find('.curve-plotter-stub[data-title="Turn-on Vds"] output').text()
    )[0]
    expect(waveformDataset.pointRadius).toBe(0)
  })

  it('previews exact event intervals and requires explicit confirmation before saving', async () => {
    const section = wrapper.get('[data-testid="dpt-energy-derivation"]')
    await section.get('[data-testid="dpt-energy-record-0"]').setValue('0')
    await section.get('[data-testid="dpt-energy-acquisition-0"]').setValue('0')
    await section.get('[data-testid="dpt-energy-current-0"]').setValue('10')
    await section.get('[data-testid="dpt-energy-vds-sign-0"]').setValue('1')
    await section.get('[data-testid="dpt-energy-id-sign-0"]').setValue('-1')
    await section.get('[data-testid="dpt-energy-record-1"]').setValue('0')
    await section.get('[data-testid="dpt-energy-acquisition-1"]').setValue('1')
    await section.get('[data-testid="dpt-energy-current-1"]').setValue('20')
    await section.get('[data-testid="dpt-energy-vds-sign-1"]').setValue('-1')
    await section.get('[data-testid="dpt-energy-id-sign-1"]').setValue('1')
    await section.get('[data-testid="dpt-energy-preview"]').trigger('click')
    await flushPromises()

    expect(previewDPTEnergy).toHaveBeenCalledWith('Waveform Device', {
      loss_type: 'e_on',
      points: [
        { measurement_index: 0, acquisition_index: 0, i_x: 10, vds_sign: 1, id_sign: -1 },
        { measurement_index: 0, acquisition_index: 1, i_x: 20, vds_sign: -1, id_sign: 1 }
      ]
    })
    expect(section.get('[data-testid="dpt-energy-preview-result"]').text())
      .toContain('Create a new curve')
    expect(section.get('[data-testid="dpt-energy-preview-result"]').text())
      .toContain('exact stored first-to-last event sample time')
    expect(section.get('[data-testid="dpt-energy-preview-result"]').text())
      .toContain('[0, 0.000002]')
    expect(section.text()).toContain('0.0014')
    expect(section.find('[data-testid="dpt-energy-save"]').exists()).toBe(true)

    await section.get('[data-testid="dpt-energy-confirm"]').setValue(true)
    expect(section.get('[data-testid="dpt-energy-save"]').element.disabled).toBe(false)
    await section.get('[data-testid="dpt-energy-save"]').trigger('click')
    await flushPromises()

    expect(saveDPTEnergy).toHaveBeenCalledWith('Waveform Device', {
      loss_type: 'e_on',
      points: [
        { measurement_index: 0, acquisition_index: 0, i_x: 10, vds_sign: 1, id_sign: -1 },
        { measurement_index: 0, acquisition_index: 1, i_x: 20, vds_sign: -1, id_sign: 1 }
      ],
      source_token: 'a'.repeat(64),
      request_token: 'b'.repeat(64),
      save_action: 'add',
      existing_curve_index: null,
      target_curve_count: 0,
      target_edit_token: null
    })
    expect(getDPTData).toHaveBeenCalledTimes(2)
    expect(section.text()).toContain('Added E_on curve #1 with 2 measured points.')
  })

  it('invalidates a reviewed energy preview when a source selection changes', async () => {
    const section = wrapper.get('[data-testid="dpt-energy-derivation"]')
    for (const index of [0, 1]) {
      await section.get(`[data-testid="dpt-energy-record-${index}"]`).setValue('0')
      await section.get(`[data-testid="dpt-energy-acquisition-${index}"]`).setValue(String(index))
      await section.get(`[data-testid="dpt-energy-current-${index}"]`).setValue(String((index + 1) * 10))
      await section.get(`[data-testid="dpt-energy-vds-sign-${index}"]`).setValue('1')
      await section.get(`[data-testid="dpt-energy-id-sign-${index}"]`).setValue('1')
    }
    await section.get('[data-testid="dpt-energy-preview"]').trigger('click')
    await flushPromises()
    expect(section.find('[data-testid="dpt-energy-preview-result"]').exists()).toBe(true)
    expect(section.find('[data-testid="dpt-energy-save"]').exists()).toBe(true)
    await section.get('[data-testid="dpt-energy-current-0"]').setValue('11')
    expect(section.find('[data-testid="dpt-energy-preview-result"]').exists()).toBe(false)
    expect(section.find('[data-testid="dpt-energy-save"]').exists()).toBe(false)
  })

  it('invalidates the reviewed energy preview after a stale-target save conflict', async () => {
    const section = wrapper.get('[data-testid="dpt-energy-derivation"]')
    for (const index of [0, 1]) {
      await section.get(`[data-testid="dpt-energy-record-${index}"]`).setValue('0')
      await section.get(`[data-testid="dpt-energy-acquisition-${index}"]`).setValue(String(index))
      await section.get(`[data-testid="dpt-energy-current-${index}"]`).setValue(String((index + 1) * 10))
      await section.get(`[data-testid="dpt-energy-vds-sign-${index}"]`).setValue('1')
      await section.get(`[data-testid="dpt-energy-id-sign-${index}"]`).setValue('1')
    }
    await section.get('[data-testid="dpt-energy-preview"]').trigger('click')
    await flushPromises()
    await section.get('[data-testid="dpt-energy-confirm"]').setValue(true)
    expect(section.find('[data-testid="dpt-energy-preview-result"]').exists()).toBe(true)

    saveDPTEnergy.mockRejectedValueOnce({
      response: {
        status: 409,
        data: { detail: 'The switching curve target changed after preview.' }
      }
    })
    await section.get('[data-testid="dpt-energy-save"]').trigger('click')
    await flushPromises()

    expect(section.find('[data-testid="dpt-energy-preview-result"]').exists()).toBe(false)
    expect(section.find('[data-testid="dpt-energy-save"]').exists()).toBe(false)
    expect(section.get('[role="alert"]').text()).toContain(
      'The switching curve target changed after preview.'
    )
  })

  it('ignores a DPT response that completes after switching devices', async () => {
    const pendingResponse = deferred()
    const currentDeviceResponse = {
      has_e_on: false,
      has_e_off: false,
      has_e_rr: false,
      metadata: null,
      raw_measurement_data: [{
        metadata: {
          dataset_type: 'dpt_u_i',
          comment: 'Second device current response'
        },
        dpt_on_vds: [],
        dpt_on_id: [],
        dpt_off_vds: [],
        dpt_off_id: []
      }]
    }
    const staleDeviceResponse = structuredClone(currentDeviceResponse)
    staleDeviceResponse.raw_measurement_data[0].metadata.comment = 'Stale first device response'
    getDPTData.mockReset()
      .mockReturnValueOnce(pendingResponse.promise)
      .mockResolvedValueOnce(currentDeviceResponse)

    const oldRequest = wrapper.vm.loadDPTData()
    await wrapper.get('.transistor-selector select').setValue('Second Device')
    await flushPromises()

    expect(getDPTData).toHaveBeenNthCalledWith(2, 'Second Device', {
      waveformPreviewPoints: 800
    })
    expect(wrapper.text()).toContain('Second device current response')

    pendingResponse.resolve(staleDeviceResponse)
    await oldRequest
    await flushPromises()

    expect(wrapper.text()).toContain('Second device current response')
    expect(wrapper.text()).not.toContain('Stale first device response')
  })

  it('previews explicit traces and appends manually selected event windows', async () => {
    const input = wrapper.get('[data-testid="dpt-raw-file"]')
    const file = new File(['native raw bytes'], 'switch.raw', {
      type: 'application/octet-stream'
    })
    Object.defineProperty(input.element, 'files', { value: [file], configurable: true })
    await input.trigger('change')
    await wrapper.get('[data-testid="dpt-raw-preview-file"]').trigger('click')
    await flushPromises()

    expect(previewDPTRawImport).toHaveBeenCalledWith('Waveform Device', file)
    expect(wrapper.get('[data-testid="dpt-raw-file-preview"]').text())
      .toContain('100 samples')
    expect(wrapper.get('[data-testid="dpt-raw-vds-trace"]').findAll('option'))
      .toHaveLength(4)

    await wrapper.get('[data-testid="dpt-raw-vds-trace"]').setValue('V(ds)')
    await wrapper.get('[data-testid="dpt-raw-id-trace"]').setValue('I(M1)')
    await wrapper.get('[data-testid="dpt-raw-preview-traces"]').trigger('click')
    await flushPromises()

    expect(previewDPTRawTraces).toHaveBeenCalledWith('Waveform Device', file, {
      time_trace: 'time', vds_trace: 'V(ds)', id_trace: 'I(M1)'
    })
    expect(wrapper.find('.curve-plotter-stub[data-title="Selected Vds trace preview"]')
      .exists()).toBe(true)
    expect(wrapper.get('[data-testid="dpt-raw-import"]').text())
      .toContain('Load inductance [µH]')
    expect(wrapper.get('[data-testid="dpt-raw-import"]').text())
      .toContain('Commutation inductance [µH]')
    expect(wrapper.get('[data-testid="dpt-raw-import-submit"]').element.disabled).toBe(true)

    await wrapper.get('[data-testid="dpt-raw-on-start"]').setValue('0.000001')
    await wrapper.get('[data-testid="dpt-raw-on-end"]').setValue('0.000003')
    await wrapper.get('[data-testid="dpt-raw-off-start"]').setValue('0.000004')
    await wrapper.get('[data-testid="dpt-raw-off-end"]').setValue('0.000007')
    expect(wrapper.get('[data-testid="dpt-raw-import-submit"]').element.disabled).toBe(false)
    await wrapper.get('[data-testid="dpt-raw-import-submit"]').trigger('click')
    await flushPromises()

    expect(importDPTRawMeasurement).toHaveBeenCalledWith('Waveform Device', file, {
      time_trace: 'time',
      vds_trace: 'V(ds)',
      id_trace: 'I(M1)',
      turn_on: { start_s: 1e-6, end_s: 3e-6 },
      turn_off: { start_s: 4e-6, end_s: 7e-6 },
      metadata: {}
    })
    expect(wrapper.text()).toContain('Added one DPT record with 3 turn-on and 4 turn-off samples')
    expect(getDPTData).toHaveBeenCalledTimes(2)
  })

  it('ignores a pending file preview after the selected file changes', async () => {
    const pendingPreview = deferred()
    previewDPTRawImport.mockReturnValueOnce(pendingPreview.promise)
    const firstFile = new File(['first raw'], 'first.raw')
    const secondFile = new File(['second raw'], 'second.raw')

    await chooseRawFile(wrapper, firstFile)
    await wrapper.get('[data-testid="dpt-raw-preview-file"]').trigger('click')
    expect(previewDPTRawImport).toHaveBeenCalledWith('Waveform Device', firstFile)

    await chooseRawFile(wrapper, secondFile)
    pendingPreview.resolve({
      filename: 'first.raw',
      plot_name: 'Transient Analysis',
      sample_count: 100,
      time_trace_name: 'time',
      trace_names: ['time', 'V(ds)', 'I(M1)']
    })
    await flushPromises()

    expect(wrapper.vm.rawImportFile.name).toBe(secondFile.name)
    expect(wrapper.find('[data-testid="dpt-raw-file-preview"]').exists()).toBe(false)
    expect(wrapper.vm.rawImportPreview).toBe(null)
    expect(wrapper.find('[data-testid="dpt-raw-import-submit"]').exists()).toBe(false)
    expect(wrapper.get('[data-testid="dpt-raw-preview-file"]').element.disabled).toBe(false)
    await wrapper.get('[data-testid="dpt-raw-preview-file"]').trigger('click')
    await flushPromises()
    expect(previewDPTRawImport).toHaveBeenLastCalledWith('Waveform Device', secondFile)
    expect(wrapper.get('[data-testid="dpt-raw-file-preview"]').text())
      .toContain('switch.raw')
  })

  it('ignores a pending trace preview after the trace mapping changes', async () => {
    const file = new File(['native raw bytes'], 'switch.raw')
    await loadRawHeader(wrapper, file)
    const pendingPreview = deferred()
    previewDPTRawTraces.mockReturnValueOnce(pendingPreview.promise)

    await wrapper.get('[data-testid="dpt-raw-preview-traces"]').trigger('click')
    expect(previewDPTRawTraces).toHaveBeenCalledWith('Waveform Device', file, {
      time_trace: 'time', vds_trace: 'V(ds)', id_trace: 'I(M1)'
    })

    await wrapper.get('[data-testid="dpt-raw-vds-trace"]').setValue('V(other)')
    pendingPreview.resolve({
      sample_count: 100,
      preview_sample_count: 3,
      time_range_s: [0, 99e-6],
      traces: { 'V(ds)': [[0, 400]], 'I(M1)': [[0, 0]] }
    })
    await flushPromises()

    expect(wrapper.find('[data-testid="dpt-raw-waveform-preview"]').exists()).toBe(false)
    expect(wrapper.find('[data-testid="dpt-raw-import-submit"]').exists()).toBe(false)
    await wrapper.vm.importRawDptMeasurement()
    expect(importDPTRawMeasurement).not.toHaveBeenCalled()
  })

  it('ignores a pending file preview after the selected device changes', async () => {
    const pendingPreview = deferred()
    previewDPTRawImport.mockReturnValueOnce(pendingPreview.promise)
    const file = new File(['native raw bytes'], 'switch.raw')
    await chooseRawFile(wrapper, file)
    wrapper.vm.rawImportMetadata.t_j = '125'
    await wrapper.get('[data-testid="dpt-raw-preview-file"]').trigger('click')

    await wrapper.get('.transistor-selector select').setValue('Second Device')
    await flushPromises()
    pendingPreview.resolve({
      filename: 'switch.raw',
      plot_name: 'Transient Analysis',
      sample_count: 100,
      time_trace_name: 'time',
      trace_names: ['time', 'V(ds)', 'I(M1)']
    })
    await flushPromises()

    expect(wrapper.vm.rawImportFile).toBe(null)
    expect(wrapper.vm.rawImportPreview).toBe(null)
    expect(wrapper.vm.rawImportTracePreview).toBe(null)
    expect(wrapper.vm.rawImportForm.turn_on_start_s).toBe('')
    expect(wrapper.vm.rawImportForm.turn_off_end_s).toBe('')
    expect(wrapper.vm.rawImportMetadata.t_j).toBe('')
    expect(wrapper.find('[data-testid="dpt-raw-file-preview"]').exists()).toBe(false)
  })

  it('clears reviewed file, trace, windows, and metadata on device change', async () => {
    const file = new File(['native raw bytes'], 'switch.raw')
    await loadRawHeader(wrapper, file)
    await wrapper.get('[data-testid="dpt-raw-preview-traces"]').trigger('click')
    await flushPromises()
    await wrapper.get('[data-testid="dpt-raw-on-start"]').setValue('0.000001')
    await wrapper.get('[data-testid="dpt-raw-on-end"]').setValue('0.000003')
    await wrapper.get('[data-testid="dpt-raw-off-start"]').setValue('0.000004')
    await wrapper.get('[data-testid="dpt-raw-off-end"]').setValue('0.000007')
    await wrapper.get('[data-testid="dpt-raw-metadata-tj"]').setValue('125')

    expect(wrapper.find('[data-testid="dpt-raw-waveform-preview"]').exists()).toBe(true)
    expect(wrapper.get('[data-testid="dpt-raw-import-submit"]').element.disabled)
      .toBe(false)

    await wrapper.get('.transistor-selector select').setValue('Second Device')
    await flushPromises()

    expect(wrapper.vm.rawImportFile).toBe(null)
    expect(wrapper.vm.rawImportPreview).toBe(null)
    expect(wrapper.vm.rawImportTracePreview).toBe(null)
    expect(wrapper.vm.rawImportForm.turn_on_start_s).toBe('')
    expect(wrapper.vm.rawImportForm.turn_off_end_s).toBe('')
    expect(wrapper.vm.rawImportMetadata.t_j).toBe('')
    expect(wrapper.find('[data-testid="dpt-raw-file-preview"]').exists()).toBe(false)
    expect(wrapper.find('[data-testid="dpt-raw-waveform-preview"]').exists()).toBe(false)
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
