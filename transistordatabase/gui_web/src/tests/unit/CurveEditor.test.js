import { describe, expect, it } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import CurveEditor from '../../components/CurveEditor.vue'

const graphCases = [
  {
    name: 'switch channel', curveType: 'channel', component: 'switch',
    initialData: { t_j: 125, v_g: 15, graph_v_i: [[0, 1, 2], [0, 10, 20]] },
    expected: { graph_v_i: [[0, 1, 2], [0, 10, 20]] }
  },
  {
    name: 'diode channel', curveType: 'channel', component: 'diode',
    initialData: { t_j: 125, graph_v_i: [[0, 1, 2], [0, 10, 20]] },
    expected: { graph_v_i: [[0, 1, 2], [0, 10, 20]] }
  },
  {
    name: 'Eon current graph', curveType: 'switching_loss', lossType: 'e_on',
    initialData: { dataset_type: 'graph_i_e', t_j: 125, v_supply: 400, v_g: 15, graph_i_e: [[0, 10], [0, 0.001]] },
    expected: { graph_i_e: [[0, 10], [0, 0.001]] }
  },
  {
    name: 'Eoff resistance graph', curveType: 'switching_loss', lossType: 'e_off',
    initialData: { dataset_type: 'graph_r_e', t_j: 125, v_supply: 400, v_g: -4, v_g_off: -4, r_g: 2, graph_r_e: [[1, 5], [0.001, 0.0005]] },
    expected: { graph_r_e: [[1, 5], [0.001, 0.0005]] }
  },
  {
    name: 'Err temperature graph', curveType: 'switching_loss', lossType: 'e_rr',
    initialData: { dataset_type: 'graph_t_e', t_j: null, v_supply: 400, v_g: 0, graph_t_e: [[25, 125], [0.0005, 0.0008]] },
    expected: { graph_t_e: [[25, 125], [0.0005, 0.0008]] }
  },
  {
    name: 'gate charge', curveType: 'gate_charge',
    initialData: { t_j: 125, v_supply: 400, i_channel: 10, i_g: 0.5, graph_q_v: [[0, 2e-8], [0, 10]] },
    expected: { graph_q_v: [[0, 2e-8], [0, 10]] }
  },
  {
    name: 'SOA', curveType: 'soa',
    initialData: { t_c: 75, time_pulse: 0.01, graph_i_v: [[0, 100], [20, 2]] },
    expected: { graph_i_v: [[0, 100], [20, 2]] }
  },
  ...['c_oss', 'c_iss', 'c_rss'].map(capacitanceType => ({
    name: capacitanceType,
    curveType: 'capacitance',
    capacitanceType,
    initialData: { t_j: 125, graph_v_c: [[0, 100], [2e-5, 1e-6]] },
    expected: { graph_v_c: [[0, 100], [2e-5, 1e-6]] }
  }))
]

function mountEditor(props) {
  return mount(CurveEditor, {
    props: { editMode: true, ...props },
    global: {
      stubs: {
        CurvePlotter: { template: '<div data-testid="curve-preview" />' },
        CurveUploader: { template: '<div />' }
      }
    }
  })
}

describe('CurveEditor saved-curve editing', () => {
  it.each(graphCases)('prefills and emits a raw patch for $name', async curveCase => {
    const { expected, ...props } = curveCase
    const wrapper = mountEditor(props)
    await flushPromises()

    expect(wrapper.vm.dataInputMethod).toBe('manual')
    const graphKey = Object.keys(expected).find(key => key.startsWith('graph_'))
    const [xValues, yValues] = expected[graphKey]
    expect(wrapper.vm.manualDataPoints).toEqual(xValues.map((x, index) => ({ x, y: yValues[index] })))

    await wrapper.find('.editor-footer .btn-primary').trigger('click')

    expect(wrapper.emitted('submit')?.[0]?.[0]).toEqual(expected)
  })

  it('allows a temperature-axis switching curve with null t_j and omits its t_j patch', async () => {
    const temperatureCurve = graphCases.find(curveCase => curveCase.name === 'Err temperature graph')
    const wrapper = mountEditor({
      curveType: temperatureCurve.curveType,
      lossType: temperatureCurve.lossType,
      initialData: temperatureCurve.initialData
    })
    await flushPromises()

    expect(wrapper.vm.formData.t_j).toBeNull()
    expect(wrapper.vm.canSubmit).toBe(true)
    expect(wrapper.findAll('label').some(label => label.text().includes('Junction Temperature'))).toBe(false)

    await wrapper.find('.editor-footer .btn-primary').trigger('click')

    expect(wrapper.emitted('submit')?.[0]?.[0]).toEqual(temperatureCurve.expected)
    expect(wrapper.emitted('submit')?.[0]?.[0]).not.toHaveProperty('t_j')
  })

  it('prefills and submits a single switching measurement without graph data', async () => {
    const wrapper = mountEditor({
      curveType: 'switching_loss',
      lossType: 'e_on',
      initialData: { dataset_type: 'single', t_j: 125, v_supply: 400, v_g: 15, i_x: 42, e_x: 0.002 }
    })
    await flushPromises()

    expect(wrapper.vm.isSingleSwitchingEdit).toBe(true)
    expect(wrapper.vm.formData.i_x).toBe(42)
    expect(wrapper.vm.formData.e_x).toBe(0.002)
    wrapper.vm.formData.i_x = 45
    await wrapper.find('.editor-footer .btn-primary').trigger('click')
    expect(wrapper.emitted('submit')?.[0]?.[0]).toEqual({ i_x: 45 })
  })

  it('submits only changed curve conditions with the graph patch', async () => {
    const { expected: _expected, ...props } = graphCases[0]
    const wrapper = mountEditor(props)
    await flushPromises()
    wrapper.vm.formData.t_j = 150
    wrapper.vm.formData.v_g = 18
    await wrapper.find('.editor-footer .btn-primary').trigger('click')
    expect(wrapper.emitted('submit')?.[0]?.[0]).toEqual({
      t_j: 150, v_g: 18, graph_v_i: [[0, 1, 2], [0, 10, 20]]
    })
  })

  it('emits cancel without submitting a patch', async () => {
    const wrapper = mountEditor(graphCases[0])
    await wrapper.find('.btn-secondary').trigger('click')
    expect(wrapper.emitted('cancel')).toHaveLength(1)
    expect(wrapper.emitted('submit')).toBeUndefined()
  })
})
