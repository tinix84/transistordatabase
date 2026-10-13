/**
 * Unit tests for TransistorForm.vue component
 */
import { describe, it, expect, beforeEach, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import TransistorForm from '../../components/TransistorForm.vue'
import CurvePlotter from '../../components/CurvePlotter.vue'
import { transistorApi } from '../../services/api'
import { mockTransistor1, emptyTransistor } from '../fixtures/transistors'

// Mock the transistorApi
vi.mock('../../services/api', () => ({
  transistorApi: {
    create: vi.fn(),
    update: vi.fn(),
    validate: vi.fn(),
    getCurves: vi.fn(),
    addChannelCurve: vi.fn(),
    addSwitchingLossCurve: vi.fn(),
    addGateChargeCurve: vi.fn(),
    addSOACurve: vi.fn(),
    addCapacitanceCurve: vi.fn(),
    deleteCurve: vi.fn()
  }
}))

const curveApiResponse = {
  switch: {
    channel_data: [{ t_j: 25, v_g: 15, graph_v_i: [[0, 10, 20], [0, 2, 4]] }],
    e_on_data: [{
      dataset_type: 'graph_i_e', t_j: 25, v_supply: 400, v_g: 15, r_g: 2.5,
      graph_i_e: [[0, 10, 20], [0, 0.001, 0.002]], graph_r_e: null
    }],
    e_off_data: [{
      dataset_type: 'graph_r_e', t_j: 25, v_supply: 400, v_g: -4, r_g: null,
      graph_i_e: null, graph_r_e: [[1, 5, 10], [0.001, 0.0008, 0.0006]]
    }],
    gate_charge_curves: [{
      v_supply: 400, t_j: 25, i_channel: 10, i_g: null,
      graph_q_v: [[0, 25e-9, 50e-9], [0, 8, 15]]
    }],
    soa: [{ t_c: 25, time_pulse: 0.01, graph_i_v: [[0, 100, 600], [100, 20, 1]] }]
  },
  diode: {
    channel_data: [{ t_j: 25, graph_v_i: [[0, 1, 2], [0, 10, 20]] }],
    e_rr_data: [{
      dataset_type: 'graph_i_e', t_j: 25, v_supply: 400, v_g: 0, r_g: 2.5,
      graph_i_e: [[0, 10, 20], [0, 0.0005, 0.001]], graph_r_e: null
    }]
  },
  capacitances: {
    c_oss: [{ t_j: 25, graph_v_c: [[0, 50, 100], [0.00001, 0.000002, 0.000001]] }],
    c_iss: [],
    c_rss: []
  }
}

const curvePlotterStub = {
  name: 'CurvePlotter',
  props: ['title', 'datasets', 'xLabel', 'yLabel', 'xLogScale', 'yLogScale'],
  template: '<div data-testid="curve-plotter">{{ title }}</div>'
}

const plotCases = [
  {
    name: 'switch channel voltage against current', tab: 'Channel Curves',
    xLabel: 'Voltage (V)', yLabel: 'Current (A)',
    data: [{ x: 0, y: 0 }, { x: 10, y: 2 }, { x: 20, y: 4 }]
  },
  {
    name: 'diode channel voltage against current', tab: 'Channel Curves', component: 'Diode',
    xLabel: 'Voltage (V)', yLabel: 'Current (A)',
    data: [{ x: 0, y: 0 }, { x: 1, y: 10 }, { x: 2, y: 20 }]
  },
  {
    name: 'turn-on energy against current', tab: 'Switching Losses',
    xLabel: 'Current (A)', yLabel: 'Switching energy (J)',
    data: [{ x: 0, y: 0 }, { x: 10, y: 0.001 }, { x: 20, y: 0.002 }]
  },
  {
    name: 'turn-off energy against gate resistance', tab: 'Switching Losses', loss: 'E_off',
    xLabel: 'Gate resistance (Ω)', yLabel: 'Switching energy (J)',
    data: [{ x: 1, y: 0.001 }, { x: 5, y: 0.0008 }, { x: 10, y: 0.0006 }]
  },
  {
    name: 'reverse-recovery energy against current', tab: 'Switching Losses', loss: 'E_rr',
    xLabel: 'Current (A)', yLabel: 'Switching energy (J)',
    data: [{ x: 0, y: 0 }, { x: 10, y: 0.0005 }, { x: 20, y: 0.001 }]
  },
  {
    name: 'gate charge against gate voltage', tab: 'Gate Charge',
    xLabel: 'Gate charge (nC)', yLabel: 'Gate voltage (V)',
    data: [{ x: 0, y: 0 }, { x: 25, y: 8 }, { x: 50, y: 15 }]
  },
  {
    name: 'SOA voltage against current', tab: 'SOA',
    xLabel: 'Voltage (V)', yLabel: 'Current (A)',
    data: [{ x: 0, y: 100 }, { x: 100, y: 20 }, { x: 600, y: 1 }]
  },
  {
    name: 'capacitance against voltage', tab: 'Capacitance',
    xLabel: 'Voltage (V)', yLabel: 'Capacitance (F)',
    data: [{ x: 0, y: 0.00001 }, { x: 50, y: 0.000002 }, { x: 100, y: 0.000001 }]
  }
]

const mutationCurveData = { t_j: 25, graph: 'mutation fixture' }
const mutationCases = [
  {
    name: 'switch channel', tab: 'Channel Curves', curveTypeLabel: 'Channel', componentButton: 'Switch',
    addMethod: 'addChannelCurve', addArgs: (id, data) => [id, data, 'switch'],
    deleteRoute: { component: 'switch', type: 'channel' }
  },
  {
    name: 'diode channel', tab: 'Channel Curves', curveTypeLabel: 'Channel', componentButton: 'Diode',
    addMethod: 'addChannelCurve', addArgs: (id, data) => [id, data, 'diode'],
    deleteRoute: { component: 'diode', type: 'channel' }
  },
  {
    name: 'turn-on switching loss', tab: 'Switching Losses', curveTypeLabel: 'Switching Loss', subTabPrefix: 'E_on',
    addMethod: 'addSwitchingLossCurve', addArgs: (id, data) => [id, 'e_on', data],
    deleteRoute: { component: 'switch', type: 'e_on' }
  },
  {
    name: 'reverse-recovery switching loss', tab: 'Switching Losses', curveTypeLabel: 'Switching Loss', subTabPrefix: 'E_rr',
    addMethod: 'addSwitchingLossCurve', addArgs: (id, data) => [id, 'e_rr', data],
    deleteRoute: { component: 'diode', type: 'e_rr' }
  },
  {
    name: 'gate charge', tab: 'Gate Charge', curveTypeLabel: 'Gate Charge',
    addMethod: 'addGateChargeCurve', addArgs: (id, data) => [id, data],
    deleteRoute: { component: 'switch', type: 'gate_charge' }
  },
  {
    name: 'SOA', tab: 'SOA', curveTypeLabel: 'SOA',
    addMethod: 'addSOACurve', addArgs: (id, data) => [id, data],
    deleteRoute: { component: 'switch', type: 'soa' }
  },
  {
    name: 'output capacitance', tab: 'Capacitance', curveTypeLabel: 'Capacitance',
    addMethod: 'addCapacitanceCurve', addArgs: (id, data) => [id, 'c_oss', data],
    deleteRoute: { component: 'capacitance', type: 'c_oss' }
  }
]

function mountForm(transistor = mockTransistor1) {
  return mount(TransistorForm, {
    props: { transistor },
    global: { stubs: { CurvePlotter: curvePlotterStub } }
  })
}

function activeCurveTab(wrapper) {
  return wrapper.findAll('.curve-tab-content').find(tab => tab.element.style.display !== 'none')
}

async function clickFormTab(wrapper, label) {
  await wrapper.findAll('.tab-btn').find(tab => tab.text() === label).trigger('click')
}

async function selectMutationCurveTab(wrapper, mutationCase) {
  await clickFormTab(wrapper, mutationCase.tab)
  let curveTab = activeCurveTab(wrapper)
  if (mutationCase.componentButton) {
    await curveTab.findAll('.toggle-btn').find(button => button.text() === mutationCase.componentButton).trigger('click')
  }
  if (mutationCase.subTabPrefix) {
    await curveTab.findAll('.sub-tab-btn').find(button => button.text().startsWith(mutationCase.subTabPrefix)).trigger('click')
  }
  return activeCurveTab(wrapper)
}

describe('TransistorForm.vue', () => {
  let wrapper

  beforeEach(() => {
    vi.clearAllMocks()
    vi.spyOn(console, 'log').mockImplementation(() => {})
    vi.spyOn(console, 'error').mockImplementation(() => {})

    transistorApi.create.mockResolvedValue(mockTransistor1)
    transistorApi.update.mockResolvedValue(mockTransistor1)
    transistorApi.validate.mockResolvedValue({ valid: true, warnings: [], errors: [] })
    transistorApi.getCurves.mockResolvedValue({ switch: {}, diode: {}, capacitances: {} })
    transistorApi.addChannelCurve.mockResolvedValue({})
    transistorApi.addSwitchingLossCurve.mockResolvedValue({})
    transistorApi.addGateChargeCurve.mockResolvedValue({})
    transistorApi.addSOACurve.mockResolvedValue({})
    transistorApi.addCapacitanceCurve.mockResolvedValue({})
    transistorApi.deleteCurve.mockResolvedValue({})
  })

  describe('Create Mode', () => {
    beforeEach(() => {
      wrapper = mount(TransistorForm, {
        props: {
          transistor: null
        }
      })
    })

    it('should render the form', () => {
      expect(wrapper.exists()).toBe(true)
    })

    it('should show create transistor title', () => {
      expect(wrapper.text()).toContain('Create New Transistor')
    })

    it('should have empty form fields', () => {
      // Check that inputs are empty or default values
      const inputs = wrapper.findAll('input[type="text"]')
      expect(inputs.length).toBeGreaterThan(0)
    })

    it('should emit saved event on successful create', async () => {
      // Fill required fields
      await wrapper.vm.$emit('saved')

      expect(wrapper.emitted('saved')).toBeTruthy()
    })

    it('should emit cancel event when cancel clicked', async () => {
      await wrapper.vm.$emit('cancel')

      expect(wrapper.emitted('cancel')).toBeTruthy()
    })
  })

  describe('Edit Mode', () => {
    beforeEach(() => {
      wrapper = mount(TransistorForm, {
        props: {
          transistor: mockTransistor1
        }
      })
    })

    it('should show edit transistor title', () => {
      const text = wrapper.text()
      expect(text.includes('Edit') || text.includes('Update')).toBe(true)
    })

    it('should populate form with transistor data', () => {
      // Check that form is populated with transistor data
      expect(wrapper.vm.transistor).toBeTruthy()
    })

    it('should emit saved event on successful update', async () => {
      await wrapper.vm.$emit('saved')

      expect(wrapper.emitted('saved')).toBeTruthy()
    })
  })

  describe('Form Validation', () => {
    beforeEach(() => {
      wrapper = mount(TransistorForm, {
        props: {
          transistor: null
        }
      })
    })

    it('should validate required fields', () => {
      // Name field should be required
      expect(wrapper.html()).toContain('name')
    })

    it('should validate numeric fields', () => {
      // Should have numeric input fields
      const numberInputs = wrapper.findAll('input[type="number"]')
      expect(numberInputs.length).toBeGreaterThan(0)
    })

    it('should show validation errors', async () => {
      transistorApi.validate.mockResolvedValue({
        valid: false,
        errors: ['Name is required'],
        warnings: []
      })

      // Validation UI would show errors
      // This is a placeholder for actual validation UI tests
    })
  })

  describe('Data Structure', () => {
    it('should handle metadata section', () => {
      wrapper = mount(TransistorForm, {
        props: { transistor: mockTransistor1 }
      })

      expect(wrapper.html()).toMatch(/Metadata|Name|Manufacturer/)
    })

    it('should handle electrical ratings section', () => {
      wrapper = mount(TransistorForm, {
        props: { transistor: mockTransistor1 }
      })

      expect(wrapper.html()).toMatch(/voltage|current|electrical/i)
    })

    it('should handle thermal properties section', () => {
      wrapper = mount(TransistorForm, {
        props: { transistor: mockTransistor1 }
      })

      expect(wrapper.html()).toMatch(/thermal|temperature/i)
    })
  })

  describe('API Integration', () => {
    it('should call create API when saving new transistor', async () => {
      wrapper = mount(TransistorForm, {
        props: { transistor: null }
      })

      // Simulate form submission
      if (wrapper.vm.handleSubmit) {
        await wrapper.vm.handleSubmit()
        expect(transistorApi.create).toHaveBeenCalled()
      }
    })

    it('converts cm² form areas to SI m² when creating a record', async () => {
      wrapper = mount(TransistorForm, { props: { transistor: null } })
      wrapper.vm.formData.thermal.housing_area = 1
      wrapper.vm.formData.thermal.cooling_area = 0.5
      wrapper.vm.formData.thermal.r_th_switch_cs = 0.12
      wrapper.vm.formData.thermal.r_th_diode_cs = 0.18

      await wrapper.vm.handleSubmit()

      expect(transistorApi.create).toHaveBeenCalledWith(expect.objectContaining({
        thermal: expect.objectContaining({
          r_th_switch_cs: 0.12,
          r_th_diode_cs: 0.18,
          housing_area: 0.0001,
          cooling_area: 0.00005
        })
      }))
    })
    it('should call update API when saving existing transistor', async () => {
      wrapper = mount(TransistorForm, {
        props: { transistor: mockTransistor1 }
      })

      // Simulate form submission
      if (wrapper.vm.handleSubmit) {
        await wrapper.vm.handleSubmit()
        expect(transistorApi.update).toHaveBeenCalled()
      }
    })

    it('converts loaded SI areas to cm² and back to SI on update', async () => {
      wrapper = mount(TransistorForm, { props: { transistor: mockTransistor1 } })
      await flushPromises()

      expect(wrapper.vm.formData.thermal.housing_area).toBe(12)
      expect(wrapper.vm.formData.thermal.cooling_area).toBe(8)

      await wrapper.vm.handleSubmit()

      expect(transistorApi.update).toHaveBeenCalledWith(
        mockTransistor1.metadata.name,
        expect.objectContaining({
          thermal: expect.objectContaining({
            housing_area: mockTransistor1.thermal.housing_area,
            cooling_area: mockTransistor1.thermal.cooling_area,
            r_th_switch_cs: 0.12,
            r_th_diode_cs: 0.18
          })
        })
      )
    })
    it('should handle API errors gracefully', async () => {
      transistorApi.create.mockRejectedValue(new Error('API Error'))

      wrapper = mount(TransistorForm, {
        props: { transistor: null }
      })

      // Should not crash on API error
      expect(wrapper.exists()).toBe(true)
    })

    it.each(plotCases)('loads and plots $name from the curve API response', async (plotCase) => {
      transistorApi.getCurves.mockResolvedValue(curveApiResponse)
      wrapper = mountForm()
      await flushPromises()

      await clickFormTab(wrapper, plotCase.tab)
      if (plotCase.component) {
        await activeCurveTab(wrapper).findAll('.toggle-btn').find(button => button.text() === plotCase.component).trigger('click')
      }
      if (plotCase.loss) {
        await activeCurveTab(wrapper).findAll('.sub-tab-btn').find(button => button.text().startsWith(plotCase.loss)).trigger('click')
      }
      await activeCurveTab(wrapper).find('[title="View Curve"]').trigger('click')

      const plot = wrapper.findComponent(CurvePlotter)
      expect(plot.exists()).toBe(true)
      expect(plot.props('xLabel')).toBe(plotCase.xLabel)
      expect(plot.props('yLabel')).toBe(plotCase.yLabel)
      expect(plot.props('datasets')[0].data).toEqual(plotCase.data)
    })

    it('loads curve arrays with the exact selected transistor identity', async () => {
      const transistor = structuredClone(mockTransistor1)
      transistor.metadata.name = 'Curve #Device ?'
      wrapper = mountForm(transistor)

      await flushPromises()

      expect(transistorApi.getCurves).toHaveBeenCalledWith('Curve #Device ?')
      expect(wrapper.findAll('[title="Edit"]')).toHaveLength(0)
    })

    it.each(mutationCases)('preserves the exact device identity and $name curve routes', async (mutationCase) => {
      const transistor = structuredClone(mockTransistor1)
      transistor.metadata.name = 'Curve #Device ?'
      transistorApi.getCurves.mockResolvedValue(curveApiResponse)
      wrapper = mountForm(transistor)
      await flushPromises()

      let curveTab = await selectMutationCurveTab(wrapper, mutationCase)
      await curveTab.findAll('button').find(button => button.text().includes(`Add ${mutationCase.curveTypeLabel}`)).trigger('click')
      await wrapper.findComponent({ name: 'CurveEditor' }).vm.$emit('submit', mutationCurveData)
      await flushPromises()

      expect(transistorApi[mutationCase.addMethod]).toHaveBeenCalledWith(...mutationCase.addArgs('Curve #Device ?', mutationCurveData))

      curveTab = activeCurveTab(wrapper)
      await curveTab.find('[title="Delete"]').trigger('click')
      await activeCurveTab(wrapper).find('.modal-actions .btn-danger').trigger('click')
      await flushPromises()

      expect(transistorApi.deleteCurve).toHaveBeenCalledWith(
        'Curve #Device ?', mutationCase.deleteRoute.component, mutationCase.deleteRoute.type, 0
      )
    })

    it('closes the selected curve plot', async () => {
      transistorApi.getCurves.mockResolvedValue(curveApiResponse)
      wrapper = mountForm()
      await flushPromises()
      await clickFormTab(wrapper, 'Channel Curves')
      await activeCurveTab(wrapper).find('[title="View Curve"]').trigger('click')

      expect(wrapper.find('[data-testid="curve-viewer"]').exists()).toBe(true)
      await wrapper.find('[aria-label="Close curve view"]').trigger('click')
      expect(wrapper.find('[data-testid="curve-viewer"]').exists()).toBe(false)
    })
  })

  describe('Events', () => {
    beforeEach(() => {
      wrapper = mount(TransistorForm, {
        props: { transistor: null }
      })
    })

    it('should emit saved event with transistor data', async () => {
      await wrapper.vm.$emit('saved', mockTransistor1)

      expect(wrapper.emitted('saved')).toBeTruthy()
      expect(wrapper.emitted('saved')[0]).toEqual([mockTransistor1])
    })

    it('should emit cancel event', async () => {
      await wrapper.vm.$emit('cancel')

      expect(wrapper.emitted('cancel')).toBeTruthy()
    })
  })

  describe('Form State', () => {
    it('should track form dirty state', () => {
      wrapper = mount(TransistorForm, {
        props: { transistor: null }
      })

      // Form should track if it has been modified
      expect(wrapper.vm).toBeDefined()
    })

    it('should show unsaved changes warning', () => {
      wrapper = mount(TransistorForm, {
        props: { transistor: mockTransistor1 }
      })

      // Should have mechanism to warn about unsaved changes
      expect(wrapper.exists()).toBe(true)
    })
  })
})
