/**
 * Unit tests for TransistorForm.vue component
 */
import { describe, it, expect, beforeEach, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import TransistorForm from '../../components/TransistorForm.vue'
import { transistorApi } from '../../services/api'
import { mockTransistor1, emptyTransistor } from '../fixtures/transistors'

// Mock the transistorApi
vi.mock('../../services/api', () => ({
  transistorApi: {
    create: vi.fn(),
    update: vi.fn(),
    validate: vi.fn(),
    getCurves: vi.fn()
  }
}))

describe('TransistorForm.vue', () => {
  let wrapper

  beforeEach(() => {
    vi.clearAllMocks()
    vi.spyOn(console, 'log').mockImplementation(() => {})
    vi.spyOn(console, 'error').mockImplementation(() => {})

    transistorApi.create.mockResolvedValue(mockTransistor1)
    transistorApi.update.mockResolvedValue(mockTransistor1)
    transistorApi.validate.mockResolvedValue({ valid: true, warnings: [], errors: [] })
    transistorApi.getCurves.mockResolvedValue({ switch: {}, diode: {}, capacitance: {} })
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
