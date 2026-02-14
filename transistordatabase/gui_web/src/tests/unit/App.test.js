/**
 * Unit tests for App.vue component
 */
import { describe, it, expect, beforeEach, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import App from '../../App.vue'
import { transistorApi } from '../../services/api'
import { mockTransistors } from '../fixtures/transistors'

// Mock the transistorApi
vi.mock('../../services/api', () => ({
  transistorApi: {
    getAll: vi.fn(),
    delete: vi.fn()
  }
}))

// Mock child components to simplify testing
vi.mock('../../components/TransistorList.vue', () => ({
  default: { name: 'TransistorList', template: '<div>TransistorList</div>' }
}))
vi.mock('../../components/TransistorForm.vue', () => ({
  default: { name: 'TransistorForm', template: '<div>TransistorForm</div>' }
}))
vi.mock('../../components/TransistorComparison.vue', () => ({
  default: { name: 'TransistorComparison', template: '<div>TransistorComparison</div>' }
}))
vi.mock('../../components/TransistorPlotter.vue', () => ({
  default: { name: 'TransistorPlotter', template: '<div>TransistorPlotter</div>' }
}))
vi.mock('../../components/DatabaseManager.vue', () => ({
  default: { name: 'DatabaseManager', template: '<div>DatabaseManager</div>' }
}))
vi.mock('../../components/SearchDatabase.vue', () => ({
  default: { name: 'SearchDatabase', template: '<div>SearchDatabase</div>' }
}))
vi.mock('../../components/ExportingTools.vue', () => ({
  default: { name: 'ExportingTools', template: '<div>ExportingTools</div>' }
}))
vi.mock('../../components/TopologyCalculator.vue', () => ({
  default: { name: 'TopologyCalculator', template: '<div>TopologyCalculator</div>' }
}))

describe('App.vue', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    transistorApi.getAll.mockResolvedValue(mockTransistors)
    // Suppress console logs
    vi.spyOn(console, 'log').mockImplementation(() => {})
    vi.spyOn(console, 'error').mockImplementation(() => {})
  })

  describe('Component Mounting', () => {
    it('should render the app', () => {
      const wrapper = mount(App)
      expect(wrapper.exists()).toBe(true)
    })

    it('should display app title', () => {
      const wrapper = mount(App)
      expect(wrapper.text()).toContain('Transistor Database')
    })

    it('should load transistors on mount', async () => {
      mount(App)
      await flushPromises()

      expect(transistorApi.getAll).toHaveBeenCalled()
    })

    it('should display loading state initially', async () => {
      const wrapper = mount(App, {
        shallow: false
      })

      // Before promises resolve, should show loading
      expect(wrapper.vm.isLoading).toBe(true)

      await flushPromises()

      // After promises resolve, should not show loading
      expect(wrapper.vm.isLoading).toBe(false)
    })
  })

  describe('Data Loading', () => {
    it('should update transistors count after loading', async () => {
      const wrapper = mount(App)
      await flushPromises()

      expect(wrapper.vm.transistors).toHaveLength(3)
      expect(wrapper.text()).toContain('Total Transistors: 3')
    })

    it('should handle empty transistor list', async () => {
      transistorApi.getAll.mockResolvedValue([])

      const wrapper = mount(App)
      await flushPromises()

      expect(wrapper.vm.transistors).toHaveLength(0)
      expect(wrapper.text()).toContain('Total Transistors: 0')
    })

    it('should handle API errors gracefully', async () => {
      transistorApi.getAll.mockRejectedValue(new Error('API Error'))

      const wrapper = mount(App)
      await flushPromises()

      expect(wrapper.vm.transistors).toHaveLength(0)
      expect(console.error).toHaveBeenCalled()
    })
  })

  describe('Navigation', () => {
    it('should default to search view', async () => {
      const wrapper = mount(App)
      await flushPromises()

      expect(wrapper.vm.currentView).toBe('search')
    })

    it('should switch to create view', async () => {
      const wrapper = mount(App)
      await flushPromises()

      const createButton = wrapper.find('button:nth-of-type(2)')
      await createButton.trigger('click')

      expect(wrapper.vm.currentView).toBe('create')
    })

    it('should switch to export view', async () => {
      const wrapper = mount(App)
      await flushPromises()

      const exportButton = wrapper.find('button:nth-of-type(3)')
      await exportButton.trigger('click')

      expect(wrapper.vm.currentView).toBe('export')
    })

    it('should switch to comparison view', async () => {
      const wrapper = mount(App)
      await flushPromises()

      const compareButton = wrapper.find('button:nth-of-type(4)')
      await compareButton.trigger('click')

      expect(wrapper.vm.currentView).toBe('compare')
    })

    it('should switch to topology view', async () => {
      const wrapper = mount(App)
      await flushPromises()

      const topologyButton = wrapper.find('button:nth-of-type(5)')
      await topologyButton.trigger('click')

      expect(wrapper.vm.currentView).toBe('topology')
    })

    it('should mark active navigation button', async () => {
      const wrapper = mount(App)
      await flushPromises()

      const searchButton = wrapper.find('button:nth-of-type(1)')
      expect(searchButton.classes()).toContain('active')
    })
  })

  describe('Theme Toggle', () => {
    it('should default to light theme', async () => {
      const wrapper = mount(App)
      await flushPromises()

      expect(wrapper.vm.isDarkTheme).toBe(false)
    })

    it('should toggle to dark theme', async () => {
      const wrapper = mount(App)
      await flushPromises()

      const themeButton = wrapper.find('.theme-toggle')
      await themeButton.trigger('click')

      expect(wrapper.vm.isDarkTheme).toBe(true)
      expect(document.documentElement.classList.contains('dark-theme')).toBe(true)
    })

    it('should persist theme preference to localStorage', async () => {
      const wrapper = mount(App)
      await flushPromises()

      const themeButton = wrapper.find('.theme-toggle')
      await themeButton.trigger('click')

      expect(localStorage.setItem).toHaveBeenCalledWith('darkTheme', 'true')
    })

    it('should load theme preference from localStorage', async () => {
      localStorage.getItem.mockReturnValue('true')

      const wrapper = mount(App)
      await flushPromises()

      expect(wrapper.vm.isDarkTheme).toBe(true)
    })
  })

  describe('Transistor Operations', () => {
    it('should handle transistor saved event', async () => {
      const wrapper = mount(App)
      await flushPromises()

      // Reset mock to check if called again
      transistorApi.getAll.mockClear()

      await wrapper.vm.handleTransistorSaved()

      expect(transistorApi.getAll).toHaveBeenCalled()
      expect(wrapper.vm.currentView).toBe('search')
    })

    it('should handle transistor deletion', async () => {
      transistorApi.delete.mockResolvedValue({ message: 'Deleted' })

      const wrapper = mount(App)
      await flushPromises()

      transistorApi.getAll.mockClear()

      await wrapper.vm.handleTransistorDeleted('CREE_C3M0060065J')

      expect(transistorApi.delete).toHaveBeenCalledWith('CREE_C3M0060065J')
      expect(transistorApi.getAll).toHaveBeenCalled()
    })

    it('should handle deletion errors', async () => {
      transistorApi.delete.mockRejectedValue(new Error('Delete failed'))

      const wrapper = mount(App)
      await flushPromises()

      await wrapper.vm.handleTransistorDeleted('CREE_C3M0060065J')

      expect(console.error).toHaveBeenCalled()
    })

    it('should handle edit transistor', async () => {
      const wrapper = mount(App)
      await flushPromises()

      const transistor = mockTransistors[0]
      wrapper.vm.showEditForm(transistor)

      expect(wrapper.vm.selectedTransistor).toEqual(transistor)
      expect(wrapper.vm.currentView).toBe('create')
    })

    it('should clear selected transistor when creating new', async () => {
      const wrapper = mount(App)
      await flushPromises()

      wrapper.vm.selectedTransistor = mockTransistors[0]

      wrapper.vm.showCreate()

      expect(wrapper.vm.selectedTransistor).toBeNull()
      expect(wrapper.vm.currentView).toBe('create')
    })
  })

  describe('Inter-Component Communication', () => {
    it('should handle load to exporting', async () => {
      const wrapper = mount(App)
      await flushPromises()

      const transistor = mockTransistors[0]
      wrapper.vm.handleLoadToExporting(transistor)

      expect(wrapper.vm.selectedForExport).toEqual([transistor])
      expect(wrapper.vm.currentView).toBe('export')
    })

    it('should handle load to comparison', async () => {
      const wrapper = mount(App)
      await flushPromises()

      const transistor = mockTransistors[0]
      wrapper.vm.handleLoadToComparison(transistor)

      expect(wrapper.vm.currentView).toBe('compare')
    })

    it('should handle load to topology', async () => {
      const wrapper = mount(App)
      await flushPromises()

      const transistor = mockTransistors[0]
      wrapper.vm.handleLoadToTopology(transistor)

      expect(wrapper.vm.selectedForTopology).toEqual(transistor)
      expect(wrapper.vm.currentView).toBe('topology')
    })

    it('should handle transistor selected', async () => {
      const wrapper = mount(App)
      await flushPromises()

      const transistor = mockTransistors[0]
      wrapper.vm.handleTransistorSelected(transistor)

      expect(wrapper.vm.selectedTransistor).toEqual(transistor)
    })

    it('should handle open database search', async () => {
      const wrapper = mount(App)
      await flushPromises()

      wrapper.vm.currentView = 'export'
      wrapper.vm.handleOpenDatabaseSearch()

      expect(wrapper.vm.currentView).toBe('search')
    })
  })

  describe('Reactive State', () => {
    it('should update stats when transistors change', async () => {
      const wrapper = mount(App)
      await flushPromises()

      expect(wrapper.text()).toContain('Total Transistors: 3')

      // Simulate adding more transistors
      wrapper.vm.transistors.push(mockTransistors[0])
      await wrapper.vm.$nextTick()

      expect(wrapper.text()).toContain('Total Transistors: 4')
    })

    it('should display selected transistor name', async () => {
      const wrapper = mount(App)
      await flushPromises()

      wrapper.vm.selectedTransistor = mockTransistors[0]
      await wrapper.vm.$nextTick()

      expect(wrapper.text()).toContain('Selected: CREE_C3M0060065J')
    })
  })
})
