/**
 * Unit tests for SearchDatabase.vue component
 */
import { describe, it, expect, beforeEach, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import SearchDatabase from '../../components/SearchDatabase.vue'
import { mockTransistors } from '../fixtures/transistors'

describe('SearchDatabase.vue', () => {
  let wrapper

  beforeEach(() => {
    // Suppress console logs
    vi.spyOn(console, 'log').mockImplementation(() => {})
    vi.spyOn(console, 'warn').mockImplementation(() => {})

    wrapper = mount(SearchDatabase, {
      props: {
        transistors: mockTransistors
      }
    })
  })

  describe('Component Rendering', () => {
    it('should render the component', () => {
      expect(wrapper.exists()).toBe(true)
    })

    it('should display search header', () => {
      expect(wrapper.text()).toContain('Search Database')
    })

    it('should display transistor count', () => {
      expect(wrapper.text()).toContain('3 devices')
    })

    it('should render filter panel', () => {
      expect(wrapper.text()).toContain('Advanced Filters')
    })

    it('should render results table', () => {
      expect(wrapper.find('.search-results').exists()).toBe(true)
    })
  })

  describe('Props Handling', () => {
    it('should accept transistors prop', () => {
      expect(wrapper.props('transistors')).toEqual(mockTransistors)
    })

    it('should handle empty transistors array', async () => {
      await wrapper.setProps({ transistors: [] })

      expect(wrapper.text()).toContain('0 devices')
    })

    it('should handle undefined transistors gracefully', async () => {
      await wrapper.setProps({ transistors: undefined })

      expect(console.warn).toHaveBeenCalled()
      expect(wrapper.vm.filteredTransistors).toHaveLength(0)
    })

    it('should handle non-array transistors gracefully', async () => {
      await wrapper.setProps({ transistors: {} })

      expect(console.warn).toHaveBeenCalled()
      expect(wrapper.vm.filteredTransistors).toHaveLength(0)
    })
  })

  describe('Filtering', () => {
    describe('Name Filter', () => {
      it('should filter by name', async () => {
        wrapper.vm.filters.name.enabled = true
        wrapper.vm.filters.name.value = 'CREE'
        await wrapper.vm.$nextTick()

        const filtered = wrapper.vm.filteredTransistors
        expect(filtered).toHaveLength(1)
        expect(filtered[0].metadata.name).toContain('CREE')
      })

      it('should be case insensitive', async () => {
        wrapper.vm.filters.name.enabled = true
        wrapper.vm.filters.name.value = 'cree'
        await wrapper.vm.$nextTick()

        const filtered = wrapper.vm.filteredTransistors
        expect(filtered).toHaveLength(1)
      })

      it('should return no results for non-matching name', async () => {
        wrapper.vm.filters.name.enabled = true
        wrapper.vm.filters.name.value = 'NonExistent'
        await wrapper.vm.$nextTick()

        expect(wrapper.vm.filteredTransistors).toHaveLength(0)
      })
    })

    describe('Type Filter', () => {
      it('should filter by transistor type', async () => {
        wrapper.vm.filters.type.enabled = true
        wrapper.vm.filters.type.value = 'IGBT'
        await wrapper.vm.$nextTick()

        const filtered = wrapper.vm.filteredTransistors
        expect(filtered).toHaveLength(1)
        expect(filtered[0].metadata.type).toBe('IGBT')
      })

      it('should return all types when not enabled', async () => {
        wrapper.vm.filters.type.enabled = false
        wrapper.vm.filters.type.value = 'IGBT'
        await wrapper.vm.$nextTick()

        expect(wrapper.vm.filteredTransistors).toHaveLength(3)
      })
    })

    describe('Manufacturer Filter', () => {
      it('should filter by manufacturer', async () => {
        wrapper.vm.filters.manufacturer.enabled = true
        wrapper.vm.filters.manufacturer.value = 'Infineon'
        await wrapper.vm.$nextTick()

        const filtered = wrapper.vm.filteredTransistors
        expect(filtered).toHaveLength(1)
        expect(filtered[0].metadata.manufacturer).toBe('Infineon')
      })
    })

    describe('Voltage Range Filter', () => {
      it('should filter by voltage range', async () => {
        wrapper.vm.filters.v_abs_max.enabled = true
        wrapper.vm.filters.v_abs_max.min = 600
        wrapper.vm.filters.v_abs_max.max = 700
        await wrapper.vm.$nextTick()

        const filtered = wrapper.vm.filteredTransistors
        expect(filtered.length).toBeGreaterThan(0)
        filtered.forEach(t => {
          expect(t.electrical.v_abs_max).toBeGreaterThanOrEqual(600)
          expect(t.electrical.v_abs_max).toBeLessThanOrEqual(700)
        })
      })

      it('should handle min-only voltage filter', async () => {
        wrapper.vm.filters.v_abs_max.enabled = true
        wrapper.vm.filters.v_abs_max.min = 1000
        wrapper.vm.filters.v_abs_max.max = null
        await wrapper.vm.$nextTick()

        const filtered = wrapper.vm.filteredTransistors
        filtered.forEach(t => {
          expect(t.electrical.v_abs_max).toBeGreaterThanOrEqual(1000)
        })
      })

      it('should handle max-only voltage filter', async () => {
        wrapper.vm.filters.v_abs_max.enabled = true
        wrapper.vm.filters.v_abs_max.min = null
        wrapper.vm.filters.v_abs_max.max = 700
        await wrapper.vm.$nextTick()

        const filtered = wrapper.vm.filteredTransistors
        filtered.forEach(t => {
          expect(t.electrical.v_abs_max).toBeLessThanOrEqual(700)
        })
      })
    })

    describe('Multiple Filters', () => {
      it('should apply multiple filters simultaneously', async () => {
        wrapper.vm.filters.type.enabled = true
        wrapper.vm.filters.type.value = 'SiC-MOSFET'
        wrapper.vm.filters.v_abs_max.enabled = true
        wrapper.vm.filters.v_abs_max.min = 600
        wrapper.vm.filters.v_abs_max.max = 700
        await wrapper.vm.$nextTick()

        const filtered = wrapper.vm.filteredTransistors
        filtered.forEach(t => {
          expect(t.metadata.type).toBe('SiC-MOSFET')
          expect(t.electrical.v_abs_max).toBeGreaterThanOrEqual(600)
          expect(t.electrical.v_abs_max).toBeLessThanOrEqual(700)
        })
      })
    })
  })

  describe('Filter Options', () => {
    it('should compute available types', () => {
      const types = wrapper.vm.availableTypes
      expect(types).toContain('SiC-MOSFET')
      expect(types).toContain('IGBT')
      expect(types).toContain('GaN-Transistor')
    })

    it('should compute available manufacturers', () => {
      const manufacturers = wrapper.vm.availableManufacturers
      expect(manufacturers).toContain('CREE')
      expect(manufacturers).toContain('Infineon')
      expect(manufacturers).toContain('GaN Systems')
    })

    it('should compute available housing types', () => {
      const housingTypes = wrapper.vm.availableHousingTypes
      expect(housingTypes).toContain('TO-247-3')
      expect(housingTypes).toContain('PrimePACK3')
      expect(housingTypes).toContain('GaNPX')
    })
  })

  describe('Reset Filters', () => {
    it('should reset all filters', async () => {
      // Apply some filters
      wrapper.vm.filters.name.enabled = true
      wrapper.vm.filters.name.value = 'CREE'
      wrapper.vm.filters.type.enabled = true
      wrapper.vm.filters.type.value = 'SiC-MOSFET'

      // Reset
      const resetButton = wrapper.find('button:contains("Reset Filters")')
      if (resetButton.exists()) {
        await resetButton.trigger('click')
      } else {
        await wrapper.vm.resetFilters()
      }

      expect(wrapper.vm.filters.name.enabled).toBe(false)
      expect(wrapper.vm.filters.name.value).toBe('')
      expect(wrapper.vm.filters.type.enabled).toBe(false)
      expect(wrapper.vm.filters.type.value).toBe('')
    })
  })

  describe('View Mode', () => {
    it('should default to table view', () => {
      expect(wrapper.vm.viewMode).toBe('table')
    })

    it('should switch to card view', async () => {
      wrapper.vm.viewMode = 'cards'
      await wrapper.vm.$nextTick()

      expect(wrapper.vm.viewMode).toBe('cards')
    })
  })

  describe('Sorting', () => {
    it('should have default sort state', () => {
      expect(wrapper.vm.sortField).toBe('')
      expect(wrapper.vm.sortDirection).toBe('asc')
    })

    it('should allow setting sort field', async () => {
      wrapper.vm.sortField = 'name'
      await wrapper.vm.$nextTick()

      expect(wrapper.vm.sortField).toBe('name')
    })
  })

  describe('Pagination', () => {
    it('should have default pagination settings', () => {
      expect(wrapper.vm.currentPage).toBe(1)
      expect(wrapper.vm.itemsPerPage).toBe(20)
    })

    it('should calculate total pages correctly', () => {
      wrapper.vm.itemsPerPage = 2
      const totalPages = wrapper.vm.totalPages

      expect(totalPages).toBe(Math.ceil(3 / 2))
    })

    it('should paginate results', async () => {
      wrapper.vm.itemsPerPage = 2
      wrapper.vm.currentPage = 1
      await wrapper.vm.$nextTick()

      const paginated = wrapper.vm.paginatedTransistors
      expect(paginated).toHaveLength(2)
    })

    it('should show remaining items on last page', async () => {
      wrapper.vm.itemsPerPage = 2
      wrapper.vm.currentPage = 2
      await wrapper.vm.$nextTick()

      const paginated = wrapper.vm.paginatedTransistors
      expect(paginated).toHaveLength(1)
    })
  })

  describe('Events', () => {
    it('should emit load-to-exporting event', () => {
      const transistor = mockTransistors[0]
      wrapper.vm.$emit('load-to-exporting', transistor)

      expect(wrapper.emitted('load-to-exporting')).toBeTruthy()
      expect(wrapper.emitted('load-to-exporting')[0]).toEqual([transistor])
    })

    it('should emit load-to-comparison event', () => {
      const transistor = mockTransistors[0]
      wrapper.vm.$emit('load-to-comparison', transistor)

      expect(wrapper.emitted('load-to-comparison')).toBeTruthy()
      expect(wrapper.emitted('load-to-comparison')[0]).toEqual([transistor])
    })

    it('should emit load-to-topology event', () => {
      const transistor = mockTransistors[0]
      wrapper.vm.$emit('load-to-topology', transistor)

      expect(wrapper.emitted('load-to-topology')).toBeTruthy()
      expect(wrapper.emitted('load-to-topology')[0]).toEqual([transistor])
    })
  })

  describe('Export Results', () => {
    it('should have export button', () => {
      const exportButton = wrapper.find('button:contains("Export Results")')
      expect(exportButton.exists()).toBe(true)
    })

    it('should disable export when no results', async () => {
      await wrapper.setProps({ transistors: [] })

      const exportButton = wrapper.find('button:contains("Export Results")')
      expect(exportButton.attributes('disabled')).toBeDefined()
    })
  })
})
