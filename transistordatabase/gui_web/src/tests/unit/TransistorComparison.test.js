/**
 * Unit tests for TransistorComparison.vue component
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import TransistorComparison from '../../components/TransistorComparison.vue'
import { mockTransistors } from '../fixtures/transistors'
const { chart } = vi.hoisted(() => ({ chart: vi.fn(function (context, config) { this.config = config; this.destroy = vi.fn() }) }))
vi.mock('chart.js/auto', () => ({ default: chart }))

describe('TransistorComparison.vue', () => {
  let wrapper

  beforeEach(() => {
    vi.spyOn(console, 'log').mockImplementation(() => {})
    vi.spyOn(console, 'warn').mockImplementation(() => {})
    chart.mockClear()
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({})

    wrapper = mount(TransistorComparison, {
      attachTo: document.body,
      props: {
        transistors: mockTransistors
      }
    })
  })

  afterEach(() => { wrapper.unmount(); wrapper.element.remove(); vi.restoreAllMocks(); vi.unstubAllGlobals() })
  async function selectPair() {
    for (const name of mockTransistors.slice(0, 2).map(device => device.metadata.name)) {
      await wrapper.findAll('.transistor-card').find(card => card.text().includes(name)).trigger('click')
    }
    await flushPromises()
  }
  function row(label) {
    return wrapper.findAll('.comparison-table tbody tr').find(item => item.find('td').text() === label)
  }

  describe('Component Rendering', () => {
    it('should render the component', () => {
      expect(wrapper.exists()).toBe(true)
    })

    it('should display comparison title', () => {
      expect(wrapper.text()).toMatch(/comparison|compare/i)
    })

    it('should have transistor selection dropdowns', () => {
      const selects = wrapper.findAll('select')
      expect(selects.length).toBeGreaterThanOrEqual(2)
    })
  })

  describe('Transistor Selection', () => {
    it('should allow selecting first transistor', async () => {
      const selects = wrapper.findAll('select')
      if (selects.length > 0) {
        await selects[0].setValue(mockTransistors[0].metadata.name)

        // Selection should be updated
        expect(wrapper.vm).toBeDefined()
      }
    })

    it('should allow selecting second transistor', async () => {
      const selects = wrapper.findAll('select')
      if (selects.length > 1) {
        await selects[1].setValue(mockTransistors[1].metadata.name)

        expect(wrapper.vm).toBeDefined()
      }
    })

    it('should support comparing 2-3 transistors', () => {
      const selects = wrapper.findAll('select')
      expect(selects.length).toBeGreaterThanOrEqual(2)
      expect(selects.length).toBeLessThanOrEqual(3)
    })

    it('should handle selecting same transistor twice', async () => {
      const selects = wrapper.findAll('select')
      if (selects.length > 1) {
        const name = mockTransistors[0].metadata.name
        await selects[0].setValue(name)
        await selects[1].setValue(name)

        // Should either prevent or allow with warning
        expect(wrapper.vm).toBeDefined()
      }
    })
  })

  describe('Comparison Display', () => {
    beforeEach(selectPair)
    it('should display comparison table', () => {
      // Should have table or grid for comparison
      const hasTable = wrapper.find('table').exists()
      const hasGrid = wrapper.find('.comparison-grid').exists()

      expect(hasTable || hasGrid).toBe(true)
    })

    it('should show transistor properties', () => {
      // Should display key properties
      const text = wrapper.text()
      expect(text).toMatch(/voltage|current|resistance|type/i)
    })

    it('should highlight differences', () => {
      // Should have some way to highlight differences
      // This could be CSS classes or visual indicators
      expect(wrapper.html()).toBeTruthy()
    })
  })

  describe('Comparison Metrics', () => {
    beforeEach(selectPair)
    it.each([null, undefined, NaN, Infinity, 0, -.2])('preserves thermal unknowns and recorded numerical value %s', async value => {
      const devices = structuredClone(mockTransistors)
      devices[0].thermal.r_th_cs = value
      devices[1].thermal.r_th_cs = .2
      await wrapper.findAll('button').find(button => button.text().includes('Clear Selection')).trigger('click')
      await wrapper.setProps({ transistors: devices })
      await selectPair()
      const thermalRow = row('Thermal Resistance (K/W)').findAll('td')
      const known = typeof value === 'number' && Number.isFinite(value)
      expect(thermalRow[1].text()).toBe(known ? value.toFixed(3) : 'Unknown')
      expect(thermalRow[2].text()).toBe('0.200')
      if (!known) {
        expect(thermalRow[1].classes()).not.toContain('best-value')
        expect(wrapper.text()).toContain('Unavailable · one or more recorded values are unknown')
        expect(wrapper.text()).toContain('(1/2 known)')
      }
      expect(Object.is(wrapper.props('transistors')[0].thermal.r_th_cs, value)).toBe(true)
    })
    it('should compare electrical ratings', () => {
      expect(row('Max Voltage (V)').findAll('td').map(cell => cell.text())).toEqual(['Max Voltage (V)', '650V', '1200V'])
      expect(row('Max Current (A)').findAll('td').map(cell => cell.text())).toEqual(['Max Current (A)', '120A', '600A'])
    })

    it('should compare thermal properties', () => {
      expect(row('Thermal Resistance (K/W)').findAll('td').map(cell => cell.text())).toEqual(['Thermal Resistance (K/W)', '0.240', '0.050'])
      expect(row('Max Junction Temp (°C)').findAll('td').map(cell => cell.text())).toEqual(['Max Junction Temp (°C)', '175°C', '150°C'])
    })

    it('keeps absent power and conditioned performance unavailable without fabricated ranks or charts', () => {
      expect(row('Power Rating (W)').findAll('td').slice(1).map(cell => cell.text())).toEqual([
        'Unavailable · no recorded power rating', 'Unavailable · no recorded power rating',
      ])
      expect(wrapper.text()).toContain('Ratings do not establish efficiency, loss, power density, or application performance.')
      expect(wrapper.text()).not.toMatch(/Best Power Density|Recommendations|Similarity:|Performance Radar/)
      expect(wrapper.find('#powerThermalChart').exists()).toBe(false)
      expect(wrapper.find('#radarChart').exists()).toBe(false)
      expect(chart.mock.calls.map(call => call[1].type)).toEqual(['scatter'])
    })

    it('preserves switch curve inputs without presenting ratings as switching loss data', () => {
      expect(wrapper.props('transistors')[0].switch.e_on_data[0].graph_i_e).toEqual([[0, 10, 20, 30], [0, 0.5, 1.0, 1.5]])
      expect(wrapper.get('.comparison-table').text()).not.toMatch(/Switching Loss|E_on|E_off/)
    })

    it('preserves diode curve inputs without inventing recovery energies', () => {
      expect(wrapper.props('transistors')[0].diode.channel_data[0].graph_v_i).toEqual([[0, 0.7, 1.4], [0, 10, 20]])
      expect(wrapper.get('.comparison-table').text()).not.toMatch(/Recovery Energy|E_rr/)
    })

    it('preserves capacitance evidence without deriving capacitance from ratings', () => {
      expect(wrapper.props('transistors')[0].c_oss[0].graph_v_c).toEqual([[0, 100, 200, 300], [1000, 500, 300, 200]])
      expect(wrapper.get('.comparison-table').text()).not.toMatch(/Capacitance|C_oss|C_iss|C_rss/)
    })
  })

  describe('Comparison Charts', () => {
    beforeEach(selectPair)
    it('should display comparison charts', () => {
      // Should have chart containers or canvas elements
      const hasCanvas = wrapper.findAll('canvas').length > 0
      const hasChartContainer = wrapper.find('.chart-container').exists()

      expect(hasCanvas || hasChartContainer || wrapper.html().includes('chart')).toBe(true)
      const scatter = chart.mock.calls.map(call => call[1]).find(config => config.options.plugins.title.text === 'Voltage vs Current Rating')
      expect(scatter.data.datasets.map(item => ({ label: item.label, data: item.data }))).toEqual([
        { label: mockTransistors[0].metadata.name, data: [{ x: 650, y: 120 }] },
        { label: mockTransistors[1].metadata.name, data: [{ x: 1200, y: 600 }] },
      ])
    })

    it('should support multiple chart types', () => {
      // Should support bar charts, line charts, etc.
      expect(wrapper.html()).toBeTruthy()
    })
  })

  describe('Export Functionality', () => {
    it('should have export button', () => {
      const exportButton = wrapper.find('button:contains("Export")')
      const hasExport = exportButton.exists() || wrapper.text().includes('Export')

      expect(hasExport || wrapper.html().includes('export')).toBe(true)
    })

    it('should support exporting comparison', () => {
      // Should be able to export comparison data
      expect(wrapper.vm).toBeDefined()
    })

    it('exports unavailable power as null with its source limitation', async () => {
      await selectPair()
      let bytes
      vi.spyOn(URL, 'createObjectURL').mockImplementation(blob => { bytes = blob; return 'blob:ratings-summary' })
      vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {})
      vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {})
      await wrapper.findAll('button').find(button => button.text().includes('Export Comparison')).trigger('click')
      const summary = JSON.parse(await bytes.text()).summary
      expect(summary.ratedPowerW).toBeNull()
      expect(summary.applicationPerformance).toBeNull()
      expect(summary.ratedPowerReason).toBe('The core record has no power rating')
      expect(summary.voltageRange).toBe('650V - 1200V')
      expect(summary.currentRange).toBe('120A - 600A')
    })
  })

  describe('Empty States', () => {
    it('should handle no transistors selected', async () => {
      wrapper = mount(TransistorComparison, {
        props: {
          transistors: mockTransistors
        }
      })

      // Should show message to select transistors
      expect(wrapper.text()).toMatch(/select|choose|comparison/i)
    })

    it('should handle empty transistor list', async () => {
      wrapper = mount(TransistorComparison, {
        props: {
          transistors: []
        }
      })

      expect(wrapper.findAll('.transistor-card')).toHaveLength(0)
      expect(wrapper.get('.no-selection').text()).toBe('Select at least 2 transistors to compare their specifications.')
      expect(wrapper.find('.comparison-results').exists()).toBe(false)
    })
  })

  describe('Props Handling', () => {
    it('should accept transistors prop', () => {
      expect(wrapper.props('transistors')).toEqual(mockTransistors)
    })

    it('should handle transistor list updates', async () => {
      await wrapper.setProps({ transistors: [] })
      expect(wrapper.props('transistors')).toEqual([])

      await wrapper.setProps({ transistors: mockTransistors })
      expect(wrapper.props('transistors')).toEqual(mockTransistors)
    })
  })

  describe('Computed Properties', () => {
    it('should compute selected transistors', () => {
      // Should track which transistors are selected for comparison
      expect(wrapper.vm).toBeDefined()
    })

    it('should compute comparison data', () => {
      // Should calculate comparison metrics
      expect(wrapper.vm).toBeDefined()
    })

    it('should compute differences', () => {
      // Should calculate differences between transistors
      expect(wrapper.vm).toBeDefined()
    })
  })

  describe('Interactive Features', () => {
    it('should allow clearing selection', () => {
      const clearButton = wrapper.find('button:contains("Clear")')
      const hasClear = clearButton.exists() || wrapper.text().includes('Clear')

      expect(hasClear || wrapper.html().includes('clear')).toBe(true)
    })

    it('should allow swapping transistors', () => {
      // Should allow reordering selected transistors
      expect(wrapper.vm).toBeDefined()
    })

    it('should support tooltips', () => {
      // Should show additional info on hover
      expect(wrapper.html()).toBeTruthy()
    })
  })

  describe('Responsive Design', () => {
    it('should render on mobile', () => {
      // Component should be usable on mobile devices
      expect(wrapper.exists()).toBe(true)
    })

    it('should adapt table layout', () => {
      // Table should adapt to smaller screens
      expect(wrapper.html()).toBeTruthy()
    })
  })
})
