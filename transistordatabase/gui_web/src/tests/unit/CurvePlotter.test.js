import { afterEach, describe, expect, it, vi } from 'vitest'
import { mount } from '@vue/test-utils'

const { MockChart } = vi.hoisted(() => {
  const MockChart = vi.fn(function () {
    this.destroy = vi.fn()
  })
  MockChart.register = vi.fn()
  return { MockChart }
})

vi.mock('chart.js/auto', () => ({ Chart: MockChart, registerables: [] }))
vi.mock('chartjs-plugin-zoom', () => ({ default: {} }))

import CurvePlotter from '../../components/CurvePlotter.vue'

describe('CurvePlotter tooltip formatting', () => {
  let wrapper

  afterEach(() => {
    wrapper?.unmount()
    vi.restoreAllMocks()
    MockChart.mockClear()
  })

  it('formats microsecond-scaled DPT samples with readable precision', () => {
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({})

    wrapper = mount(CurvePlotter, {
      props: {
        title: 'Turn-on Vds',
        datasets: [{
          label: 'Turn-on Vds',
          data: [{ x: 1, y: 210 }],
          borderColor: '#2563eb'
        }],
        xLabel: 'Time [µs]',
        xValueDecimals: 6,
        yLabel: 'Vds [V]',
        showLegend: false
      }
    })

    const options = MockChart.mock.calls[0][1].options
    expect(options.scales.x.title.text).toBe('Time [µs]')
    const formatLabel = options.plugins.tooltip.callbacks.label
    expect(formatLabel({
      dataset: { label: 'Turn-on Vds' },
      parsed: { x: -0.079605, y: 450 }
    })).toBe('Turn-on Vds: (-0.079605, 450.000)')
    expect(formatLabel({
      dataset: { label: 'Turn-on Vds' },
      parsed: { x: -0.079445, y: 460 }
    })).toBe('Turn-on Vds: (-0.079445, 460.000)')
  })
})
