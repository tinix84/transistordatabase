import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { mount } from '@vue/test-utils'

const { ChartMock } = vi.hoisted(() => {
  const ChartMock = vi.fn(function (context, config) {
    this.data = config.data; this.options = config.options
    this.destroy = vi.fn(); this.update = vi.fn(); this.draw = vi.fn(); this.stop = vi.fn()
    this.config = config
  })
  ChartMock.register = vi.fn()
  return { ChartMock }
})
vi.mock('chart.js/auto', () => ({ Chart: ChartMock, registerables: [] }))
vi.mock('chartjs-plugin-zoom', () => ({ default: {} }))
import CurvePlotter from '../../components/CurvePlotter.vue'

describe('optional compact scientific y-axis tick display', () => {
  let wrapper
  beforeEach(() => { ChartMock.mockClear(); vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({}) })
  afterEach(() => { wrapper?.unmount(); vi.restoreAllMocks() })
  it('keeps the default tick configuration for all other plots', () => {
    wrapper = mount(CurvePlotter, { props: { datasets: [{ label: 'Other data', data: [{ x: 1, y: 1e-15 }] }] } })
    expect(wrapper.vm.chart.options.scales.y).not.toHaveProperty('ticks')
    expect(wrapper.props('cursorX')).toBe(null)
    expect(wrapper.props('yMin')).toBe(null)
    expect(wrapper.props('yMax')).toBe(null)
  })
  it('retains explicit full-source Y bounds when only one sparse sample is inside the selected time range', async () => {
    const points = [{ x: 0, y: 2 }, { x: .001, y: 4 }, { x: .002, y: 6 }]
    wrapper = mount(CurvePlotter, { props: { yMin: 2, yMax: 6, datasets: [{ label: 'Current', data: points }] } })
    await wrapper.setProps({ xMin: .0005, xMax: .0015 })
    expect(wrapper.vm.chart.options.scales.y.min).toBe(2)
    expect(wrapper.vm.chart.options.scales.y.max).toBe(6)
    expect(wrapper.vm.chart.data.datasets[0].data).toEqual(points)
    await wrapper.setProps({ yMin: null, yMax: null })
    expect(wrapper.vm.chart.options.scales.y.min).toBeUndefined()
    expect(wrapper.vm.chart.options.scales.y.max).toBeUndefined()
    expect(wrapper.vm.chart.update).toHaveBeenCalledWith('none')
  })
  it('draws the optional shared cursor at its actual SI x coordinate without changing raw points', async () => {
    const points = [{ x: 0, y: 2 }, { x: .001, y: 4 }]
    wrapper = mount(CurvePlotter, { props: { cursorX: .001, datasets: [{ label: 'Current', data: points }] } })
    const ctx = Object.fromEntries(['save', 'restore', 'beginPath', 'setLineDash', 'moveTo', 'lineTo', 'stroke'].map(key => [key, vi.fn()]))
    const getPixelForValue = vi.fn().mockReturnValue(120)
    const chart = { ctx, chartArea: { left: 40, right: 300, top: 15, bottom: 200 }, scales: { x: { getPixelForValue } } }
    wrapper.vm.chart.config.plugins.find(plugin => plugin.id === 'sharedTimeCursor').afterDatasetsDraw(chart)
    expect(getPixelForValue).toHaveBeenCalledWith(.001)
    expect(ctx.moveTo).toHaveBeenCalledWith(120, 15)
    expect(ctx.lineTo).toHaveBeenCalledWith(120, 200)
    expect(ctx.stroke).toHaveBeenCalledOnce()
    expect(ctx.strokeStyle).toBe('#7c3aed')
    await wrapper.setProps({ cursorX: 0 })
    expect(wrapper.vm.chart.draw).toHaveBeenCalledOnce()
    expect(wrapper.vm.chart.data.datasets[0].data).toEqual(points)
    await wrapper.setProps({ cursorX: null })
    ctx.stroke.mockClear(); wrapper.vm.drawCursor(chart)
    expect(ctx.stroke).not.toHaveBeenCalled()
  })
  it('formats residual ticks without changing raw samples or tooltip values', () => {
    const points = [{ x: 4, y: -1.234567890123456e-15 }, { x: 5, y: 2.345678901234567e-15 }]
    const before = structuredClone(points)
    wrapper = mount(CurvePlotter, { props: { compactScientificYTicks: true,
      datasets: [{ label: 'Residual', data: points }] } })
    const chart = wrapper.vm.chart
    const tick = chart.options.scales.y.ticks.callback
    expect(tick(-1.234567890123456e-15)).toBe('-1.23e-15')
    expect(tick(2.345678901234567e-15)).toBe('2.35e-15')
    expect(tick(1.23456789e9)).toBe('1.23e+9')
    expect(tick(12.3456789)).toBe('12.3')
    expect(tick(0)).toBe('0')
    expect(tick(-0)).toBe('0')
    expect(points).toEqual(before)
    expect(chart.data.datasets[0].data).toEqual(before)
    const tooltip = chart.options.plugins.tooltip.callbacks.label({ dataset: { label: 'Residual' }, parsed: points[0] })
    expect(tooltip).toBe(`Residual: (${points[0].x.toPrecision(6)}, ${points[0].y.toPrecision(6)})`)
    expect(chart.data.datasets[0].data[0].y).toBe(-1.234567890123456e-15)
  })
})
