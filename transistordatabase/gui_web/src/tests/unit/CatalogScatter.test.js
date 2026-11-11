import { afterEach, describe, expect, it, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
const { charts } = vi.hoisted(() => ({ charts: [] }))
vi.mock('chart.js', () => ({
  registerables: [],
  Chart: class {
    static register() {}
    constructor(canvas, config) { this.config = config; this.data = config.data; this.destroy = vi.fn(); this.update = vi.fn(); charts.push(this) }
  },
}))
vi.mock('chartjs-plugin-zoom', () => ({ default: {} }))
import CatalogScatter from '../../components/CatalogScatter.vue'
describe('relative catalog plots', () => {
  let wrapper
  afterEach(() => { wrapper?.unmount(); charts.length = 0 })
  it('plots supplied relative values, reports unavailable membership, and updates selection without reconstructing charts', async () => {
    wrapper = mount(CatalogScatter, { props: { title: 'Fixture', points: [
      { id: 'A1', source_id: 'A', row: 2, technology: 'SiC', pareto: null, metrics: { voltage_v: 4, current_a: 8 }, normalization: { voltage_v: { relative: 2 }, current_a: { relative: 4 } } },
    ], xKey: 'voltage_v', yKey: 'current_a', relative: true,
    metrics: { voltage_v: { label: 'Voltage', unit: 'V' }, current_a: { label: 'Current', unit: 'A' } }, matchedCount: 2, excludedCount: 1 } })
    await flushPromises()
    expect(charts[0].data.datasets[0].data[0]).toMatchObject({ x: 2, y: 4 })
    expect(charts[0].config.options.scales.x.title.text).toContain('relative to positive minimum')
    const point = charts[0].data.datasets[0].data[0]
    expect(charts[0].config.options.plugins.tooltip.callbacks.afterLabel({ raw: point })).toContain('Pareto unavailable: chosen objectives')
    await wrapper.setProps({ selectedId: 'A1' })
    expect(charts).toHaveLength(1)
    expect(charts[0].update).toHaveBeenCalledWith('none')
    await wrapper.setProps({ relative: false }); await flushPromises()
    expect(charts[0].destroy).toHaveBeenCalledOnce()
    expect(charts[1].data.datasets[0].data[0]).toMatchObject({ x: 4, y: 8 })
    wrapper.unmount(); wrapper = null
    expect(charts[1].destroy).toHaveBeenCalledOnce()
  })
  it('retains SI axes when engineering-unit scaling would overflow', async () => {
    wrapper = mount(CatalogScatter, { props: { points: [{ id: 'huge', technology: 'SiC', metrics: { qg_c: 1e308, voltage_v: 650 } }], xKey: 'qg_c', yKey: 'voltage_v', metrics: { qg_c: { label: 'Charge', unit: 'C' }, voltage_v: { label: 'Voltage', unit: 'V' } } } })
    await flushPromises()
    expect(charts[0].data.datasets[0].data[0].x).toBe(1e308)
    expect(charts[0].config.options.scales.x.title.text).toBe('Charge (C)')
  })
  it('plots recorded zero on linear axes, excludes missing inputs, and hides zero only for a logarithmic axis', async () => {
    wrapper = mount(CatalogScatter, { props: { points: [
      { id: 'zero', technology: 'IGBT', metrics: { thermal_current_a: 0, voltage_v: 650 } },
      { id: 'missing', technology: 'IGBT', metrics: { thermal_current_a: null, voltage_v: 650 } },
    ], xKey: 'thermal_current_a', yKey: 'voltage_v', metrics: { thermal_current_a: { label: 'Thermal current screening estimate', unit: 'A' }, voltage_v: { label: 'Voltage', unit: 'V' } } } })
    await flushPromises()
    expect(charts[0].data.datasets[0].data.map(p => p.record.id)).toEqual(['zero'])
    expect(wrapper.text()).toContain('1 points')
    await wrapper.findAll('input[type="checkbox"]')[0].setValue(true); await flushPromises()
    expect(charts.at(-1).data.datasets).toEqual([])
    expect(wrapper.text()).toContain('All complete pairs fall outside the selected axis scale.')
  })
  it('counts a hidden normalized zero separately from a missing relative value', async () => {
    const points = [0, 1, null].map((relative, i) => ({
      id: String(i), source_id: 'source', metrics: { x: i, y: 1 },
      normalization: { x: { relative }, y: { relative: 1 } },
    }))
    wrapper = mount(CatalogScatter, { props: {
      points, relative: true, xKey: 'x', yKey: 'y', matchedCount: 3, excludedCount: 0,
      metrics: { x: { label: 'Current', unit: 'A' }, y: { label: 'Voltage', unit: 'V' } },
    } })
    await flushPromises()
    expect(charts.at(-1).data.datasets[0].data.map(point => point.x)).toEqual([0, 1])
    await wrapper.findAll('input[type="checkbox"]')[0].setValue(true)
    await flushPromises()
    expect(charts.at(-1).data.datasets[0].data.map(point => point.x)).toEqual([1])
    expect(wrapper.get('.coverage').text()).toContain('1 unavailable relative pairs')
    expect(wrapper.get('.coverage').text()).toContain('1 pairs outside the selected axis scale')
  })
  it('keeps source colors stable across populations and retains technology in the tooltip', async () => {
    const points = ['A', 'B'].map(source_id => ({ id: source_id, source_id, row: 2, technology: 'SiC', metrics: { voltage_v: 650, current_a: 30 } }))
    wrapper = mount(CatalogScatter, { props: { points, xKey: 'voltage_v', yKey: 'current_a', metrics: { voltage_v: { label: 'Voltage', unit: 'V' }, current_a: { label: 'Current', unit: 'A' } } } })
    await flushPromises()
    const sourceB = charts[0].data.datasets[1]
    expect(sourceB.label).toBe('B')
    expect(charts[0].config.options.plugins.tooltip.callbacks.afterLabel({ raw: sourceB.data[0] })).toContain('SiC')
    await wrapper.setProps({ points: [points[1]] }); await flushPromises()
    expect(charts.at(-1).data.datasets[0].backgroundColor).toBe(sourceB.backgroundColor)
    expect(wrapper.get('.axis-labels').text()).toBe('X: Voltage (V) · Y: Current (A)')
  })
})
