import { describe, expect, it } from 'vitest'
import { returnedPlotBounds } from '../../components/returnedPlotBounds.js'

describe('full-source presentation bounds', () => {
  it('retains sparse source endpoints without evaluating values at the selected range boundaries', () => {
    const datasets = [{ data: [{ x: 0, y: 2 }, { x: .001, y: 4 }, { x: .002, y: 6 }] }]
    const original = JSON.stringify(datasets)
    expect(returnedPlotBounds(datasets)).toEqual({ min: 2, max: 6 })
    expect(JSON.stringify(datasets)).toBe(original)
  })
  it('adds a power baseline only when actual returned values exist', () => {
    expect(returnedPlotBounds([{ data: [] }], true)).toEqual({ min: null, max: null })
    expect(returnedPlotBounds([{ data: [{ x: 1, y: 5 }, { x: 2, y: 7 }] }], true)).toEqual({ min: 0, max: 7 })
  })
  it.each([0, 3, -3, Number.MAX_VALUE, -Number.MAX_VALUE, Number.MIN_VALUE])('pads the finite constant %s without changing its point', value => {
    const points = [{ data: [{ x: 1, y: value }] }]
    const bounds = returnedPlotBounds(points)
    expect(Number.isFinite(bounds.min) && Number.isFinite(bounds.max)).toBe(true)
    expect(bounds.min).toBeLessThan(bounds.max)
    expect(bounds.min).toBeLessThanOrEqual(value); expect(bounds.max).toBeGreaterThanOrEqual(value)
    expect(points[0].data[0].y).toBe(value)
  })
})
