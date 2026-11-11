import { describe, expect, it } from 'vitest'
import { buildScene, normalize, pickPoint, projectScene, sourceObservations } from '../../components/energyScene.js'

const observation = (id, current_a, voltage_v, energy_j) => ({
  id, current_a, voltage_v, energy_j, available: energy_j !== null,
})

describe('independent energy geometry checks', () => {
  it.each([
    [1.5, 1.5000000000000002, 1.5000000000000004],
    [Number.MAX_VALUE - 2 ** 972, Number.MAX_VALUE - 2 ** 971, Number.MAX_VALUE],
    [0, Number.MIN_VALUE, 2 * Number.MIN_VALUE],
    [-Number.MAX_VALUE, 0, Number.MAX_VALUE],
  ])('places the exact midpoint of %s to the third value at the origin', (low, middle, high) => {
    expect(normalize(middle, [low, middle, high])).toBe(0)
    expect(normalize(low, [low, middle, high])).toBe(-1)
    expect(normalize(high, [low, middle, high])).toBe(1)
  })

  it('keeps an otherwise unrepresented axis coordinate from an unknown source observation', () => {
    const rows = [observation('a', 1, 20, 0), observation('b', 9, 20, 2),
      observation('c', 1, 40, 3), observation('d', 9, 40, 4)]
    expect(buildScene(rows).faces).toHaveLength(2)
    for (const barrier of [observation('missing energy', 5, 20, null), observation('missing energy', 1, 30, null)]) {
      const scene = buildScene([...rows, barrier])
      expect(scene.faces).toHaveLength(0)
      expect(scene.renderedPoints.map(point => point.id)).toEqual(rows.map(point => point.id))
    }
  })

  it('joins only the two supported current intervals of unsorted disjoint prepared slices', () => {
    const slice = (id, voltage_v, current_a) => ({
      id, voltage_v, current_a, energy_j: current_a.map(current => current / 10),
      conditions: { t_j_c: 25 }, source: id, source_kind: 'manual',
    })
    const slices = [slice('full', 100, [10, 0, 9, 1]), slice('right', 200, [10, 9]), slice('left', 200, [1, 0])]
    const result = {
      source_slices: slices,
      slice_diagnostics: slices.map(({ id }) => ({ slice_id: id, status: 'selected' })),
      currents_a: [0, 10], voltages_v: [100, 200],
    }
    const scene = buildScene(sourceObservations(result, 'prepared'))
    expect(scene.renderedPoints).toHaveLength(8)
    expect(scene.faces.map(face => [...new Set(face.map(point => point.current_a))].sort((a, b) => a - b)))
      .toEqual([[0, 1], [0, 1], [9, 10], [9, 10]])
  })

  it('resolves a known quarter-turn screen position on a portrait canvas after zoom', () => {
    const scene = buildScene([observation('low', 0, 200, 0), observation('pick', 8, 600, 2)])
    const camera = { yaw: Math.PI / 2, pitch: 0, zoom: 1.25 }
    const { points } = projectScene(scene, camera, 360, 430)
    // Under a quarter turn, positive voltage moves left and positive energy moves up.
    const offset = 360 * 0.23 * 1.25
    expect(pickPoint(points, 180 - offset, 215 - offset, 0.001)?.id).toBe('pick')
    expect(pickPoint(points, 180 + offset, 215 + offset, 0.001)?.id).toBe('low')
  })
})
