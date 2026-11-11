// Axis bounds use existing displayed values. They do not change source points.
export function returnedPlotBounds(datasets, zeroBaseline = false) {
  let min = Infinity, max = -Infinity
  for (const dataset of datasets) for (const point of dataset.data) {
    if (!Number.isFinite(point.y)) continue
    min = Math.min(min, point.y); max = Math.max(max, point.y)
  }
  if (!Number.isFinite(min)) return { min: null, max: null }
  if (zeroBaseline) { min = Math.min(0, min); max = Math.max(0, max) }
  if (min === max) {
    const value = min, padding = Math.abs(value) * .05 || 1
    min = Number.isFinite(value - padding) ? value - padding : value
    max = Number.isFinite(value + padding) ? value + padding : value
    if (min === max) {
      if (value > 0) min = value / 2
      else if (value < 0) max = value / 2
    }
    if (zeroBaseline && value === 0) min = 0
  }
  return { min, max }
}
