// Display geometry uses original observations only. It does not estimate energy.
export const POINT_LIMIT = 12000
export const FACE_GRID_LIMIT = 65536
export const defaultCamera = () => ({ yaw: -.65, pitch: .48, zoom: 1 })
const finite = value => typeof value === 'number' && Number.isFinite(value)
const key = point => JSON.stringify([point.current_a, point.voltage_v])
const axis = values => [...new Set(values.filter(finite))].sort((a, b) => a - b)

export function sourceObservations(result, mode) {
  if (mode === 'xyz') return result.rows.map(row => ({ ...row,
    source: result.source.source, source_kind: result.source.source_kind,
    conditions: result.conditions, source_sha256: result.source.sha256,
    available: row.cell_status === 'available',
  }))
  const selected = new Set(result.slice_diagnostics.filter(item => item.status === 'selected').map(item => item.slice_id))
  const slices = result.source_slices.filter(slice => selected.has(slice.id))
  const domains = slices.map(slice => ({ slice, min: slice.current_a.reduce((a, b) => Math.min(a, b), Infinity),
    max: slice.current_a.reduce((a, b) => Math.max(a, b), -Infinity) }))
  const byVoltage = new Map()
  for (const domain of domains) {
    const voltage = domain.slice.voltage_v
    if (!byVoltage.has(voltage)) byVoltage.set(voltage, [])
    byVoltage.get(voltage).push(domain)
  }
  return slices.flatMap(slice => slice.current_a.map((current, index) => {
    // Competing bounded slices make a coordinate ambiguous, even at different knots.
    let covering = 0
    for (const domain of byVoltage.get(slice.voltage_v)) if (current >= domain.min && current <= domain.max) covering++
    return { id: `${slice.id}:point-${index + 1}`, slice_id: slice.id, current_a: current,
      voltage_v: slice.voltage_v, energy_j: slice.energy_j[index], source: slice.source,
      source_kind: slice.source_kind, conditions: slice.conditions,
      cell_status: covering === 1 ? 'available' : 'ambiguous', available: covering === 1 }
  }))
}

export function buildScene(observations) {
  const currents = axis(observations.map(point => point.current_a))
  const voltages = axis(observations.map(point => point.voltage_v))
  const points = observations.filter(point => finite(point.current_a) && finite(point.voltage_v) && finite(point.energy_j))
  const capped = points.length > POINT_LIMIT
  const faceCapped = currents.length * voltages.length > FACE_GRID_LIMIT
  const groups = new Map()
  if (!capped && !faceCapped) for (const point of observations) { const id = key(point); if (!groups.has(id)) groups.set(id, []); groups.get(id).push(point) }
  const lookup = (i, v) => {
    const group = groups.get(JSON.stringify([i, v]))
    return group?.length === 1 && group[0].available && finite(group[0].energy_j) ? group[0] : null
  }
  const faces = []
  if (!capped && !faceCapped) for (let y = 0; y < voltages.length - 1; y++) for (let x = 0; x < currents.length - 1; x++) {
    const corners = [lookup(currents[x], voltages[y]), lookup(currents[x + 1], voltages[y]),
      lookup(currents[x + 1], voltages[y + 1]), lookup(currents[x], voltages[y + 1])]
    if (corners.every(Boolean) &&
      (!corners[0].slice_id || corners[0].slice_id === corners[1].slice_id) &&
      (!corners[2].slice_id || corners[2].slice_id === corners[3].slice_id)) {
      faces.push([corners[0], corners[1], corners[2]], [corners[0], corners[2], corners[3]])
    }
  }
  return { observations, points, faces, currents, voltages, energies: axis(points.map(point => point.energy_j)), capped, faceCapped,
    renderedPoints: capped ? [] : points }
}

// Subtract first to preserve close coordinates. Scale only for an overflowing span.
export function normalize(value, values) {
  if (!values.length || values[0] === values[values.length - 1]) return 0
  const lo = values[0], hi = values[values.length - 1], scale = Math.max(Math.abs(lo), Math.abs(hi))
  const span = hi - lo
  if (Number.isFinite(span)) return 2 * ((value - lo) / span) - 1
  return 2 * ((value / scale - lo / scale) / (hi / scale - lo / scale)) - 1
}

export function projectNormalized(x, y, z, camera, width, height) {
  const c = Math.cos(camera.yaw), s = Math.sin(camera.yaw), cp = Math.cos(camera.pitch), sp = Math.sin(camera.pitch)
  const horizontal = c * x - s * y, back = s * x + c * y
  const vertical = cp * z - sp * back, depth = sp * z + cp * back
  const scale = Math.min(width, height) * .23 * camera.zoom
  return { x: width / 2 + horizontal * scale, y: height / 2 - vertical * scale, depth }
}

export function projectScene(scene, camera, width, height) {
  const project = point => ({ ...point, ...projectNormalized(normalize(point.current_a, scene.currents),
    normalize(point.voltage_v, scene.voltages), normalize(point.energy_j, scene.energies), camera, width, height) })
  const points = scene.renderedPoints.map(project).sort((a, b) => a.depth - b.depth)
  const byId = new Map(points.map(point => [point.id, point]))
  const faces = scene.faces.map(face => face.map(point => byId.get(point.id)))
    .sort((a, b) => a.reduce((sum, point) => sum + point.depth, 0) - b.reduce((sum, point) => sum + point.depth, 0))
  return { points, faces }
}

export function pickPoint(points, x, y, radius = 10) {
  let best = null, distance = radius * radius
  for (const point of points) {
    const next = (point.x - x) ** 2 + (point.y - y) ** 2
    if (next < distance || (next === distance && (!best || point.depth >= best.depth))) { best = point; distance = next }
  }
  return best
}

export function moveCamera(camera, key) {
  const next = { ...camera }
  if (key === 'ArrowLeft') next.yaw -= .15
  else if (key === 'ArrowRight') next.yaw += .15
  else if (key === 'ArrowUp') next.pitch = Math.min(1.4, next.pitch + .12)
  else if (key === 'ArrowDown') next.pitch = Math.max(-1.4, next.pitch - .12)
  else if (key === '+' || key === '=') next.zoom = Math.min(1.7, next.zoom * 1.15)
  else if (key === '-') next.zoom = Math.max(.5, next.zoom / 1.15)
  else if (key.toLowerCase() === 'r' || key === 'Home') return defaultCamera()
  return next
}
