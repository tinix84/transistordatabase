import { serializeConditions } from './converterForm.js'

export function model(energy = false, enabled = true) {
  const conditions = { t_j_c: 25, v_g_v: null, r_g_ohm: null, v_supply_v: energy ? 400 : null }
  return { enabled, mode: energy ? 'recorded' : 'linear-drop-estimate', selected: false,
    source: 'Editable hand fixture assumption', offset_v: 1, resistance_ohm: .1,
    pairs: energy ? '0,0\n10,0.001' : '0,0\n10,1',
    recorded: { ...conditions }, requested: { ...conditions } }
}
export function finite(value, label) {
  const number = typeof value === 'number' ? value : Number(value)
  if (value === '' || !Number.isFinite(number)) throw new Error(`${label} needs a finite number`)
  return number
}
export function parsePairs(text) {
  return text.trim().split(/\r?\n/).filter(line => line.trim()).map((line, index) => {
    const values = line.trim().split(/[,;\t ]+/)
    if (values.length !== 2) throw new Error(`Curve row ${index + 1} needs exactly two values`)
    return values.map(value => finite(value, `Curve row ${index + 1}`))
  })
}
export function branch(value, energy = false) {
  if (!value.enabled) return null
  const common = { mode: value.mode, source: value.source }
  if (value.mode === 'linear-drop-estimate') return { ...common, selected: value.selected,
    offset_v: finite(value.offset_v, 'Threshold'), resistance_ohm: finite(value.resistance_ohm, 'Resistance'), conditions: serializeConditions(value.recorded) }
  const pairs = parsePairs(value.pairs)
  return { ...common, ...(value.mode !== 'recorded' ? { selected: value.selected } : {}),
    current_a: pairs.map(pair => pair[0]), [energy ? 'energy_j' : 'drop_v']: pairs.map(pair => pair[1]),
    recorded_conditions: serializeConditions(value.recorded), requested_conditions: serializeConditions(value.requested) }
}
