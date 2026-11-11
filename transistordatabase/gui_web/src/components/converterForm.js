// Form serialization only. All loss calculations and topology mappings are Python-owned.
export function finite(value, label) {
  if (value === null || value === undefined || typeof value === 'boolean' || String(value).trim() === '' || !Number.isFinite(Number(value))) throw new Error(`${label} needs a finite number`)
  return Number(value)
}
export function serializeConditions(values) {
  return Object.fromEntries(Object.entries(values).map(([key, value]) => [key, value === null ? null : finite(value, `Known condition ${key}`)]))
}
export function vector(text, label) {
  if (!text.trim()) throw new Error(`${label} needs data`)
  if (/[,;]\s*[,;]/.test(text) || /^[,;]|[,;]$/.test(text.trim())) throw new Error(`${label} has a missing numeric value`)
  return text.trim().split(/[,;\s]+/).map(value => finite(value, label))
}
export function curve(energy = false) {
  const conditions = { t_j_c: 25, v_g_v: null, r_g_ohm: null, v_supply_v: energy ? 200 : null }
  return { enabled: true, mode: 'recorded', selected: false,
    source: 'Editable algebraic hand fixture; not device measurements',
    offset_v: 0, resistance_ohm: .1, pairs: energy ? '0,0.001\n1000,0.001' : '0,2\n1000,2',
    recorded: { ...conditions }, requested: { ...conditions } }
}
export function roleModel() {
  return { source: 'Editable role hand fixture; not a selected device', conductionOmit: false, conductionReason: '',
    positive: curve(), negative: curve(), eventMode: 'gate', threshold: .5, onMask: '', offMask: '', eventReason: '',
    onOmit: false, offOmit: false, onReason: '', offReason: '',
    onPositive: curve(true), onNegative: curve(true), offPositive: curve(true), offNegative: curve(true) }
}
export function serializeCurve(value, energy = false) {
  if (!value.enabled) return null
  const common = { mode: value.mode, source: value.source }
  if (value.mode === 'linear-drop-estimate') return { ...common, selected: value.selected,
    offset_v: finite(value.offset_v, 'Threshold'), resistance_ohm: finite(value.resistance_ohm, 'Resistance'), conditions: serializeConditions(value.recorded) }
  const pairs = value.pairs.trim().split(/\r?\n/).filter(line => line.trim()).map((line, i) => {
    if (/[,;]\s*[,;]/.test(line) || /^[,;]|[,;]$/.test(line.trim())) throw new Error(`Curve row ${i + 1} has a missing value`)
    const cells = line.trim().split(/[,;\t ]+/)
    if (cells.length !== 2) throw new Error(`Curve row ${i + 1} needs exactly two values`)
    return cells.map(cell => finite(cell, 'Curve value'))
  })
  if (!pairs.length) throw new Error('Curve needs paired values')
  return { ...common, ...(value.mode !== 'recorded' ? { selected: value.selected } : {}),
    current_a: pairs.map(pair => pair[0]), [energy ? 'energy_j' : 'drop_v']: pairs.map(pair => pair[1]),
    recorded_conditions: serializeConditions(value.recorded), requested_conditions: serializeConditions(value.requested) }
}
export function serializeRole(value, sourceOmission = false) {
  const omitted = reason => ({ omit: true, omission_reason: reason })
  const switching = side => value[side + 'Omit'] ? omitted(value[side + 'Reason'])
    : { positive: serializeCurve(value[side + 'Positive'], true), negative: serializeCurve(value[side + 'Negative'], true) }
  const base = { source: value.source,
    conduction: value.conductionOmit ? omitted(value.conductionReason)
      : { positive: serializeCurve(value.positive), negative: serializeCurve(value.negative) } }
  if (sourceOmission) return base
  function mask(text) {
    return vector(text, 'Event mask').map(number => {
      if (number !== 0 && number !== 1) throw new Error('Event masks require 0 or 1')
      return Boolean(number)
    })
  }
  const events = value.eventMode === 'masks' ? { mode: 'masks', turn_on: mask(value.onMask), turn_off: mask(value.offMask) }
    : value.eventMode === 'omitted' ? { mode: 'omitted', omission_reason: value.eventReason }
      : { mode: 'gate', threshold_v: finite(value.threshold, 'Gate threshold') }
  return { ...base,
    switching: { turn_on: switching('on'), turn_off: switching('off') },
    events }
}
