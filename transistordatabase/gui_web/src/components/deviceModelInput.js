import { finiteDecimal } from './analysisFiles.js'

export const modelFields = {
  ron: ['t_j_c'], diode: ['current_a', 't_j_c', 'v_g_v'],
  energy: ['current_a', 'voltage_v', 't_j_c', 'r_g_ohm'],
}
export const coordinateLabels = { t_j_c: 'Junction temperature (°C)', current_a: 'Current (A)', voltage_v: 'Voltage (V)', v_g_v: 'Gate voltage (V)', r_g_ohm: 'Gate resistance (ohm)' }
export const modelHeader = family => ['id', 'source', 'series', 'split', ...modelFields[family], 'value']
export function nullableDecimal(value, label) {
  return String(value).trim() === '' || String(value).trim() === 'null' ? null : finiteDecimal(value, label)
}

// Keep quoted source labels intact. Numeric fields use the same strict decimal parser as the analysis tools.
function csvRecords(text) {
  const records = [], cells = []
  let value = '', quoted = false, closed = false
  for (let index = 0; index < text.length; index++) {
    const char = text[index]
    if (quoted) {
      if (char === '"' && text[index + 1] === '"') { value += '"'; index++ }
      else if (char === '"') { quoted = false; closed = true }
      else value += char
    } else if (char === '"') {
      if (value.trim() || closed) throw new Error('Use quotes around a complete CSV cell.')
      quoted = true; value = ''
    } else if (char === ',' || char === '\n' || char === '\r') {
      cells.push(value.trim()); value = ''; closed = false
      if (char !== ',') {
        if (char === '\r' && text[index + 1] === '\n') index++
        if (cells.some(cell => cell !== '')) records.push(cells.splice(0))
        else cells.length = 0
      }
    } else {
      if (closed && char.trim()) throw new Error('Unexpected text after a quoted CSV cell.')
      value += char
    }
  }
  if (quoted) throw new Error('Close the quoted CSV cell.')
  cells.push(value.trim())
  if (cells.some(cell => cell !== '')) records.push(cells)
  return records
}

export function parseObservations(text, family, commonSource, commonSeries) {
  const header = modelHeader(family), records = csvRecords(text)
  if (records.shift()?.join(',') !== header.join(',')) throw new Error(`Use the exact CSV header: ${header.join(',')}.`)
  if (!records.length) throw new Error('Supply observation rows.')
  const ids = new Set()
  return records.map((cells, index) => {
    const label = `CSV row ${index + 2}`
    if (cells.length !== header.length) throw new Error(`${label} does not match the CSV header.`)
    const row = Object.fromEntries(header.map((key, col) => [key, cells[col]]))
    if (!row.id || ids.has(row.id)) throw new Error(`${label} requires a unique, nonblank observation ID.`)
    ids.add(row.id)
    const source = row.source || commonSource.trim(), series = row.series || commonSeries.trim()
    if (!source || !series) throw new Error(`${label} requires source and series labels, or their common labels.`)
    if (!['train', 'holdout'].includes(row.split)) throw new Error(`${label} split must be train or holdout.`)
    return { id: row.id, source, series, split: row.split,
      coordinates: Object.fromEntries(modelFields[family].map(key => [key, key === 'v_g_v' || (family === 'diode' && key === 't_j_c') ? nullableDecimal(row[key], `${label} ${key}`) : finiteDecimal(row[key], `${label} ${key}`)])),
      value: finiteDecimal(row.value, `${label} value`) }
  })
}

export function fittingFixture(family) {
  const source = 'Independent hand fixture. No measured device data.', series = 'Algebra verification'
  const rows = family === 'ron'
    ? ['r1,,,train,25,.010', 'r2,,,train,75,.0125', 'r3,,,train,125,.015', 'rh,,,holdout,50,.01150']
    : family === 'diode'
      ? ['d1,,,train,1,25,null,.55', 'd2,,,train,3,25,null,.65', 'd3,,,train,5,25,null,.75', 'dh,,,holdout,2,25,null,.61']
      : ['i1,,,train,10,400,25,4,.001', 'nominal,,,train,20,400,25,4,.002', 'i3,,,train,30,400,25,4,.003',
        'v1,,,train,20,200,25,4,.001', 'v3,,,train,20,600,25,4,.003',
        't2,,,train,20,400,75,4,.0025', 't3,,,train,20,400,125,4,.003',
        'g2,,,train,20,400,25,6,.0022', 'g3,,,train,20,400,25,8,.0024',
        'eh,,,holdout,15,500,75,6,.002678125']
  return { source, series, csv: [modelHeader(family).join(','), ...rows].join('\n') }
}
