export function finiteDecimal(value, name) {
  const text = String(value).trim(), parsed = Number(text)
  if (!/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(text) || !Number.isFinite(parsed)) throw new Error(`${name} requires a finite decimal value.`)
  if (parsed === 0 && /[1-9]/.test(text.split(/[eE]/)[0])) throw new Error(`${name} is too small for a nonzero numeric value.`)
  return parsed
}

function decimalIdentity(value) {
  const text = String(value).trim(), negative = text.startsWith('-')
  const [mantissa, exponent = '0'] = text.replace(/^[+-]/, '').split(/[eE]/)
  const [whole, decimals = ''] = mantissa.split('.')
  let digits = (whole + decimals).replace(/^0+/, '')
  if (!digits) return '0'
  const trailing = digits.match(/0+$/)?.[0].length ?? 0
  digits = digits.slice(0, digits.length - trailing)
  return `${negative ? '-' : ''}${digits}e${BigInt(exponent) - BigInt(decimals.length) + BigInt(trailing)}`
}

export function binary64Decimal(value, name) {
  const number = finiteDecimal(value, name)
  if (decimalIdentity(value) !== decimalIdentity(number)) throw new Error(`${name} exceeds accepted binary64 precision.`)
  return number
}

export function binaryDecimal(value, name) {
  const number = finiteDecimal(value, name)
  const digits = String(value).trim().split(/[eE]/)[0].replace(/[+\-.]/g, '').replace(/^0+|0+$/g, '')
  if (number !== 0 && (number !== 1 || digits !== '1')) throw new Error(`${name} requires exact numeric 0/1 samples.`)
  return number
}

export function downloadAnalysis(value, filename, type = 'application/json') {
  const text = type === 'application/json' ? JSON.stringify(value, null, 2) : value
  const url = URL.createObjectURL(new Blob([text], { type }))
  const link = document.createElement('a')
  link.href = url; link.download = filename; link.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

export function analysisCsv(rows) {
  if (!rows.length) return ''
  const keys = Object.keys(rows[0])
  const quote = value => {
    const safe = typeof value === 'string' && /^\s*[=+\-@\t\r]/.test(value) ? `'${value}` : value
    return '"' + String(typeof safe === 'object' ? JSON.stringify(safe) : safe ?? '').replaceAll('"', '""') + '"'
  }
  return [keys.map(quote).join(','), ...rows.map(row => keys.map(key => quote(row[key])).join(','))].join('\r\n')
}

export function analysisError(error) {
  const detail = error.response?.data?.detail
  return typeof detail === 'string' ? detail : detail ? JSON.stringify(detail) : error.message
}

export function displayNumber(value) {
  return value === null || value === undefined ? 'Unavailable' : Number(value).toPrecision(6)
}

export function sameAnalysisEvidence(left, right) {
  if (left === right) return true
  if (left === null || right === null || typeof left !== 'object' || typeof right !== 'object' || Array.isArray(left) !== Array.isArray(right)) return false
  const keys = Object.keys(left), otherKeys = Object.keys(right)
  return keys.length === otherKeys.length && keys.every(key => Object.hasOwn(right, key) && sameAnalysisEvidence(left[key], right[key]))
}
