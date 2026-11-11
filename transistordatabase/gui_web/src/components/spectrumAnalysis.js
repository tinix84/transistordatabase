import { finiteDecimal } from './analysisFiles.js'

export function defaultSpectrumAnalysis() {
  return { maskEnabled: false, maskCsv: 'frequency_hz,limit_peak\n', maskSource: '', interpolation: 'exact',
    bandsEnabled: false, bandsCsv: 'name,lower_hz,upper_hz,endpoints\n' }
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

function coordinate(value, label) {
  const number = finiteDecimal(value, label)
  if (number < 0) throw new Error(`${label} must be nonnegative`)
  if (decimalIdentity(value) !== decimalIdentity(number)) throw new Error(`${label} exceeds the accepted FFT coordinate precision`)
  return number
}

function rows(text, header, maximum, label) {
  if (text.includes('"')) throw new Error(`${label} uses unquoted comma-separated fields`)
  const lines = text.split(/\r?\n/).filter(line => line.trim())
  if (lines.length < 2 || lines.length > maximum + 1) throw new Error(`${label} needs 1 to ${maximum} data rows`)
  if (lines[0].split(',').map(value => value.trim()).join(',') !== header.join(',')) throw new Error(`${label} requires header ${header.join(',')}`)
  return lines.slice(1).map((line, index) => {
    const values = line.split(',').map(value => value.trim())
    if (values.length !== header.length) throw new Error(`${label} row ${index + 2} does not match its header`)
    return values
  })
}

export function spectrumAnalysisRequest(input) {
  const result = {}
  if (input.maskEnabled) {
    if (!input.maskSource.trim()) throw new Error('Declare the current mask source')
    const parsed = rows(input.maskCsv, ['frequency_hz', 'limit_peak'], 5001, 'Spectrum mask')
    const frequencies = parsed.map(([frequency], index) => coordinate(frequency, `Mask frequency row ${index + 2}`))
    if (frequencies.some((value, index) => index > 0 && value <= frequencies[index - 1])) throw new Error('Mask frequencies must increase strictly without duplicate coordinates')
    const limits = parsed.map(([, limit], index) => {
      if (limit === '' || limit === 'null') return null
      const value = finiteDecimal(limit, `Mask limit row ${index + 2}`)
      if (value < 0) throw new Error('Mask peak limits must be nonnegative')
      return value
    })
    result.mask = { unit: 'A_peak', frequency_hz: frequencies, limit_peak: limits,
      interpolation: input.interpolation, source: input.maskSource.trim() }
  }
  if (input.bandsEnabled) {
    const names = new Set()
    result.bands = rows(input.bandsCsv, ['name', 'lower_hz', 'upper_hz', 'endpoints'], 32, 'Frequency bands').map(([name, low, high, endpoints]) => {
      if (!name || names.has(name)) throw new Error('Band names must be nonblank and unique')
      names.add(name)
      const lower = coordinate(low, 'Band lower frequency'), upper = coordinate(high, 'Band upper frequency')
      if (lower > upper) throw new Error('Band lower frequency must not exceed its upper frequency')
      if (!['closed', 'open', 'left-closed', 'right-closed'].includes(endpoints)) throw new Error('Select a declared band endpoint convention')
      return { name, lower_hz: lower, upper_hz: upper, endpoints }
    })
  }
  return result
}
