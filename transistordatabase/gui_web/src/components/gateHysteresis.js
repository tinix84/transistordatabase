import { finiteDecimal } from './analysisFiles.js'

export function defaultHysteresis() {
  return { scale: 'absolute', low: '2.5', high: '7.5', minimum: '', maximum: '', initial: 'auto' }
}

function fractionDecimal(value, name) {
  const number = finiteDecimal(value, name)
  if (number < 0 || number > 1) throw new Error(`${name} must be between 0 and 1`)
  if (number === 1) {
    // Inspect the validated original decimal before constructing the request.
    const [mantissa, exponent = '0'] = String(value).trim().replace(/^\+/, '').split(/[eE]/)
    const [whole, decimals = ''] = mantissa.split('.')
    const digits = (whole + decimals).replace(/^0+/, '')
    const order = BigInt(digits.length) + BigInt(exponent) - BigInt(decimals.length)
    if (order > 1n || (order === 1n && !/^10*$/.test(digits))) {
      throw new Error(`${name} must be between 0 and 1 before numeric rounding`)
    }
  }
  return number
}

export function hysteresisRequest(input) {
  const parse = input.scale === 'relative' ? fractionDecimal : finiteDecimal
  const low = parse(input.low, 'Hysteresis low threshold')
  const high = parse(input.high, 'Hysteresis high threshold')
  if (!(low < high)) throw new Error('Hysteresis low threshold must be below the high threshold')
  let thresholds
  if (input.scale === 'absolute') thresholds = { scale: 'absolute', low_v: low, high_v: high }
  else if (input.scale === 'relative') {
    if (low < 0 || high > 1) throw new Error('Relative hysteresis fractions must be between 0 and 1')
    const minimum = finiteDecimal(input.minimum, 'Calibrated minimum gate voltage')
    const maximum = finiteDecimal(input.maximum, 'Calibrated maximum gate voltage')
    if (!(minimum < maximum)) throw new Error('Calibrated minimum gate voltage must be below the maximum')
    thresholds = { scale: 'relative', low_fraction: low, high_fraction: high, minimum_v: minimum, maximum_v: maximum }
  } else throw new Error('Select absolute or relative hysteresis thresholds')
  if (!['auto', 'off', 'on'].includes(input.initial)) throw new Error('Select the initial hysteresis state')
  return { thresholds, ...(input.initial === 'auto' ? {} : { initial_state: input.initial === 'on' }) }
}

export function gateDigitizationColumns(evidence) {
  if (!evidence) return {}
  return { digitization_method: evidence.method, threshold_scale: evidence.threshold_scale,
    low_v: evidence.low_v, high_v: evidence.high_v, calibration_min_v: evidence.calibration_min_v,
    calibration_max_v: evidence.calibration_max_v, initial_state: evidence.initial_state,
    initial_state_source: evidence.initial_state_source, digitization_semantics: evidence.semantics }
}
