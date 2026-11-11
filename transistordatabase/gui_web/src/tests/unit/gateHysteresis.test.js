import { describe, expect, it } from 'vitest'
import { defaultHysteresis, hysteresisRequest, gateDigitizationColumns } from '../../components/gateHysteresis.js'

describe('explicit hysteresis form settings', () => {
  it('keeps infer-only outside-band initialization distinct from declared on or off', () => {
    const form = defaultHysteresis()
    expect(hysteresisRequest(form)).toEqual({ thresholds: { scale: 'absolute', low_v: 2.5, high_v: 7.5 } })
    expect(hysteresisRequest({ ...form, initial: 'off' }).initial_state).toBe(false)
    expect(hysteresisRequest({ ...form, initial: 'on' }).initial_state).toBe(true)
    expect(form.initial).toBe('auto')
  })
  it('requires explicit finite calibrated bounds for relative fractions', () => {
    const form = { ...defaultHysteresis(), scale: 'relative', low: '.25', high: '.75' }
    expect(() => hysteresisRequest(form)).toThrow('Calibrated minimum')
    expect(hysteresisRequest({ ...form, minimum: '-5', maximum: '15' })).toEqual({ thresholds: {
      scale: 'relative', low_fraction: .25, high_fraction: .75, minimum_v: -5, maximum_v: 15 } })
    expect(() => hysteresisRequest({ ...form, minimum: '1', maximum: '1' })).toThrow('below the maximum')
  })
  it('rejects cleared, underflowing, reversed and out-of-domain threshold values', () => {
    const form = defaultHysteresis()
    for (const patch of [{ low: '' }, { low: '1e-999' }, { high: 'Infinity' }, { low: '8' },
      { scale: 'relative', low: '-.1', high: '.75', minimum: '0', maximum: '10' },
      { scale: 'relative', low: '.25', high: '1.1', minimum: '0', maximum: '10' }]) {
      expect(() => hysteresisRequest({ ...form, ...patch })).toThrow()
    }
  })
  it('checks the original relative fraction lexeme before rounding at the upper boundary', () => {
    const form = { ...defaultHysteresis(), scale: 'relative', low: '.25', minimum: '0', maximum: '10' }
    for (const high of ['1.00000000000000000001', '10.00000000000000000001e-1', '100000000000000000001e-20']) {
      expect(() => hysteresisRequest({ ...form, high })).toThrow('before numeric rounding')
    }
    for (const high of ['1', '1.0000000000', '100000000000000000000e-20', '.999999999999999999999']) {
      expect(hysteresisRequest({ ...form, high }).thresholds.high_fraction).toBe(1)
    }
    expect(() => hysteresisRequest({ ...form, low: '1e-999', high: '.75' })).toThrow('too small')
  })
  it('exports constant-size scalar policy metadata without repeating sampled vectors per row', () => {
    const evidence = { method: 'strict-hysteresis-v1', low_v: -2, high_v: 2,
      initial_state: false, initial_state_source: 'declared', states: Array(10000).fill(false), turn_on_brackets: [] }
    const columns = gateDigitizationColumns(evidence)
    expect(columns.low_v).toBe(-2)
    expect(columns.initial_state).toBe(false)
    expect(Object.values(columns).some(Array.isArray)).toBe(false)
    expect(columns).not.toHaveProperty('states')
    expect(gateDigitizationColumns(null)).toEqual({})
  })
})
