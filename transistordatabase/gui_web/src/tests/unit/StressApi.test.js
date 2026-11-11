import { afterEach, describe, expect, it } from 'vitest'
import MockAdapter from 'axios-mock-adapter'
import api, { stressApi } from '../../services/api.js'

const mock = new MockAdapter(api)
afterEach(() => mock.reset())

describe('stress review response transport', () => {
  it('preserves exact JSON number text and negative zero for evidence downloads', async () => {
    const rawJson = '{"inputs":{"voltage_v":[400.0,-0.0]},"evidence":{"input_sha256":"hash"}}'
    mock.onPost('/api/stress/calculate').reply(200, rawJson)
    const result = await stressApi.calculate({ source: 'fixture' })
    expect(result.rawJson).toBe(rawJson)
    expect(result.data.inputs.voltage_v[0]).toBe(400)
    expect(Object.is(result.data.inputs.voltage_v[1], -0)).toBe(true)
  })

  it('retains structured API validation errors when successful responses use raw text', async () => {
    mock.onPost('/api/stress/calculate').reply(422, '{"detail":"Time must increase strictly"}')
    await expect(stressApi.calculate({})).rejects.toMatchObject({ response: { data: { detail: 'Time must increase strictly' } } })
  })
})
