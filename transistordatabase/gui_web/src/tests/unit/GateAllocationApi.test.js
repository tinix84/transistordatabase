import { afterEach, expect, it } from 'vitest'
import MockAdapter from 'axios-mock-adapter'
import api, { waveformApi } from '../../services/api.js'
const mock = new MockAdapter(api)
afterEach(() => mock.reset())
it('keeps raw allocation evidence text and negative zero', async () => {
  const rawJson = '{"inputs":{"current_a":[-0.0,4.0]},"evidence":{"input_sha256":"hash"}}'
  mock.onPost('/api/waveform/gate-allocation').reply(200, rawJson)
  const result = await waveformApi.allocate({})
  expect(result.rawJson).toBe(rawJson); expect(Object.is(result.data.inputs.current_a[0], -0)).toBe(true)
})
it('retains structured allocation validation errors', async () => {
  mock.onPost('/api/waveform/gate-allocation').reply(422, '{"detail":"Binary gate state is invalid"}')
  await expect(waveformApi.allocate({})).rejects.toMatchObject({ response: { data: { detail: 'Binary gate state is invalid' } } })
})
it('sends the original advanced waveform JSON and explicit media type without reserialization', async () => {
  const raw = '{"spectrum":{"mask":{"frequency_hz":[1.0000000000000000001],"limit_peak":[-1e-400]}}}'
  mock.onPost('/api/waveform/analyze').reply(config => {
    expect(config.data).toBe(raw)
    expect(config.headers['Content-Type']).toBe('application/json')
    return [422, { detail: 'Original numeric precision rejected' }]
  })
  await expect(waveformApi.analyze(raw)).rejects.toMatchObject({ response: { data: { detail: 'Original numeric precision rejected' } } })
})
it('keeps object waveform requests in the ordinary JSON transport', async () => {
  const request = { spectrum: { window: 'rectangular', detrend: 'none' } }
  mock.onPost('/api/waveform/analyze').reply(config => {
    expect(JSON.parse(config.data)).toEqual(request)
    return [200, { exact: true }]
  })
  expect(await waveformApi.analyze(request)).toEqual({ exact: true })
})
