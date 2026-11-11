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
