import { afterEach, expect, it } from 'vitest'
import MockAdapter from 'axios-mock-adapter'
import api, { assemblyApi } from '../../services/api.js'
const mock = new MockAdapter(api)
afterEach(() => mock.reset())
it('retains exact reverse-sharing response bytes and null unsupported values', async () => {
  const rawJson = '{"inputs":{"current_a":-0.0},"result":{"channel_current_a":null},"evidence":{"input_sha256":"hash"}}'
  mock.onPost('/api/assemblies/reverse-sharing').reply(200, rawJson)
  const result = await assemblyApi.reverseSharing({})
  expect(result.rawJson).toBe(rawJson); expect(Object.is(result.data.inputs.current_a, -0)).toBe(true); expect(result.data.result.channel_current_a).toBe(null)
})
it('retains structured reverse-sharing validation errors', async () => {
  mock.onPost('/api/assemblies/reverse-sharing').reply(422, '{"detail":[{"msg":"Gate bias is incompatible"}]}')
  await expect(assemblyApi.reverseSharing({})).rejects.toMatchObject({ response: { data: { detail: [{ msg: 'Gate bias is incompatible' }] } } })
})
