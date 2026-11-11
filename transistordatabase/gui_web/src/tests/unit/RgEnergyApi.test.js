import { afterEach, expect, it } from 'vitest'
import MockAdapter from 'axios-mock-adapter'
import api, { rgEnergyApi } from '../../services/api.js'
const mock = new MockAdapter(api)
afterEach(() => mock.reset())
it('loads a repository inventory through the encoded device locator', async () => {
  mock.onGet('/api/transistors/device%20one/rg-energy').reply(200, { device_id: 'device one', curves: [] })
  expect(await rgEnergyApi.inventory('device one')).toEqual({ device_id: 'device one', curves: [] })
})
it('retains exact HTTP response text and zero without reserialization', async () => {
  const rawJson = '{"value":0.0,"unit":"J","conditions":{"v_g":null,"v_g_off":0,"method":"unknown"}}'
  mock.onPost('/api/transistors/sample/rg-energy').reply(200, rawJson)
  const result = await rgEnergyApi.calculate('sample', {})
  expect(result.rawJson).toBe(rawJson); expect(result.data.value).toBe(0); expect(result.data.conditions).toEqual({ v_g: null, v_g_off: 0, method: 'unknown' })
})
it('retains core validation diagnostics in rejected raw HTTP responses', async () => {
  mock.onPost('/api/transistors/sample/rg-energy').reply(422, '{"detail":"Multiple inverse crossings"}')
  await expect(rgEnergyApi.calculate('sample', {})).rejects.toMatchObject({ response: { status: 422, data: { detail: 'Multiple inverse crossings' } } })
})
